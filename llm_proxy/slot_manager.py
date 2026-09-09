"""Slot Manager — assign sessions to slots, save/restore KV cache."""

import asyncio
import hashlib
import re
import time
from pathlib import Path

import aiohttp

from llm_proxy.oplog import elapsed_ms, log_event, now


class SlotManager:
    """Manages slot ↔ session mapping with LRU eviction and KV cache persistence."""

    def __init__(self, cache_dir=None, *, checkpoint_interval_minutes=5,
                 checkpoint_max_requests=5):
        # model_name → {slot_id: {session, last_used, busy,
        #                            checkpointed, has_checkpoint,
        #                            requests_since_checkpoint,
        #                            last_checkpoint_at, live_dirty}}
        self._slots = {}
        self._conditions = {}
        # Models where save/restore is known to be unsupported
        self._no_persistence = set()
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir else None
        self.checkpoint_interval_seconds = (
            None
            if checkpoint_interval_minutes is None
            else float(checkpoint_interval_minutes) * 60.0
        )
        self.checkpoint_max_requests = (
            None
            if checkpoint_max_requests is None
            else int(checkpoint_max_requests)
        )

    def init_model(self, model_name, num_slots):
        """Register slots for a newly started model."""
        self._ensure_runtime_cache_dir(model_name)
        self._slots[model_name] = {
            i: {
                "session": None,
                "last_used": 0,
                "busy": False,
                # True only while the live slot exactly matches its latest
                # durable checkpoint. A throttled successful request clears it.
                "checkpointed": False,
                "has_checkpoint": False,
                "requests_since_checkpoint": 0,
                "last_checkpoint_at": None,
                # The live llama-server slot contains work from a request that
                # did not complete.  Keep it in memory so llama-server can use
                # its prompt checkpoints to reconcile the next prompt, but do
                # not persist it over the last committed disk checkpoint.
                "live_dirty": False,
            }
            for i in range(num_slots)
        }
        self._conditions[model_name] = asyncio.Condition()

    def remove_model(self, model_name):
        """Remove slot tracking for a stopped model."""
        self._slots.pop(model_name, None)
        self._conditions.pop(model_name, None)
        self._no_persistence.discard(model_name)

    def get_slot(self, model_name, session_id):
        """Find slot currently holding this session, or None."""
        slots = self._slots.get(model_name, {})
        for slot_id, info in slots.items():
            if info["session"] == session_id:
                return slot_id
        return None

    def active_sessions(self):
        """Return model_name -> set(session_id) currently assigned to slots."""
        active = {}
        for model_name, slots in self._slots.items():
            sessions = {
                str(info["session"])
                for info in slots.values()
                if info.get("session") is not None
            }
            if sessions:
                active[model_name] = sessions
        return active

    def active_models(self):
        """Return model names with registered slots."""
        return set(self._slots)

    async def acquire(self, model_name, session_id, port):
        """Acquire an exclusive slot lease for a session. Returns slot_id.

        If the session is already loaded in an idle slot, uses it.
        If a free idle slot is available, restores the session into it.
        Otherwise waits for an idle slot, then evicts the LRU session.
        A None session_id acquires one shared transient slot without restoring
        transient state. If no transient slot exists, it may evict the LRU idle
        persistent session; concurrent transient requests wait for that slot.
        """
        slots = self._slots.get(model_name)
        if not slots:
            log_event("slot acquire", model=model_name, session=_session_label(session_id), slot=0, ms=0, reason="untracked")
            return 0

        started = now()
        transient = session_id is None
        condition = self._conditions.setdefault(model_name, asyncio.Condition())
        async with condition:
            while True:
                # Already in a slot?
                if not transient:
                    existing = self.get_slot(model_name, session_id)
                    if existing is not None:
                        info = slots[existing]
                        if info["busy"]:
                            await condition.wait()
                            continue
                        info["busy"] = True
                        info["last_used"] = time.time()
                        log_event(
                            "slot acquire",
                            model=model_name,
                            session=_session_label(session_id),
                            slot=existing,
                            ms=elapsed_ms(started),
                            reason="hit",
                        )
                        return existing

                if transient and any(info["session"] is None and info["busy"] for info in slots.values()):
                    await condition.wait()
                    continue

                # Free idle slot?
                for slot_id, info in slots.items():
                    if info["session"] is None and not info["busy"]:
                        info["busy"] = True
                        info["session"] = session_id
                        info["last_used"] = time.time()
                        if not transient:
                            restored = await self._restore(
                                model_name, port, slot_id, session_id
                            )
                            self._record_restore(info, restored)
                        else:
                            self._reset_checkpoint_state(info)
                        info["live_dirty"] = False
                        log_event(
                            "slot acquire",
                            model=model_name,
                            session=_session_label(session_id),
                            slot=slot_id,
                            ms=elapsed_ms(started),
                            reason="free",
                        )
                        return slot_id

                idle_slots = [slot_id for slot_id, info in slots.items() if not info["busy"]]
                if not idle_slots:
                    await condition.wait()
                    continue

                # Evict LRU among idle slots only.
                lru_slot = min(idle_slots, key=lambda s: slots[s]["last_used"])
                info = slots[lru_slot]
                old_session = info["session"]
                info["busy"] = True

                if old_session and info["live_dirty"]:
                    log_event(
                        "slot discard dirty live state",
                        model=model_name,
                        session=_session_label(old_session),
                        slot=lru_slot,
                        checkpointed=info["checkpointed"],
                    )
                if (
                    old_session
                    and not info["checkpointed"]
                    and not info["live_dirty"]
                ):
                    await self._checkpoint(
                        model_name, port, lru_slot, old_session
                    )

                self._reset_checkpoint_state(info)
                if not transient:
                    restored = await self._restore(
                        model_name, port, lru_slot, session_id
                    )
                    self._record_restore(info, restored)
                info["live_dirty"] = False
                info["session"] = session_id
                info["last_used"] = time.time()
                log_event(
                    "slot acquire",
                    model=model_name,
                    session=_session_label(session_id),
                    slot=lru_slot,
                    evicted=_session_label(old_session),
                    ms=elapsed_ms(started),
                    reason="evict",
                )
                return lru_slot

    async def assign(self, model_name, session_id, port):
        """Backward-compatible alias. Callers must release the returned slot."""
        return await self.acquire(model_name, session_id, port)

    async def release(self, model_name, slot_id):
        """Release a slot lease after the backend request has fully completed."""
        slots = self._slots.get(model_name)
        if not slots or slot_id not in slots:
            return
        condition = self._conditions.setdefault(model_name, asyncio.Condition())
        async with condition:
            started = now()
            slots[slot_id]["busy"] = False
            slots[slot_id]["last_used"] = time.time()
            condition.notify_all()
            log_event("slot release", model=model_name, slot=slot_id, ms=elapsed_ms(started))

    async def finalize_request(self, model_name, port, slot_id, *, success):
        """Commit one request or retain its live recovery state, then release.

        llama-server mutates a slot while evaluating a request.  A failed or
        disconnected request must not become the persisted state for the
        session.  It must remain in the live slot, however: SWA and
        hybrid/recurrent runtimes keep rewind checkpoints only in memory.  A
        disk restore contains tokens and the terminal sequence state but not
        those checkpoints, so restoring here can force the next request to
        reprocess the entire prompt.

        Successful requests become eligible for a throttled disk checkpoint.
        The first clean state is persisted immediately; later states are saved
        when either the time or request-count limit is reached. Failed requests
        mark only the live slot dirty, so eviction and shutdown preserve the
        previous committed checkpoint instead of saving unfinished state.
        """
        slots = self._slots.get(model_name)
        info = slots.get(slot_id) if slots else None
        session_id = info.get("session") if info else None

        if session_id is not None and model_name not in self._no_persistence:
            if success:
                info["live_dirty"] = False
                info["checkpointed"] = False
                info["requests_since_checkpoint"] += 1
                reason = self._checkpoint_reason(info)
                if reason is not None:
                    log_event(
                        "slot checkpoint due",
                        model=model_name,
                        session=_session_label(session_id),
                        slot=slot_id,
                        requests=info["requests_since_checkpoint"],
                        reason=reason,
                    )
                    await self._checkpoint(
                        model_name, port, slot_id, session_id
                    )
                else:
                    age = time.monotonic() - info["last_checkpoint_at"]
                    log_event(
                        "slot checkpoint deferred",
                        model=model_name,
                        session=_session_label(session_id),
                        slot=slot_id,
                        requests=info["requests_since_checkpoint"],
                        age_s=round(age, 1),
                    )
            else:
                info["live_dirty"] = True
                log_event(
                    "slot retain dirty live state",
                    model=model_name,
                    session=_session_label(session_id),
                    slot=slot_id,
                    checkpointed=info["checkpointed"],
                )

        await self.release(model_name, slot_id)

    async def save_all(self, model_name, port):
        """Save all active slots for a model (before stopping)."""
        if model_name in self._no_persistence:
            return
        slots = self._slots.get(model_name, {})
        if not slots:
            return
        condition = self._conditions.setdefault(model_name, asyncio.Condition())
        async with condition:
            started = now()
            while any(info["busy"] for info in slots.values()):
                await condition.wait()
            saved = 0
            for slot_id, info in slots.items():
                if info["session"] and info["live_dirty"]:
                    log_event(
                        "slot save-all skip dirty",
                        model=model_name,
                        session=_session_label(info["session"]),
                        slot=slot_id,
                        checkpointed=info["checkpointed"],
                    )
                if (
                    info["session"]
                    and not info["live_dirty"]
                    and not info["checkpointed"]
                ):
                    if await self._checkpoint(
                        model_name, port, slot_id, info["session"]
                    ):
                        saved += 1
            log_event("slots save-all", model=model_name, count=saved, ms=elapsed_ms(started))

    async def _save(self, model_name, port, slot_id, session_id):
        """Save slot KV cache to disk via llama-server API."""
        if model_name in self._no_persistence:
            return False
        self._ensure_runtime_cache_dir(model_name)
        started = now()
        url = f"http://127.0.0.1:{port}/slots/{slot_id}?action=save"
        filename = _cache_filename(session_id)
        body = {"filename": filename}
        try:
            async with aiohttp.ClientSession() as sess:
                async with sess.post(url, json=body, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status == 200:
                        payload = await _json_or_empty(resp)
                        log_event(
                            "slot save",
                            model=model_name,
                            session=_session_label(session_id),
                            slot=slot_id,
                            status=resp.status,
                            ms=elapsed_ms(started),
                            file=filename,
                            tokens=payload.get("n_saved"),
                            bytes=payload.get("n_written"),
                        )
                        return True
                    elif resp.status == 400:
                        text = await resp.text()
                        unsupported = _looks_persistence_unsupported(text)
                        if unsupported:
                            self._no_persistence.add(model_name)
                        log_event(
                            "slot save disabled" if unsupported else "slot save failed",
                            model=model_name,
                            session=_session_label(session_id),
                            slot=slot_id,
                            status=resp.status,
                            ms=elapsed_ms(started),
                            file=filename,
                            error=_short_error(text),
                        )
                    else:
                        text = await resp.text()
                        log_event(
                            "slot save failed",
                            model=model_name,
                            session=_session_label(session_id),
                            slot=slot_id,
                            status=resp.status,
                            ms=elapsed_ms(started),
                            file=filename,
                            error=_short_error(text),
                        )
        except Exception as e:
            log_event(
                "slot save error",
                model=model_name,
                session=_session_label(session_id),
                slot=slot_id,
                ms=elapsed_ms(started),
                file=filename,
                error=e,
            )
        return False

    async def _restore(self, model_name, port, slot_id, session_id):
        """Restore slot KV cache from disk. Silently skips if unsupported or no saved state."""
        if model_name in self._no_persistence:
            return False
        self._ensure_runtime_cache_dir(model_name)
        started = now()
        url = f"http://127.0.0.1:{port}/slots/{slot_id}?action=restore"
        filename = _cache_filename(session_id)
        body = {"filename": filename}
        try:
            async with aiohttp.ClientSession() as sess:
                async with sess.post(url, json=body, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status == 200:
                        payload = await _json_or_empty(resp)
                        log_event(
                            "slot restore",
                            model=model_name,
                            session=_session_label(session_id),
                            slot=slot_id,
                            status=resp.status,
                            ms=elapsed_ms(started),
                            file=filename,
                            tokens=payload.get("n_restored"),
                            bytes=payload.get("n_read"),
                        )
                        return True
                    elif resp.status == 400:
                        text = await resp.text()
                        log_event(
                            "slot restore skipped",
                            model=model_name,
                            session=_session_label(session_id),
                            slot=slot_id,
                            status=resp.status,
                            ms=elapsed_ms(started),
                            file=filename,
                            reason="missing-or-unsupported",
                            error=_short_error(text),
                        )
                    else:
                        text = await resp.text()
                        log_event(
                            "slot restore failed",
                            model=model_name,
                            session=_session_label(session_id),
                            slot=slot_id,
                            status=resp.status,
                            ms=elapsed_ms(started),
                            file=filename,
                            error=_short_error(text),
                        )
        except Exception as e:
            log_event(
                "slot restore error",
                model=model_name,
                session=_session_label(session_id),
                slot=slot_id,
                ms=elapsed_ms(started),
                file=filename,
                error=e,
            )
        return False

    async def _checkpoint(self, model_name, port, slot_id, session_id):
        info = self._slots.get(model_name, {}).get(slot_id)
        saved = await self._save(model_name, port, slot_id, session_id)
        if info is not None:
            info["checkpointed"] = saved
            if saved:
                info["has_checkpoint"] = True
                info["requests_since_checkpoint"] = 0
                info["last_checkpoint_at"] = time.monotonic()
        return saved

    def _checkpoint_reason(self, info):
        if not info["has_checkpoint"]:
            return "initial"
        if info["last_checkpoint_at"] is None:
            return "restored"
        if (
            self.checkpoint_max_requests is not None
            and info["requests_since_checkpoint"] >= self.checkpoint_max_requests
        ):
            return "requests"
        if (
            self.checkpoint_interval_seconds is not None
            and time.monotonic() - info["last_checkpoint_at"]
            >= self.checkpoint_interval_seconds
        ):
            return "interval"
        return None

    @staticmethod
    def _reset_checkpoint_state(info):
        info["checkpointed"] = False
        info["has_checkpoint"] = False
        info["requests_since_checkpoint"] = 0
        info["last_checkpoint_at"] = None

    @staticmethod
    def _record_restore(info, restored):
        info["checkpointed"] = restored
        info["has_checkpoint"] = restored
        info["requests_since_checkpoint"] = 0
        # Force one fresh checkpoint after a restored session is advanced.
        info["last_checkpoint_at"] = None

    async def _erase(self, model_name, port, slot_id, session_id):
        """Erase an uncommitted slot when no checkpoint can be restored."""
        started = now()
        url = f"http://127.0.0.1:{port}/slots/{slot_id}?action=erase"
        try:
            async with aiohttp.ClientSession() as sess:
                async with sess.post(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status == 200:
                        log_event(
                            "slot rollback erase",
                            model=model_name,
                            session=_session_label(session_id),
                            slot=slot_id,
                            status=resp.status,
                            ms=elapsed_ms(started),
                        )
                        return True
                    text = await resp.text()
                    log_event(
                        "slot rollback erase failed",
                        model=model_name,
                        session=_session_label(session_id),
                        slot=slot_id,
                        status=resp.status,
                        ms=elapsed_ms(started),
                        error=_short_error(text),
                    )
        except Exception as e:
            log_event(
                "slot rollback erase error",
                model=model_name,
                session=_session_label(session_id),
                slot=slot_id,
                ms=elapsed_ms(started),
                error=e,
            )
        return False

    def _ensure_runtime_cache_dir(self, model_name):
        if self.cache_dir is None:
            return
        path = self.cache_dir / str(model_name)
        try:
            root = self.cache_dir.resolve()
            resolved = path.resolve()
            resolved.relative_to(root)
            resolved.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            log_event("slot cache dir ensure failed", model=model_name, path=path, error=exc)
        except ValueError:
            log_event("slot cache dir rejected", model=model_name, path=path)


def _session_label(session_id):
    if session_id is None:
        return "transient"
    text = str(session_id)
    if len(text) <= 16:
        return text
    return f"{text[:8]}..{text[-4:]}"


def _short_error(text, limit=120):
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


async def _json_or_empty(response):
    try:
        payload = await response.json(content_type=None)
    except (AttributeError, TypeError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _cache_filename(session_id):
    text = str(session_id)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._-")
    if not prefix:
        prefix = "session"
    return f"{prefix[:48]}-{digest}"


def _looks_persistence_unsupported(text):
    lowered = str(text).lower()
    return (
        "not supported" in lowered
        or "unsupported" in lowered
        or "slot_save_path" in lowered
        or "slot save path" in lowered
    )
