"""Slot Manager — assign sessions to slots, save/restore KV cache."""

import asyncio
import hashlib
import re
import time

import aiohttp

from llm_proxy.oplog import elapsed_ms, log_event, now


class SlotManager:
    """Manages slot ↔ session mapping with LRU eviction and KV cache persistence."""

    def __init__(self):
        # model_name → {slot_id: {"session": str|None, "last_used": float, "busy": bool}}
        self._slots = {}
        self._conditions = {}
        # Models where save/restore is known to be unsupported
        self._no_persistence = set()

    def init_model(self, model_name, num_slots):
        """Register slots for a newly started model."""
        self._slots[model_name] = {
            i: {"session": None, "last_used": 0, "busy": False}
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
                            await self._restore(model_name, port, slot_id, session_id)
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

                if old_session:
                    await self._save(model_name, port, lru_slot, old_session)

                if not transient:
                    await self._restore(model_name, port, lru_slot, session_id)
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
                if info["session"]:
                    await self._save(model_name, port, slot_id, info["session"])
                    saved += 1
            log_event("slots save-all", model=model_name, count=saved, ms=elapsed_ms(started))

    async def _save(self, model_name, port, slot_id, session_id):
        """Save slot KV cache to disk via llama-server API."""
        if model_name in self._no_persistence:
            return
        started = now()
        url = f"http://127.0.0.1:{port}/slots/{slot_id}?action=save"
        filename = _cache_filename(session_id)
        body = {"filename": filename}
        try:
            async with aiohttp.ClientSession() as sess:
                async with sess.post(url, json=body, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status == 200:
                        log_event(
                            "slot save",
                            model=model_name,
                            session=_session_label(session_id),
                            slot=slot_id,
                            status=resp.status,
                            ms=elapsed_ms(started),
                            file=filename,
                        )
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

    async def _restore(self, model_name, port, slot_id, session_id):
        """Restore slot KV cache from disk. Silently skips if unsupported or no saved state."""
        if model_name in self._no_persistence:
            return
        started = now()
        url = f"http://127.0.0.1:{port}/slots/{slot_id}?action=restore"
        filename = _cache_filename(session_id)
        body = {"filename": filename}
        try:
            async with aiohttp.ClientSession() as sess:
                async with sess.post(url, json=body, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status == 200:
                        log_event(
                            "slot restore",
                            model=model_name,
                            session=_session_label(session_id),
                            slot=slot_id,
                            status=resp.status,
                            ms=elapsed_ms(started),
                            file=filename,
                        )
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
