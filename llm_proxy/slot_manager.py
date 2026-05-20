"""Slot Manager — assign sessions to slots, save/restore KV cache."""

import asyncio
import time

import aiohttp


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

    async def acquire(self, model_name, session_id, port):
        """Acquire an exclusive slot lease for a session. Returns slot_id.

        If the session is already loaded in an idle slot, uses it.
        If a free idle slot is available, restores the session into it.
        Otherwise waits for an idle slot, then evicts the LRU session.
        A None session_id acquires a transient slot without restoring or saving
        transient state.
        """
        slots = self._slots.get(model_name)
        if not slots:
            return 0

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
                        return existing

                # Free idle slot?
                for slot_id, info in slots.items():
                    if info["session"] is None and not info["busy"]:
                        info["busy"] = True
                        info["session"] = session_id
                        info["last_used"] = time.time()
                        if not transient:
                            await self._restore(model_name, port, slot_id, session_id)
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
            slots[slot_id]["busy"] = False
            slots[slot_id]["last_used"] = time.time()
            condition.notify_all()

    async def save_all(self, model_name, port):
        """Save all active slots for a model (before stopping)."""
        if model_name in self._no_persistence:
            return
        slots = self._slots.get(model_name, {})
        if not slots:
            return
        condition = self._conditions.setdefault(model_name, asyncio.Condition())
        async with condition:
            while any(info["busy"] for info in slots.values()):
                await condition.wait()
            for slot_id, info in slots.items():
                if info["session"]:
                    await self._save(model_name, port, slot_id, info["session"])

    async def _save(self, model_name, port, slot_id, session_id):
        """Save slot KV cache to disk via llama-server API."""
        if model_name in self._no_persistence:
            return
        url = f"http://127.0.0.1:{port}/slots/{slot_id}?action=save"
        body = {"filename": session_id}
        try:
            async with aiohttp.ClientSession() as sess:
                async with sess.post(url, json=body, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status == 200:
                        print(f"  [Slots] Saved slot {slot_id} → {session_id}")
                    elif resp.status == 400:
                        self._no_persistence.add(model_name)
                        print(f"  [Slots] Save not supported for {model_name}, disabling persistence")
                    else:
                        text = await resp.text()
                        print(f"  [Slots] Save failed slot {slot_id}: {resp.status} {text}")
        except Exception as e:
            print(f"  [Slots] Save error slot {slot_id}: {e}")

    async def _restore(self, model_name, port, slot_id, session_id):
        """Restore slot KV cache from disk. Silently skips if unsupported or no saved state."""
        if model_name in self._no_persistence:
            return
        url = f"http://127.0.0.1:{port}/slots/{slot_id}?action=restore"
        body = {"filename": session_id}
        try:
            async with aiohttp.ClientSession() as sess:
                async with sess.post(url, json=body, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status == 200:
                        print(f"  [Slots] Restored slot {slot_id} ← {session_id}")
                    elif resp.status == 400:
                        pass  # No saved state or not supported
                    else:
                        text = await resp.text()
                        print(f"  [Slots] Restore failed slot {slot_id}: {resp.status} {text}")
        except Exception as e:
            print(f"  [Slots] Restore error slot {slot_id}: {e}")
