"""Slot Manager — assign sessions to slots, save/restore KV cache."""

import time

import aiohttp


class SlotManager:
    """Manages slot ↔ session mapping with LRU eviction and KV cache persistence."""

    def __init__(self):
        # model_name → {slot_id: {"session": str|None, "last_used": float}}
        self._slots = {}
        # Models where save/restore is known to be unsupported
        self._no_persistence = set()

    def init_model(self, model_name, num_slots):
        """Register slots for a newly started model."""
        self._slots[model_name] = {
            i: {"session": None, "last_used": 0}
            for i in range(num_slots)
        }

    def remove_model(self, model_name):
        """Remove slot tracking for a stopped model."""
        self._slots.pop(model_name, None)
        self._no_persistence.discard(model_name)

    def get_slot(self, model_name, session_id):
        """Find slot currently holding this session, or None."""
        slots = self._slots.get(model_name, {})
        for slot_id, info in slots.items():
            if info["session"] == session_id:
                return slot_id
        return None

    async def assign(self, model_name, session_id, port):
        """Assign a slot to a session. Returns slot_id.

        If session is already loaded — returns its slot (cache hit).
        If free slot available — uses it.
        Otherwise — evicts LRU slot (saving its cache first).
        """
        slots = self._slots.get(model_name)
        if not slots:
            return 0

        # Already in a slot?
        existing = self.get_slot(model_name, session_id)
        if existing is not None:
            slots[existing]["last_used"] = time.time()
            return existing

        # Free slot?
        for slot_id, info in slots.items():
            if info["session"] is None:
                info["session"] = session_id
                info["last_used"] = time.time()
                await self._restore(model_name, port, slot_id, session_id)
                return slot_id

        # Evict LRU
        lru_slot = min(slots, key=lambda s: slots[s]["last_used"])
        old_session = slots[lru_slot]["session"]

        if old_session:
            await self._save(model_name, port, lru_slot, old_session)

        await self._restore(model_name, port, lru_slot, session_id)
        slots[lru_slot]["session"] = session_id
        slots[lru_slot]["last_used"] = time.time()
        return lru_slot

    async def save_all(self, model_name, port):
        """Save all active slots for a model (before stopping)."""
        if model_name in self._no_persistence:
            return
        slots = self._slots.get(model_name, {})
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
