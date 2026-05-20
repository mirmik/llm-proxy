import asyncio
import unittest

from llm_proxy.slot_manager import SlotManager


class RecordingSlotManager(SlotManager):
    def __init__(self):
        super().__init__()
        self.saved = []
        self.restored = []

    async def _save(self, model_name, port, slot_id, session_id):
        self.saved.append((model_name, port, slot_id, session_id))

    async def _restore(self, model_name, port, slot_id, session_id):
        self.restored.append((model_name, port, slot_id, session_id))


class SlotManagerConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    async def test_same_session_waits_for_busy_slot(self):
        sm = RecordingSlotManager()
        sm.init_model("model", 1)

        first_slot = await sm.acquire("model", "session-a", 8081)
        self.assertEqual(first_slot, 0)

        second = asyncio.create_task(sm.acquire("model", "session-a", 8081))
        await asyncio.sleep(0)
        self.assertFalse(second.done())

        await sm.release("model", first_slot)
        second_slot = await asyncio.wait_for(second, timeout=1)
        self.assertEqual(second_slot, 0)
        await sm.release("model", second_slot)

    async def test_same_session_does_not_duplicate_across_free_slots(self):
        sm = RecordingSlotManager()
        sm.init_model("model", 2)

        first_slot = await sm.acquire("model", "session-a", 8081)
        self.assertEqual(first_slot, 0)

        second = asyncio.create_task(sm.acquire("model", "session-a", 8081))
        await asyncio.sleep(0)
        self.assertFalse(second.done())

        await sm.release("model", first_slot)
        second_slot = await asyncio.wait_for(second, timeout=1)
        self.assertEqual(second_slot, first_slot)
        self.assertIsNone(sm._slots["model"][1]["session"])
        await sm.release("model", second_slot)

    async def test_eviction_waits_for_busy_slot_before_save_restore(self):
        sm = RecordingSlotManager()
        sm.init_model("model", 1)

        first_slot = await sm.acquire("model", "session-a", 8081)
        second = asyncio.create_task(sm.acquire("model", "session-b", 8081))
        await asyncio.sleep(0)

        self.assertFalse(second.done())
        self.assertEqual(sm.saved, [])

        await sm.release("model", first_slot)
        second_slot = await asyncio.wait_for(second, timeout=1)
        self.assertEqual(second_slot, 0)
        self.assertEqual(sm.saved, [("model", 8081, 0, "session-a")])
        self.assertEqual(
            sm.restored,
            [
                ("model", 8081, 0, "session-a"),
                ("model", 8081, 0, "session-b"),
            ],
        )
        await sm.release("model", second_slot)

    async def test_transient_request_saves_evicted_session_without_restoring(self):
        sm = RecordingSlotManager()
        sm.init_model("model", 1)

        session_slot = await sm.acquire("model", "session-a", 8081)
        await sm.release("model", session_slot)
        sm.saved.clear()
        sm.restored.clear()

        transient_slot = await sm.acquire("model", None, 8081)

        self.assertEqual(transient_slot, 0)
        self.assertEqual(sm.saved, [("model", 8081, 0, "session-a")])
        self.assertEqual(sm.restored, [])
        self.assertIsNone(sm._slots["model"][0]["session"])
        await sm.release("model", transient_slot)


if __name__ == "__main__":
    unittest.main()
