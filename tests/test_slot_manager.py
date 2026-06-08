import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from llm_proxy.slot_manager import SlotManager, _cache_filename


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

    async def test_transient_request_can_evict_persistent_session_without_restoring(self):
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

    async def test_transient_request_reuses_busy_transient_slot_without_evicting_persistent_session(self):
        sm = RecordingSlotManager()
        sm.init_model("model", 2)

        session_slot = await sm.acquire("model", "session-a", 8081)
        await sm.release("model", session_slot)
        first_transient_slot = await sm.acquire("model", None, 8081)
        sm.saved.clear()
        sm.restored.clear()

        second_transient = asyncio.create_task(sm.acquire("model", None, 8081))
        await asyncio.sleep(0)

        self.assertFalse(second_transient.done())
        self.assertEqual(sm.saved, [])
        self.assertEqual(sm._slots["model"][session_slot]["session"], "session-a")

        await sm.release("model", first_transient_slot)
        second_transient_slot = await asyncio.wait_for(second_transient, timeout=1)

        self.assertEqual(second_transient_slot, first_transient_slot)
        self.assertEqual(sm.saved, [])
        self.assertEqual(sm.restored, [])
        self.assertEqual(sm._slots["model"][session_slot]["session"], "session-a")
        await sm.release("model", second_transient_slot)

    async def test_transient_requests_share_one_slot_even_when_another_slot_is_empty(self):
        sm = RecordingSlotManager()
        sm.init_model("model", 2)

        first_transient_slot = await sm.acquire("model", None, 8081)

        second_transient = asyncio.create_task(sm.acquire("model", None, 8081))
        await asyncio.sleep(0)

        self.assertFalse(second_transient.done())
        self.assertIsNone(sm._slots["model"][1]["session"])

        await sm.release("model", first_transient_slot)
        second_transient_slot = await asyncio.wait_for(second_transient, timeout=1)

        self.assertEqual(second_transient_slot, first_transient_slot)
        self.assertEqual(sm.saved, [])
        self.assertEqual(sm.restored, [])
        await sm.release("model", second_transient_slot)

    async def test_cache_filename_is_safe_for_path_like_session_ids(self):
        filename = _cache_filename("atom/2026/05/21/session:1529")

        self.assertNotIn("/", filename)
        self.assertNotIn("\\", filename)
        self.assertIn("-", filename)

    async def test_save_recreates_runtime_cache_dir_before_backend_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            runtime_dir = cache_dir / "model"
            sm = SlotManager(cache_dir=cache_dir)

            with patch("llm_proxy.slot_manager.aiohttp.ClientSession", _DummyClientSession):
                await sm._save("model", 1, 0, "session-a")

            self.assertTrue(runtime_dir.exists())


class _DummyClientSession:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    def post(self, *args, **kwargs):
        return _DummyResponse()


class _DummyResponse:
    status = 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def text(self):
        return ""


if __name__ == "__main__":
    unittest.main()
