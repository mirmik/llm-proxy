import unittest

from llm_proxy.router import (
    _add_session_header,
    _acquire_slot,
    _extract_session_marker,
    _strip_local_session_markers,
)
from llm_proxy.slot_manager import SlotManager


class RouterSessionMarkerTests(unittest.TestCase):
    def test_extracts_top_level_session_id_first(self):
        body = {
            "session_id": "snake-session",
            "metadata": {"sessionId": "metadata-session"},
        }

        session_id, source = _extract_session_marker(body)

        self.assertEqual(session_id, "snake-session")
        self.assertEqual(source, "body.session_id")

    def test_extracts_qwen_metadata_session_id(self):
        body = {"metadata": {"sessionId": "qwen-session"}}

        session_id, source = _extract_session_marker(body)

        self.assertEqual(session_id, "qwen-session")
        self.assertEqual(source, "metadata.sessionId")

    def test_extracts_qwen_subagent_session_from_prompt_id(self):
        body = {
            "metadata": {
                "sessionId": "parent-session",
                "promptId": "parent-session#Explore-a1b2c3d4#2",
            },
        }

        session_id, source = _extract_session_marker(body)

        self.assertEqual(session_id, "parent-session#Explore-a1b2c3d4")
        self.assertEqual(source, "metadata.promptId:qwen")

    def test_extracts_qwen_main_session_from_prompt_id_delimiter(self):
        body = {
            "metadata": {
                "sessionId": "parent-session",
                "promptId": "parent-session########0",
            },
        }

        session_id, source = _extract_session_marker(body)

        self.assertEqual(session_id, "parent-session")
        self.assertEqual(source, "metadata.promptId:qwen")

    def test_ignores_non_qwen_prompt_id_shape(self):
        body = {
            "metadata": {
                "sessionId": "parent-session",
                "promptId": "not-a-qwen-prompt-id",
            },
        }

        session_id, source = _extract_session_marker(body)

        self.assertEqual(session_id, "parent-session")
        self.assertEqual(source, "metadata.sessionId")

    def test_treats_qwen_managed_auto_prompt_as_transient(self):
        body = {
            "metadata": {
                "sessionId": "parent-session",
                "promptId": "parent-session_managed-aut-abc123########0",
            },
        }

        session_id, source = _extract_session_marker(body)

        self.assertIsNone(session_id)
        self.assertEqual(source, "metadata.promptId:qwen:internal")

    def test_treats_qwen_side_query_prompt_as_transient(self):
        body = {
            "metadata": {
                "sessionId": "parent-session",
                "promptId": "side-query:auto-memory-recall",
            },
        }

        session_id, source = _extract_session_marker(body)

        self.assertIsNone(session_id)
        self.assertEqual(source, "metadata.promptId:qwen:internal")

    def test_treats_qwen_managed_auto_subagent_prompt_as_transient(self):
        body = {
            "metadata": {
                "sessionId": "parent-session",
                "promptId": "parent-session#managed-auto-memory-extractor-abc123#0",
            },
        }

        session_id, source = _extract_session_marker(body)

        self.assertIsNone(session_id)
        self.assertEqual(source, "metadata.promptId:qwen:internal")

    def test_extracts_header_session_id(self):
        body = {"model": "model"}
        headers = {"X-LLM-Proxy-Session-ID": "header-session"}

        session_id, source = _extract_session_marker(body, headers)

        self.assertEqual(session_id, "header-session")
        self.assertEqual(source, "header.X-LLM-Proxy-Session-ID")

    def test_ignores_empty_and_structured_session_values(self):
        body = {
            "session_id": "",
            "metadata": {"sessionId": {"bad": "shape"}},
        }

        session_id, source = _extract_session_marker(body)

        self.assertIsNone(session_id)
        self.assertIsNone(source)

    def test_strips_local_session_markers_but_keeps_other_metadata(self):
        body = {
            "model": "model",
            "sessionId": "camel-session",
            "metadata": {
                "sessionId": "qwen-session",
                "trace_id": "trace-1",
            },
        }

        stripped = _strip_local_session_markers(body)

        self.assertIs(stripped, body)
        self.assertNotIn("sessionId", stripped)
        self.assertEqual(stripped["metadata"], {"trace_id": "trace-1"})

    def test_adds_forwarding_header_without_overwriting(self):
        headers = {"X-LLM-Proxy-Session-ID": "existing"}

        _add_session_header(headers, "new")

        self.assertEqual(headers["X-LLM-Proxy-Session-ID"], "existing")


class RouterSlotTrackingTests(unittest.IsolatedAsyncioTestCase):
    async def test_reinitializes_slot_tracking_when_runtime_slot_count_changes(self):
        class RuntimeManager:
            slots = 1

            def get_slots(self, _model_name):
                return self.slots

        rm = RuntimeManager()
        sm = SlotManager()

        slot = await _acquire_slot(rm, sm, "model", None, 8081)
        self.assertEqual(slot, 0)
        await sm.release("model", slot)
        self.assertEqual(len(sm._slots["model"]), 1)

        rm.slots = 2
        slot = await _acquire_slot(rm, sm, "model", None, 8081)

        self.assertEqual(slot, 0)
        self.assertEqual(len(sm._slots["model"]), 2)


if __name__ == "__main__":
    unittest.main()
