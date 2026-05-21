import unittest

from llm_proxy.router import (
    _add_session_header,
    _extract_session_marker,
    _strip_local_session_markers,
)


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


if __name__ == "__main__":
    unittest.main()
