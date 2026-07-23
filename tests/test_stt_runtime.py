import json
import unittest

from llm_proxy.router import _resolve_stt_runtime


class RuntimeManager:
    def runtimes_by_kind(self, kind):
        self.requested_kind = kind
        return ["stt-main", "stt-gigaam"]


class STTRuntimeResolutionTests(unittest.TestCase):
    def test_explicit_runtime_wins_over_default(self):
        runtime, error = _resolve_stt_runtime(RuntimeManager(), requested="stt-gigaam")

        self.assertEqual(runtime, "stt-gigaam")
        self.assertIsNone(error)

    def test_default_remains_stt_main_without_selector(self):
        runtime, error = _resolve_stt_runtime(RuntimeManager())

        self.assertEqual(runtime, "stt-main")
        self.assertIsNone(error)

    def test_unknown_runtime_returns_available_names(self):
        runtime, error = _resolve_stt_runtime(RuntimeManager(), requested="missing")

        self.assertIsNone(runtime)
        self.assertEqual(error.status, 404)
        payload = json.loads(error.text)
        self.assertEqual(payload["available"], ["stt-main", "stt-gigaam"])


if __name__ == "__main__":
    unittest.main()
