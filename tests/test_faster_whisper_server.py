import importlib
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from aiohttp.test_utils import TestClient, TestServer


class FasterWhisperPromptTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        fake_numpy = types.ModuleType("numpy")
        fake_numpy.float32 = object()
        fake_numpy.frombuffer = lambda _raw, dtype: [0.0] * 16000
        fake_whisper = types.ModuleType("faster_whisper")
        fake_whisper.WhisperModel = object
        cls.module_patch = patch.dict(
            sys.modules,
            {"numpy": fake_numpy, "faster_whisper": fake_whisper},
        )
        cls.module_patch.start()
        sys.modules.pop("llm_proxy.runners.faster_whisper_server", None)
        cls.runner = importlib.import_module(
            "llm_proxy.runners.faster_whisper_server"
        )

    @classmethod
    def tearDownClass(cls):
        sys.modules.pop("llm_proxy.runners.faster_whisper_server", None)
        cls.module_patch.stop()

    async def asyncSetUp(self):
        self.model = Mock()
        self.model.transcribe.return_value = (
            [],
            SimpleNamespace(language="ru"),
        )
        args = SimpleNamespace(
            token=None,
            sample_rate=16000,
            language=None,
            model_size="test",
        )
        app = self.runner.build_app(args, self.model)
        self.client = TestClient(TestServer(app))
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()

    async def test_unicode_prompt_reaches_model_unchanged(self):
        response = await self.client.post(
            "/stt",
            data=b"audio",
            headers={
                "Content-Type": "application/octet-stream",
                "X-Initial-Prompt-Encoded":
                    "v1:0JDQstCw0LvQvtC9IOKAlCDQs9C+0LvQvtGB0L7QstC+0Lkg0LDRgdGB0LjRgdGC0LXQvdGC",
            },
        )

        self.assertEqual(response.status, 200)
        self.assertEqual(
            self.model.transcribe.call_args.kwargs["initial_prompt"],
            "Авалон — голосовой ассистент",
        )

    async def test_malformed_prompt_is_rejected_before_model_call(self):
        response = await self.client.post(
            "/stt",
            data=b"audio",
            headers={
                "Content-Type": "application/octet-stream",
                "X-Initial-Prompt-Encoded": "v1:not base64!",
            },
        )

        self.assertEqual(response.status, 400)
        self.model.transcribe.assert_not_called()


if __name__ == "__main__":
    unittest.main()
