import json
import unittest
from types import SimpleNamespace

from llm_proxy.router import handle_capabilities, handle_models


class FakeRuntimeManager:
    def available_models(self):
        return ["local-llm"]

    def runtimes_by_kind(self, kind):
        return ["stt-main"] if kind == "stt" else []

    def is_running(self, name):
        return name == "local-llm"

    def get_port(self, name):
        return 12345 if name == "local-llm" else None


class FakeRoutes:
    def __init__(self):
        self.routes = {
            "remote-llm": SimpleNamespace(kind="llm"),
            "remote-tts": SimpleNamespace(kind="tts"),
        }

    def available_routes(self):
        return list(self.routes)

    def get_route(self, name):
        return self.routes.get(name)


class DiscoveryTests(unittest.IsolatedAsyncioTestCase):
    def request(self):
        return SimpleNamespace(app={
            "rm": FakeRuntimeManager(),
            "routes": FakeRoutes(),
            "auth": SimpleNamespace(enabled=True),
        })

    async def test_capabilities_are_public_protocol_metadata(self):
        response = await handle_capabilities(self.request())
        payload = json.loads(response.text)
        self.assertEqual(payload["protocol"], 1)
        self.assertTrue(payload["auth_required"])
        self.assertEqual(payload["services"], {
            "llm": True,
            "embedding": False,
            "stt": True,
            "tts": True,
        })

    async def test_models_exclude_non_llm_routes(self):
        request = self.request()
        request.app["auth"] = None
        response = await handle_models(request)
        ids = [item["id"] for item in json.loads(response.text)["data"]]
        self.assertEqual(ids, ["local-llm", "remote-llm"])


if __name__ == "__main__":
    unittest.main()
