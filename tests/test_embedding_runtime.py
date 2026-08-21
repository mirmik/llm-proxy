import json
import os
import tempfile
import unittest

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from llm_proxy.drivers import build_driver_registry
from llm_proxy.router import _resolve_runtime_by_kind, create_api_app
from llm_proxy.statistics import StatisticsStore


class FakeAuth:
    enabled = True

    def require_client(self, request):
        return "embedding-client"


class DenyAuth:
    def require_client(self, request):
        raise web.HTTPUnauthorized(
            text='{"error":"invalid token"}',
            content_type="application/json",
        )


class FakeSlotManager:
    _slots = {}


class FakeRuntimeManager:
    def __init__(self, embedding_names=None, chat_names=None, ports=None, on_demand=None):
        self.embedding_names = embedding_names or []
        self.chat_names = chat_names or []
        self.ports = dict(ports or {})
        self.on_demand_names = set(on_demand or [])
        self.running = set(self.ports)
        self.ensure_calls = []
        self.config = {}

    def available_models(self):
        return list(self.chat_names)

    def runtimes_by_kind(self, kind):
        if kind == "embedding":
            return list(self.embedding_names)
        return []

    def is_running(self, name):
        return name in self.running

    def on_demand(self, name):
        return name in self.on_demand_names

    async def ensure(self, name):
        self.ensure_calls.append(name)
        self.running.add(name)
        return self.ports.get(name)

    def get_port(self, name):
        return self.ports.get(name)


class EmbeddingRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = StatisticsStore({
            "state_dir": self.tmp.name,
            "statistics": {"enabled": True, "include_host": False},
        })
        await self.store.start()
        self.clients = []

    async def asyncTearDown(self):
        for client in reversed(self.clients):
            await client.close()
        await self.store.stop()
        self.tmp.cleanup()

    async def start_app(self, app):
        client = TestClient(TestServer(app))
        await client.start_server()
        self.clients.append(client)
        return client

    async def backend(self, handler):
        app = web.Application()
        app.router.add_post("/v1/embeddings", handler)
        return await self.start_app(app)

    async def proxy(self, rm, auth=None):
        app = create_api_app(rm, FakeSlotManager(), auth or FakeAuth())
        app["statistics"] = self.store
        app["config"] = {"statistics": {"enabled": True, "include_host": False}}
        return await self.start_app(app)

    def test_embedding_driver_is_registered_with_distinct_kind(self):
        registry = build_driver_registry()
        self.assertEqual(registry["llama_cpp"].kind(), "llm")
        self.assertEqual(registry["llama_cpp_embedding"].kind(), "embedding")

    def test_kind_aware_resolution(self):
        rm = FakeRuntimeManager(
            embedding_names=["bge-m3", "bge-small"],
            chat_names=["bge-chat"],
        )
        self.assertEqual(_resolve_runtime_by_kind(rm, "bge-m3", "embedding"), ("bge-m3", None))
        self.assertEqual(_resolve_runtime_by_kind(rm, "small", "embedding"), ("bge-small", None))

        runtime, response = _resolve_runtime_by_kind(rm, "bge", "embedding")
        self.assertIsNone(runtime)
        self.assertEqual(response.status, 400)

        runtime, response = _resolve_runtime_by_kind(rm, "bge-chat", "embedding")
        self.assertIsNone(runtime)
        self.assertEqual(response.status, 404)

    async def test_string_and_batch_inputs_are_forwarded_unchanged(self):
        received = []

        async def embeddings(request):
            body = await request.json()
            received.append(body)
            inputs = body["input"] if isinstance(body["input"], list) else [body["input"]]
            return web.json_response({
                "object": "list",
                "model": body["model"],
                "data": [
                    {"object": "embedding", "index": index, "embedding": [float(index), 1.0]}
                    for index, _ in enumerate(inputs)
                ],
                "usage": {"prompt_tokens": len(inputs), "total_tokens": len(inputs)},
            })

        backend = await self.backend(embeddings)
        rm = FakeRuntimeManager(
            embedding_names=["bge-m3"],
            ports={"bge-m3": backend.server.port},
        )
        proxy = await self.proxy(rm)

        capabilities = await (await proxy.get("/v1/capabilities")).json()
        models = await (await proxy.get("/v1/models")).json()
        self.assertTrue(capabilities["services"]["embedding"])
        self.assertEqual(models["data"], [])

        single = {"model": "bge-m3", "input": "секретный текст"}
        batch = {"model": "bge-m3", "input": ["первый", "second"]}
        self.assertEqual((await proxy.post("/v1/embeddings", json=single)).status, 200)
        response = await proxy.post("/v1/embeddings", json=batch)
        self.assertEqual(response.status, 200)
        self.assertEqual([item["index"] for item in (await response.json())["data"]], [0, 1])
        self.assertEqual(received, [single, batch])

        await self.store.flush()
        rows = self.store.fetch_requests()
        self.assertEqual([row["kind"] for row in rows], ["embedding", "embedding"])
        self.assertEqual(rows[1]["runtime"], "bge-m3")
        self.assertEqual(rows[1]["total_tokens"], 2)
        serialized_rows = json.dumps(rows, ensure_ascii=False)
        self.assertNotIn("секретный текст", serialized_rows)
        self.assertNotIn("embedding\":", serialized_rows)

    async def test_upstream_status_is_preserved(self):
        async def embeddings(request):
            return web.json_response({"error": "input is too large"}, status=413)

        backend = await self.backend(embeddings)
        rm = FakeRuntimeManager(
            embedding_names=["bge-m3"],
            ports={"bge-m3": backend.server.port},
        )
        response = await (await self.proxy(rm)).post(
            "/v1/embeddings", json={"model": "bge-m3", "input": "x"}
        )
        self.assertEqual(response.status, 413)
        self.assertEqual((await response.json())["error"], "input is too large")

    async def test_invalid_requests_and_wrong_kind_are_rejected(self):
        rm = FakeRuntimeManager(embedding_names=["bge-m3"], chat_names=["chat-main"])
        proxy = await self.proxy(rm)

        invalid_json = await proxy.post(
            "/v1/embeddings",
            data=b"{",
            headers={"Content-Type": "application/json"},
        )
        invalid_shape = await proxy.post("/v1/embeddings", json=[])
        empty_model = await proxy.post("/v1/embeddings", json={"model": " ", "input": "x"})
        chat_model = await proxy.post(
            "/v1/embeddings", json={"model": "chat-main", "input": "x"}
        )
        embedding_as_chat = await proxy.post(
            "/v1/chat/completions", json={"model": "bge-m3", "messages": []}
        )
        self.assertEqual(
            [
                invalid_json.status,
                invalid_shape.status,
                empty_model.status,
                chat_model.status,
                embedding_as_chat.status,
            ],
            [400, 400, 400, 404, 404],
        )

    async def test_stopped_and_on_demand_runtimes(self):
        stopped = FakeRuntimeManager(embedding_names=["bge-m3"])
        stopped_response = await (await self.proxy(stopped)).post(
            "/v1/embeddings", json={"model": "bge-m3", "input": "x"}
        )
        self.assertEqual(stopped_response.status, 503)

        class FailingRuntimeManager(FakeRuntimeManager):
            async def ensure(self, name):
                raise RuntimeError("startup failed")

        failing = FailingRuntimeManager(
            embedding_names=["bge-m3"],
            on_demand=["bge-m3"],
        )
        failed_response = await (await self.proxy(failing)).post(
            "/v1/embeddings", json={"model": "bge-m3", "input": "x"}
        )
        self.assertEqual(failed_response.status, 503)
        self.assertIn("startup failed", (await failed_response.json())["error"])

        async def embeddings(request):
            return web.json_response({"object": "list", "data": [], "usage": {}})

        backend = await self.backend(embeddings)
        on_demand = FakeRuntimeManager(
            embedding_names=["bge-m3"],
            ports={"bge-m3": backend.server.port},
            on_demand=["bge-m3"],
        )
        on_demand.running.clear()
        response = await (await self.proxy(on_demand)).post(
            "/v1/embeddings", json={"model": "bge-m3", "input": "x"}
        )
        self.assertEqual(response.status, 200)
        self.assertEqual(on_demand.ensure_calls, ["bge-m3"])

    async def test_api_auth_is_required_before_forwarding(self):
        rm = FakeRuntimeManager(embedding_names=["bge-m3"])
        response = await (await self.proxy(rm, auth=DenyAuth())).post(
            "/v1/embeddings", json={"model": "bge-m3", "input": "x"}
        )
        self.assertEqual(response.status, 401)


if __name__ == "__main__":
    unittest.main()
