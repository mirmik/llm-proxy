import os
import tempfile
import unittest

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from llm_proxy.route_manager import RouteManager
from llm_proxy.router import create_api_app, create_control_app
from llm_proxy.statistics import StatisticsStore


class FakeAuth:
    def require_client(self, request):
        return "client-a"

    def require_control(self, request):
        return "web:admin"


class DenyControlAuth(FakeAuth):
    def require_control(self, request):
        raise web.HTTPUnauthorized(text='{"error":"login required"}', content_type="application/json")


class FakeSlotManager:
    def __init__(self):
        self._slots = {}

    def init_model(self, name, slots):
        self._slots[name] = {index: None for index in range(slots)}

    async def acquire(self, model, session, port):
        return None

    async def release(self, model, slot):
        return None


class TrackingSlotManager(FakeSlotManager):
    def __init__(self):
        super().__init__()
        self.finalized = []

    async def acquire(self, model, session, port):
        return 0

    async def finalize_request(self, model, port, slot, *, success):
        self.finalized.append((model, port, slot, success))


class FakeRuntimeManager:
    def __init__(self, ports=None):
        self.ports = ports or {}
        self.config = {}

    def available_models(self):
        return [name for name in self.ports if name == "model"]

    def is_running(self, name):
        return name in self.ports

    def on_demand(self, name):
        return False

    def get_port(self, name):
        return self.ports.get(name)

    def get_slots(self, name):
        return 1

    def runtimes_by_kind(self, kind):
        expected = {"stt": "stt-main", "tts": "tts-main"}.get(kind)
        return [expected] if expected in self.ports else []


class RouterStatisticsTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.config = {
            "state_dir": self.tmp.name,
            "statistics": {"enabled": True, "include_host": False},
        }
        self.store = StatisticsStore(self.config)
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

    async def backend(self, routes):
        app = web.Application()
        for method, path, handler in routes:
            app.router.add_route(method, path, handler)
        return await self.start_app(app)

    async def proxy(self, rm, routes=None, sm=None):
        app = create_api_app(
            rm,
            sm or FakeSlotManager(),
            FakeAuth(),
            route_manager=routes,
        )
        app["statistics"] = self.store
        app["config"] = self.config
        return await self.start_app(app)

    async def control(self, rm=None):
        app = create_control_app(rm or FakeRuntimeManager(), FakeSlotManager(), FakeAuth())
        app["statistics"] = self.store
        app["config"] = self.config
        return await self.start_app(app)

    async def test_local_chat_records_client_request_and_attempt(self):
        async def chat(request):
            payload = await request.json()
            return web.json_response({
                "model": payload["model"],
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            })

        backend = await self.backend([("POST", "/v1/chat/completions", chat)])
        proxy = await self.proxy(FakeRuntimeManager({"model": backend.server.port}))
        raw = b'{"model":"model","messages":[{"role":"user","content":"hello"}]}'

        response = await proxy.post(
            "/v1/chat/completions", data=raw, headers={"Content-Type": "application/json"}
        )
        self.assertEqual(response.status, 200)
        response_body = await response.read()
        await self.store.flush()

        request_row = self.store.fetch_requests()[0]
        attempt = self.store.fetch_attempts()[0]
        self.assertEqual(request_row["client"], "client-a")
        self.assertIsNone(request_row["host"])
        self.assertEqual(request_row["client_input_bytes"], len(raw))
        self.assertEqual(request_row["client_output_bytes"], len(response_body))
        self.assertEqual(request_row["attempt_count"], 1)
        self.assertEqual(request_row["total_tokens"], 3)
        self.assertEqual(request_row["usage_available"], 1)
        self.assertEqual(attempt["runtime"], "model")
        self.assertEqual(attempt["outcome"], "success")

    async def test_streaming_chat_collects_fragmented_usage_and_ttft(self):
        async def chat(request):
            await request.json()
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            await response.write(b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n')
            await response.write(b'data: {"choices":[],"usage":{"prompt_tokens":4,')
            await response.write(b'"completion_tokens":2,"total_tokens":6}}\n\ndata: [DONE]\n\n')
            await response.write_eof()
            return response

        backend = await self.backend([("POST", "/v1/chat/completions", chat)])
        proxy = await self.proxy(FakeRuntimeManager({"model": backend.server.port}))

        response = await proxy.post("/v1/chat/completions", json={"model": "model", "stream": True})
        streamed = await response.read()
        self.assertIn(b"[DONE]", streamed)
        await self.store.flush()

        row = self.store.fetch_requests()[0]
        self.assertEqual(row["total_tokens"], 6)
        self.assertEqual(row["usage_available"], 1)
        self.assertIsNotNone(row["ttft_ms"])
        self.assertEqual(row["client_output_bytes"], len(streamed))

    async def test_complete_local_stream_commits_slot(self):
        async def chat(request):
            await request.json()
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            await response.write(b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n')
            await response.write(b"data: [DO")
            await response.write(b"NE]\n\n")
            await response.write_eof()
            return response

        backend = await self.backend([("POST", "/v1/chat/completions", chat)])
        sm = TrackingSlotManager()
        proxy = await self.proxy(FakeRuntimeManager({"model": backend.server.port}), sm=sm)

        response = await proxy.post(
            "/v1/chat/completions",
            json={"model": "model", "stream": True, "session_id": "session-a"},
        )
        await response.read()

        self.assertEqual(sm.finalized, [("model", backend.server.port, 0, True)])

    async def test_incomplete_local_stream_rolls_back_slot(self):
        async def chat(request):
            await request.json()
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            await response.write(b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n')
            await response.write_eof()
            return response

        backend = await self.backend([("POST", "/v1/chat/completions", chat)])
        sm = TrackingSlotManager()
        proxy = await self.proxy(FakeRuntimeManager({"model": backend.server.port}), sm=sm)

        response = await proxy.post(
            "/v1/chat/completions",
            json={"model": "model", "stream": True, "session_id": "session-a"},
        )
        await response.read()

        self.assertEqual(sm.finalized, [("model", backend.server.port, 0, False)])

    async def test_routed_failover_is_one_request_with_two_attempts(self):
        async def fail(request):
            return web.json_response({"error": "unavailable"}, status=503)

        async def succeed(request):
            payload = await request.json()
            return web.json_response({"model": payload["model"], "choices": []})

        first = await self.backend([("POST", "/v1/chat/completions", fail)])
        second = await self.backend([("POST", "/v1/chat/completions", succeed)])
        route_config = {
            "routes": {
                "route-model": {
                    "kind": "llm",
                    "backends": [
                        {"url": str(first.make_url("/")).rstrip("/"), "model": "model-a"},
                        {"url": str(second.make_url("/")).rstrip("/"), "model": "model-b"},
                    ],
                }
            }
        }
        proxy = await self.proxy(FakeRuntimeManager(), RouteManager(route_config))

        response = await proxy.post("/v1/chat/completions", json={"model": "route-model"})
        self.assertEqual(response.status, 200)
        await response.read()
        await self.store.flush()

        request_row = self.store.fetch_requests()[0]
        attempts = self.store.fetch_attempts()
        self.assertEqual(request_row["attempt_count"], 2)
        self.assertEqual(request_row["route"], "route-model")
        self.assertEqual([row["status"] for row in attempts], [503, 200])
        self.assertEqual([row["route_reason"] for row in attempts], ["least-loaded-no-session", "failover"])
        self.assertEqual(request_row["upstream_input_bytes"], sum(row["input_bytes"] for row in attempts))

    async def test_all_routed_backends_failed_finishes_one_request(self):
        async def fail(request):
            return web.json_response({"error": "unavailable"}, status=503)

        first = await self.backend([("POST", "/v1/chat/completions", fail)])
        second = await self.backend([("POST", "/v1/chat/completions", fail)])
        route_config = {"routes": {"route-model": {"kind": "llm", "backends": [
            {"url": str(first.make_url("/")).rstrip("/"), "model": "model-a"},
            {"url": str(second.make_url("/")).rstrip("/"), "model": "model-b"},
        ]}}}
        proxy = await self.proxy(FakeRuntimeManager(), RouteManager(route_config))

        response = await proxy.post("/v1/chat/completions", json={"model": "route-model"})
        self.assertEqual(response.status, 503)
        await response.read()
        await self.store.flush()

        rows = self.store.fetch_requests()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["attempt_count"], 2)
        self.assertEqual(rows[0]["error"], "all-backends-failed")
        self.assertEqual([row["status"] for row in self.store.fetch_attempts()], [503, 503])

    async def test_local_stt_counts_uploaded_and_returned_payload(self):
        async def stt(request):
            incoming = await request.read()
            self.assertEqual(incoming, b"audio-bytes")
            return web.json_response({"text": "hello"})

        backend = await self.backend([("POST", "/stt", stt)])
        proxy = await self.proxy(FakeRuntimeManager({"stt-main": backend.server.port}))

        response = await proxy.post("/stt", data=b"audio-bytes", headers={"Content-Type": "audio/wav"})
        response_body = await response.read()
        self.assertEqual(response.status, 200)
        await self.store.flush()

        row = self.store.fetch_requests()[0]
        self.assertEqual(row["kind"], "stt")
        self.assertEqual(row["client_input_bytes"], len(b"audio-bytes"))
        self.assertEqual(row["client_output_bytes"], len(response_body))

    async def test_named_stt_route_is_selected_by_runtime_header(self):
        received = []

        async def stt(request):
            received.append(await request.read())
            return web.json_response({"text": "routed"})

        backend = await self.backend([("POST", "/stt", stt)])
        config = {"routes": {"stt-gigaam": {"kind": "stt", "backends": [
            {"url": str(backend.make_url("/")).rstrip("/")},
        ]}}}
        proxy = await self.proxy(FakeRuntimeManager(), RouteManager(config))

        response = await proxy.post(
            "/stt",
            data=b"float32-audio",
            headers={"X-STT-Runtime": "stt-gigaam"},
        )

        self.assertEqual(response.status, 200)
        self.assertEqual((await response.json())["text"], "routed")
        self.assertEqual(received, [b"float32-audio"])

    async def test_named_stt_route_health_is_forwarded(self):
        async def health(request):
            return web.json_response({"status": "ok", "worker": "remote"})

        backend = await self.backend([("GET", "/health", health)])
        config = {"routes": {"stt-gigaam": {"kind": "stt", "backends": [
            {"url": str(backend.make_url("/")).rstrip("/")},
        ]}}}
        proxy = await self.proxy(FakeRuntimeManager(), RouteManager(config))

        response = await proxy.get(
            "/stt/health",
            headers={"X-STT-Runtime": "stt-gigaam"},
        )
        body = await response.json()

        self.assertEqual(response.status, 200)
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["worker"], "remote")
        self.assertEqual(body["runtime"], "stt-gigaam")
        self.assertEqual(body["route"], "stt-gigaam")
        self.assertTrue(body["loaded"])

    async def test_local_tts_counts_streamed_audio(self):
        async def tts(request):
            await request.json()
            response = web.StreamResponse(headers={"Content-Type": "audio/wav"})
            await response.prepare(request)
            await response.write(b"audio-")
            await response.write(b"result")
            await response.write_eof()
            return response

        backend = await self.backend([("POST", "/tts/generate", tts)])
        proxy = await self.proxy(FakeRuntimeManager({"tts-main": backend.server.port}))
        raw = b'{"voice":"tts-main","input":"hello"}'

        response = await proxy.post(
            "/v1/audio/speech", data=raw, headers={"Content-Type": "application/json"}
        )
        audio = await response.read()
        self.assertEqual(audio, b"audio-result")
        await self.store.flush()

        row = self.store.fetch_requests()[0]
        self.assertEqual(row["kind"], "tts")
        self.assertEqual(row["client_input_bytes"], len(raw))
        self.assertEqual(row["client_output_bytes"], len(audio))
        self.assertIsNotNone(row["ttft_ms"])

    async def test_control_statistics_aggregates_requests_and_attempts(self):
        now = __import__("time").time()
        request = {
            "request_id": "aggregate-1", "started_at": now - 10, "completed_at": now - 9,
            "duration_ms": 1000, "kind": "chat", "client": "client-a",
            "resolved_model": "model", "route": "route", "status": 200,
            "outcome": "success", "client_input_bytes": 10, "client_output_bytes": 20,
            "upstream_input_bytes": 11, "upstream_output_bytes": 21,
            "completion_tokens": 5, "total_tokens": 8, "usage_available": True,
            "ttft_ms": 500, "attempt_count": 2,
        }
        attempt_base = {
            "request_id": "aggregate-1", "started_at": now - 10, "completed_at": now - 9,
            "duration_ms": 500, "kind": "chat", "model": "model", "route": "route",
            "input_bytes": 11, "output_bytes": 10,
        }
        self.store.record_request(request)
        self.store.record_attempt(dict(attempt_base, attempt_no=1, backend="one", status=503,
                                       outcome="error", route_reason="least-loaded"))
        self.store.record_attempt(dict(attempt_base, attempt_no=2, backend="two", status=200,
                                       outcome="success", route_reason="failover"))
        control = await self.control()

        response = await control.get("/v1/control/statistics?period=1h")
        payload = await response.json()

        self.assertEqual(response.status, 200)
        self.assertEqual(payload["summary"]["requests"], 1)
        self.assertEqual(payload["summary"]["attempts"], 2)
        self.assertEqual(payload["summary"]["failovers"], 1)
        self.assertEqual(payload["summary"]["total_tokens"], 8)
        self.assertEqual(payload["summary"]["tokens_per_second"], 10.0)
        self.assertEqual(len(payload["breakdown"]["backend"]), 2)

    async def test_control_statistics_validates_period_and_can_clear(self):
        control = await self.control()

        invalid = await control.get("/v1/control/statistics?period=year")
        self.assertEqual(invalid.status, 400)
        cleared = await control.delete("/v1/control/statistics")
        self.assertEqual(cleared.status, 200)
        self.assertEqual((await cleared.json())["deleted"], {"requests": 0, "attempts": 0})

    async def test_control_statistics_requires_control_auth(self):
        app = create_control_app(FakeRuntimeManager(), FakeSlotManager(), DenyControlAuth())
        app["statistics"] = self.store
        app["config"] = self.config
        control = await self.start_app(app)

        response = await control.get("/v1/control/statistics")

        self.assertEqual(response.status, 401)


if __name__ == "__main__":
    unittest.main()
