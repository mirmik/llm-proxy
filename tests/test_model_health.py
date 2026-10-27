import asyncio
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from llm_proxy.auth import AuthManager
from llm_proxy.route_manager import Route, RouteManager
from llm_proxy.router import create_api_app
from llm_proxy.runtime_manager import RuntimeManager


class ModelHealthTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.clients = []
        self.backend_status = 200
        self.backend_calls = 0

        async def health(request):
            self.backend_calls += 1
            return web.json_response({"status": "backend"}, status=self.backend_status)

        backend_app = web.Application()
        backend_app.router.add_get("/custom/health", health)
        self.backend = await self.start_app(backend_app)
        self.config = {
            "state_dir": self.tmp.name,
            "cache_dir": self.tmp.name,
            "auth": {"enabled": True},
            "runtimes": {
                "chat": {
                    "driver": "llama_cpp", "port": self.backend.server.port,
                    "health_path": "/custom/health", "command": "fake-server",
                },
                "embedding": {
                    "driver": "llama_cpp_embedding", "port": self.backend.server.port,
                    "health_path": "/custom/health", "command": "fake-server",
                },
            },
        }
        self.rm = RuntimeManager(self.config)
        self.rm.ensure = AsyncMock()
        self.auth = AuthManager(self.config)
        self.token = self.auth.create_token("router")["token"]
        self.headers = {"Authorization": f"Bearer {self.token}"}
        self.routes = RouteManager({"routes": {"routed-only": {
            "backends": [{"url": "http://must-not-be-contacted"}],
        }}})
        self.proxy = await self.start_app(create_api_app(self.rm, None, self.auth, self.routes))
        self.url = str(self.proxy.make_url("/")).rstrip("/")

    async def asyncTearDown(self):
        for client in reversed(self.clients):
            await client.close()
        self.rm.ensure.assert_not_awaited()

    async def start_app(self, app):
        client = TestClient(TestServer(app))
        await client.start_server()
        self.clients.append(client)
        return client

    def running(self, name="chat"):
        self.rm._processes[name] = SimpleNamespace(returncode=None, pid=123)

    async def get_health(self, suffix="", name="chat"):
        response = await self.proxy.get(f"/v1/models/{name}/health{suffix}", headers=self.headers)
        return response.status, await response.json()

    async def test_public_health_has_no_model_details_even_with_running_models(self):
        self.running()
        for headers in ({}, {"Authorization": "Bearer invalid"}):
            response = await self.proxy.get("/health", headers=headers)
            self.assertEqual(response.status, 200)
            self.assertEqual(await response.json(), {"status": "ok"})
        self.assertEqual(self.backend_calls, 0)

    async def test_capabilities_remain_public(self):
        response = await self.proxy.get("/v1/capabilities")
        self.assertEqual(response.status, 200)
        payload = await response.json()
        self.assertTrue(payload["auth_required"])
        self.assertTrue(payload["services"]["llm"])

    async def test_model_endpoints_require_api_token_not_web_session(self):
        self.auth.set_web_password("password")
        session = self.auth.web_login("password")
        for path in ("/v1/models", "/v1/models/chat/health", "/v1/models/missing/health"):
            for headers, cookies, expected in (
                ({}, {}, 401),
                ({}, {self.auth.session_cookie_name(): session}, 401),
                ({"Authorization": "Bearer invalid"}, {}, 403),
            ):
                with self.subTest(path=path, expected=expected):
                    response = await self.proxy.get(path, headers=headers, cookies=cookies)
                    self.assertEqual(response.status, expected)
        self.assertEqual(self.backend_calls, 0)

    async def test_model_catalog_accepts_valid_token_and_rejects_revoked_token(self):
        response = await self.proxy.get("/v1/models", headers=self.headers)
        self.assertEqual(response.status, 200)
        self.assertEqual([item["id"] for item in (await response.json())["data"]],
                         ["chat", "routed-only"])
        self.auth.revoke_token("router")
        for path in ("/v1/models", "/v1/models/chat/health"):
            self.assertEqual((await self.proxy.get(path, headers=self.headers)).status, 403)

    async def test_model_endpoints_are_open_when_api_auth_is_disabled(self):
        self.auth.set_api_auth_enabled(False)
        self.running()
        for path in ("/v1/models", "/v1/models/chat/health"):
            self.assertEqual((await self.proxy.get(path)).status, 200)

    async def test_stopped_model_is_unavailable_while_server_is_alive(self):
        status, body = await self.get_health("?allow_on_demand=true")
        self.assertEqual(status, 503)
        self.assertEqual(body, {
            "model": "chat", "status": "stopped", "ready": False,
            "available": False, "on_demand": False,
        })
        self.assertEqual((await self.proxy.get("/health")).status, 200)
        self.assertEqual(self.backend_calls, 0)

    async def test_on_demand_availability_does_not_start_model(self):
        self.rm.runtimes["chat"]["on_demand"] = True
        status, body = await self.get_health()
        self.assertEqual(status, 503)
        self.assertFalse(body["ready"])
        self.assertTrue(body["available"])
        status, body = await self.get_health("?allow_on_demand=true")
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "stopped")
        self.assertFalse(body["ready"])
        self.assertEqual(self.backend_calls, 0)

    async def test_ready_model_uses_driver_health_path(self):
        self.running()
        status, body = await self.get_health()
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "ready")
        self.assertTrue(body["ready"])
        self.assertTrue(body["available"])
        self.assertEqual(self.backend_calls, 1)

    async def test_health_accepts_exact_local_runtime_names_including_embedding(self):
        self.running("embedding")
        self.assertEqual((await self.get_health(name="embedding"))[0], 200)
        for name in ("missing", "cha", "routed-only"):
            status, body = await self.get_health(name=name)
            self.assertEqual(status, 404)
            self.assertEqual(body, {"error": "Unknown local model", "model": name})
        self.assertEqual(self.backend_calls, 1)

    async def test_live_process_with_unhealthy_backend_is_not_ready(self):
        self.running()
        self.rm.runtimes["chat"]["on_demand"] = True
        for code in (204, 302, 401, 403, 404, 500, 503):
            with self.subTest(code=code):
                self.backend_status = code
                status, body = await self.get_health("?allow_on_demand=true")
                self.assertEqual(status, 503)
                self.assertEqual(body["status"], "unavailable")
                self.assertFalse(body["ready"])
                self.assertFalse(body["available"])

    async def test_unreachable_backend_is_not_ready(self):
        self.running()
        await self.backend.close()
        status, body = await self.get_health()
        self.assertEqual(status, 503)
        self.assertEqual(body["status"], "unavailable")

    async def test_model_health_timeout_is_bounded(self):
        self.running()
        with patch("llm_proxy.router.aiohttp.ClientSession.get", side_effect=asyncio.TimeoutError):
            status, body = await self.get_health()
        self.assertEqual(status, 503)
        self.assertEqual(body["status"], "unavailable")

    async def test_startup_is_not_reported_ready_until_start_finishes(self):
        entered = asyncio.Event()
        release = asyncio.Event()

        async def wait_healthy(*args, **kwargs):
            entered.set()
            await release.wait()
            return True

        proc = SimpleNamespace(returncode=None, pid=123)
        with patch("llm_proxy.runtime_manager.asyncio.create_subprocess_exec", new=AsyncMock(return_value=proc)), \
                patch.object(self.rm, "_log_output", new=AsyncMock()), \
                patch.object(self.rm, "_wait_healthy", side_effect=wait_healthy):
            task = asyncio.create_task(self.rm.start("chat"))
            try:
                await asyncio.wait_for(entered.wait(), 1)
                status, body = await self.get_health()
                self.assertEqual(status, 503)
                self.assertEqual(body["status"], "starting")
                self.assertEqual(self.backend_calls, 0)
            finally:
                release.set()
                await task
        self.assertFalse(self.rm.is_starting("chat"))
        self.assertEqual((await self.get_health())[0], 200)

    async def test_startup_failure_clears_starting_flag(self):
        proc = SimpleNamespace(returncode=None, pid=123)
        with patch("llm_proxy.runtime_manager.asyncio.create_subprocess_exec", new=AsyncMock(return_value=proc)), \
                patch.object(self.rm, "_log_output", new=AsyncMock()), \
                patch.object(self.rm, "_wait_healthy", new=AsyncMock(return_value=False)), \
                patch.object(self.rm, "stop", new=AsyncMock()) as stop:
            with self.assertRaises(TimeoutError):
                await self.rm.start("chat")
        self.assertFalse(self.rm.is_starting("chat"))
        stop.assert_awaited_once_with("chat")

    def make_route(self, **kwargs):
        return Route("alias", [{"url": self.url, "model": "chat", "auth": "remote"}],
                     hosts={"remote": {"token": self.token}},
                     health_path="/v1/models/{model}/health" + kwargs.get("query", ""))

    async def test_routing_checks_target_model_and_sends_auth(self):
        route = self.make_route()
        self.assertFalse((await route.probe(self.url))[0])
        self.running()
        self.assertTrue((await route.probe(self.url))[0])
        self.backend_status = 503
        self.assertFalse((await route.probe(self.url))[0])
        self.assertFalse(route.status()["backends"][0]["healthy"])

    async def test_routing_rejects_missing_invalid_auth_and_unknown_model(self):
        self.running()
        for field, value in (("auth", None), ("auth", "missing"), ("model", "missing")):
            route = self.make_route()
            route.backends[0][field] = value
            self.assertFalse((await route.probe(self.url))[0])
        route = self.make_route()
        route.hosts["remote"]["token"] = "invalid"
        self.assertFalse((await route.probe(self.url))[0])

    async def test_routing_can_include_on_demand_models_without_starting(self):
        self.rm.runtimes["chat"]["on_demand"] = True
        self.assertFalse((await self.make_route().probe(self.url))[0])
        route = self.make_route(query="?allow_on_demand=true")
        self.assertTrue((await route.probe(self.url))[0])
        self.assertEqual(self.backend_calls, 0)

    async def test_routing_marks_unreachable_proxy_unhealthy(self):
        route = self.make_route()
        await self.proxy.close()
        self.assertFalse((await route.probe(self.url))[0])

    async def test_routing_prefers_ready_model_over_alive_proxy_with_stopped_model(self):
        ready_rm = RuntimeManager(self.config)
        ready_rm._processes["chat"] = SimpleNamespace(returncode=None, pid=456)
        ready_proxy = await self.start_app(create_api_app(ready_rm, None, self.auth))
        ready_url = str(ready_proxy.make_url("/")).rstrip("/")
        route = Route("alias", [
            {"url": self.url, "model": "chat", "auth": "remote"},
            {"url": ready_url, "model": "chat", "auth": "remote"},
        ], hosts={"remote": {"token": self.token}},
            health_path="/v1/models/{model}/health")
        route.note_request_start("session", self.url)
        route.note_request_end("session", self.url)

        results = await route.probe_all()

        self.assertEqual([ok for _, ok, _ in results], [False, True])
        self.assertEqual(route.active_url(), ready_url)
        self.assertEqual(route.failover_candidates(session_id="session")[0]["url"], ready_url)
        self.assertEqual((await self.proxy.get("/health")).status, 200)


if __name__ == "__main__":
    unittest.main()
