import asyncio
import unittest
from unittest.mock import patch

from llm_proxy.route_manager import (
    Route,
    RouteManager,
    _derive_endpoint_url,
    backend_client_kwargs,
)


class _DummyResponse:
    status = 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False


class _DummyClientSession:
    def __init__(self):
        self.closed = False
        self.requests = []

    def get(self, url, **kwargs):
        self.requests.append((url, kwargs))
        return _DummyResponse()

    async def close(self):
        self.closed = True


class RouteSessionDispatchTests(unittest.IsolatedAsyncioTestCase):
    def test_embedding_endpoint_is_openai_compatible(self):
        self.assertEqual(
            _derive_endpoint_url("https://host:8090", "embedding"),
            "https://host:8090/v1/embeddings",
        )

    async def test_tls_verify_false_disables_aiohttp_ssl_verification(self):
        kwargs = backend_client_kwargs({"url": "https://host:8090", "tls_verify": False})
        connector = kwargs.get("connector")

        self.assertIsNotNone(connector)
        self.assertFalse(connector._ssl)
        await connector.close()

    async def test_probe_reuses_provided_session_with_per_request_tls_setting(self):
        session = _DummyClientSession()
        route = Route(
            "model",
            [{"url": "https://host:8090", "tls_verify": False}],
        )

        ok, latency = await route.probe("https://host:8090", session=session)
        second_ok, _ = await route.probe("https://host:8090", session=session)

        self.assertTrue(ok)
        self.assertTrue(second_ok)
        self.assertIsNotNone(latency)
        self.assertFalse(session.closed)
        self.assertEqual(len(session.requests), 2)
        self.assertIs(session.requests[0][1]["ssl"], False)

    async def test_monitor_closes_shared_health_session_on_stop(self):
        session = _DummyClientSession()
        manager = RouteManager({
            "routes": {
                "model": {
                    "backends": [{"url": "http://host:8090"}],
                    "check_interval": 60,
                },
            },
        })

        with patch("llm_proxy.route_manager.aiohttp.ClientSession", return_value=session) as factory:
            await manager.start_monitoring()
            await asyncio.sleep(0)
            await manager.stop_monitoring()

        factory.assert_called_once_with()
        self.assertTrue(session.closed)
        self.assertIsNone(manager._health_session)

    def test_new_sessions_use_least_loaded_backend(self):
        route = Route(
            "model",
            [
                {"url": "http://host-a:8090"},
                {"url": "http://host-b:8090"},
            ],
        )

        first = route.failover_candidates(session_id="session-a")[0]
        self.assertEqual(first["url"], "http://host-a:8090")
        route.note_request_start("session-a", first["url"])

        second = route.failover_candidates(session_id="session-b")[0]
        self.assertEqual(second["url"], "http://host-b:8090")
        self.assertEqual(second["_route_reason"], "least-loaded")

    def test_existing_session_stays_sticky(self):
        route = Route(
            "model",
            [
                {"url": "http://host-a:8090"},
                {"url": "http://host-b:8090"},
            ],
        )

        route.note_request_start("session-a", "http://host-a:8090")
        route.note_request_end("session-a", "http://host-a:8090")

        candidate = route.failover_candidates(session_id="session-a")[0]

        self.assertEqual(candidate["url"], "http://host-a:8090")
        self.assertEqual(candidate["_route_reason"], "sticky")

    async def test_unhealthy_sticky_backend_is_not_used(self):
        route = Route(
            "model",
            [
                {"url": "http://host-a:8090"},
                {"url": "http://host-b:8090"},
            ],
        )
        route.note_request_start("session-a", "http://host-a:8090")
        route.note_request_end("session-a", "http://host-a:8090")

        await route.mark_unhealthy("http://host-a:8090")
        candidate = route.failover_candidates(session_id="session-a")[0]

        self.assertEqual(candidate["url"], "http://host-b:8090")
        self.assertEqual(candidate["_route_reason"], "least-loaded")

    def test_backend_weight_reduces_effective_load(self):
        route = Route(
            "model",
            [
                {"url": "http://small:8090", "weight": 1},
                {"url": "http://large:8090", "weight": 4},
            ],
        )
        route.note_request_start("a", "http://small:8090")
        route.note_request_start("b", "http://large:8090")

        candidate = route.failover_candidates(session_id="c")[0]

        self.assertEqual(candidate["url"], "http://large:8090")

    def test_overflow_backend_waits_until_primary_capacity_is_full(self):
        route = Route(
            "model",
            [
                {"url": "http://fast:8090", "capacity": 2},
                {"url": "http://slow:8090", "capacity": 1, "overflow_only": True},
            ],
        )
        route.note_request_start("a", "http://fast:8090")

        candidate = route.failover_candidates(session_id="b")[0]

        self.assertEqual(candidate["url"], "http://fast:8090")

    def test_overflow_backend_is_used_when_primary_capacity_is_full(self):
        route = Route(
            "model",
            [
                {"url": "http://fast:8090", "capacity": 2},
                {"url": "http://slow:8090", "capacity": 1, "overflow_only": True},
            ],
        )
        route.note_request_start("a", "http://fast:8090")
        route.note_request_start("b", "http://fast:8090")

        candidate = route.failover_candidates(session_id="c")[0]

        self.assertEqual(candidate["url"], "http://slow:8090")

    def test_sticky_overflow_session_rebalances_to_free_primary(self):
        route = Route(
            "model",
            [
                {"url": "http://fast:8090", "weight": 4, "capacity": 2},
                {"url": "http://slow:8090", "weight": 1, "capacity": 1, "overflow_only": True},
            ],
            sticky_rebalance=True,
        )
        route.note_request_start("session-a", "http://slow:8090")
        route.note_request_end("session-a", "http://slow:8090")

        candidate = route.failover_candidates(session_id="session-a")[0]

        self.assertEqual(candidate["url"], "http://fast:8090")
        self.assertEqual(candidate["_route_reason"], "rebalance")

    def test_recently_migrated_session_respects_rebalance_cooldown(self):
        route = Route(
            "model",
            [
                {"url": "http://fast:8090", "weight": 4, "capacity": 2},
                {"url": "http://slow:8090", "weight": 1, "capacity": 1, "overflow_only": True},
            ],
            sticky_rebalance=True,
            rebalance_cooldown=60,
        )
        route.note_request_start("session-a", "http://slow:8090")
        route.note_request_end("session-a", "http://slow:8090")
        route._session_migrated["session-a"] = 9999999999

        candidate = route.failover_candidates(session_id="session-a")[0]

        self.assertEqual(candidate["url"], "http://slow:8090")
        self.assertEqual(candidate["_route_reason"], "sticky")


if __name__ == "__main__":
    unittest.main()
