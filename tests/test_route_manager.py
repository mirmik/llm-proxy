import unittest

from llm_proxy.route_manager import Route


class RouteSessionDispatchTests(unittest.IsolatedAsyncioTestCase):
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


if __name__ == "__main__":
    unittest.main()
