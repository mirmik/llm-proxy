import asyncio
import os
import sqlite3
import tempfile
import time
import unittest

from llm_proxy.statistics import RequestTrace, SSEUsageParser, StatisticsStore, apply_usage


def request_record(request_id, started_at=None):
    if started_at is None:
        started_at = time.time()
    return {
        "request_id": request_id,
        "started_at": started_at,
        "completed_at": started_at + 0.125,
        "duration_ms": 125,
        "kind": "chat",
        "client": "test-client",
        "requested_model": "route-name",
        "resolved_model": "backend-model",
        "route": "route-name",
        "stream": False,
        "status": 200,
        "outcome": "success",
        "client_input_bytes": 12,
        "client_output_bytes": 34,
        "upstream_input_bytes": 12,
        "upstream_output_bytes": 34,
        "prompt_tokens": 3,
        "completion_tokens": 5,
        "total_tokens": 8,
        "usage_available": True,
        "attempt_count": 1,
    }


def attempt_record(request_id, attempt_no=1):
    return {
        "request_id": request_id,
        "attempt_no": attempt_no,
        "started_at": 100.0,
        "completed_at": 100.125,
        "duration_ms": 125,
        "kind": "chat",
        "model": "backend-model",
        "route": "route-name",
        "backend": "http://backend:8090/v1/chat/completions",
        "route_reason": "least-loaded",
        "status": 200,
        "outcome": "success",
        "input_bytes": 12,
        "output_bytes": 34,
    }


class StatisticsStoreTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.config = {
            "state_dir": self.tmp.name,
            "statistics": {"enabled": True},
        }
        self.store = StatisticsStore(self.config)
        await self.store.start()

    async def asyncTearDown(self):
        await self.store.stop()
        self.tmp.cleanup()

    async def test_records_requests_and_attempts(self):
        self.assertTrue(self.store.record_request(request_record("req-1")))
        self.assertTrue(self.store.record_attempt(attempt_record("req-1")))
        await self.store.flush()

        self.assertEqual(self.store.request_count(), 1)
        self.assertEqual(self.store.attempt_count(), 1)
        request = self.store.fetch_requests()[0]
        self.assertEqual(request["client_input_bytes"], 12)
        self.assertEqual(request["total_tokens"], 8)
        self.assertEqual(self.store.fetch_attempts()[0]["route_reason"], "least-loaded")

    async def test_serializes_concurrent_producers(self):
        async def producer(index):
            await asyncio.sleep(0)
            self.store.record_request(request_record(f"req-{index}", started_at=100 + index))

        await asyncio.gather(*(producer(index) for index in range(50)))
        await self.store.flush()

        self.assertEqual(self.store.request_count(), 50)

    async def test_stop_flushes_accepted_events(self):
        for index in range(20):
            self.store.record_request(request_record(f"req-{index}"))

        await self.store.stop()

        connection = sqlite3.connect(self.store.path)
        try:
            count = connection.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
        finally:
            connection.close()
        self.assertEqual(count, 20)
        self.assertFalse(self.store.record_request(request_record("too-late")))

        # asyncTearDown may safely call stop again.
        await self.store.start()

    async def test_restart_preserves_history_and_schema_is_idempotent(self):
        path = self.store.path
        self.store.record_request(request_record("before-restart"))
        await self.store.stop()

        restarted = StatisticsStore(self.config)
        await restarted.start()
        try:
            self.assertEqual(restarted.path, path)
            self.assertEqual(restarted.request_count(), 1)
            restarted.record_request(request_record("after-restart"))
            await restarted.flush()
            self.assertEqual(restarted.request_count(), 2)
        finally:
            await restarted.stop()

        await self.store.start()

    async def test_disabled_store_does_not_create_database(self):
        disabled_dir = os.path.join(self.tmp.name, "disabled")
        disabled = StatisticsStore({
            "state_dir": disabled_dir,
            "statistics": {"enabled": False},
        })

        await disabled.start()
        self.assertFalse(disabled.record_request(request_record("ignored")))
        await disabled.stop()
        self.assertFalse(os.path.exists(disabled.path))

    async def test_request_trace_separates_one_request_from_failover_attempts(self):
        trace = RequestTrace(
            self.store,
            kind="chat",
            client="client-a",
            requested_model="route-name",
            input_bytes=10,
        )
        trace.route = "route-name"
        failed = trace.start_attempt(
            model="model-a", route="route-name", backend="backend-a",
            route_reason="least-loaded", input_bytes=8,
        )
        failed.finish(503, error=ConnectionError(), output_bytes=3)
        success = trace.start_attempt(
            model="model-b", route="route-name", backend="backend-b",
            route_reason="failover", input_bytes=9,
        )
        success.finish(200, output_bytes=20)
        trace.resolved_model = "model-b"
        trace.finish(200, output_bytes=20)
        await self.store.flush()

        request = self.store.fetch_requests()[0]
        self.assertEqual(request["attempt_count"], 2)
        self.assertEqual(request["upstream_input_bytes"], 17)
        self.assertEqual(request["upstream_output_bytes"], 23)
        self.assertEqual(self.store.attempt_count(), 2)
        self.assertEqual(self.store.fetch_attempts()[0]["error"], "ConnectionError")

    async def test_usage_is_observed_without_estimating_missing_fields(self):
        trace = RequestTrace(self.store, kind="chat")
        self.assertTrue(apply_usage(trace, {
            "usage": {"prompt_tokens": 4, "completion_tokens": 6, "total_tokens": 10}
        }))
        trace.finish(200)
        await self.store.flush()

        row = self.store.fetch_requests()[0]
        self.assertEqual(row["prompt_tokens"], 4)
        self.assertEqual(row["completion_tokens"], 6)
        self.assertEqual(row["total_tokens"], 10)
        self.assertEqual(row["usage_available"], 1)

    async def test_sse_usage_parser_handles_fragmented_lines(self):
        trace = RequestTrace(self.store, kind="chat", stream=True)
        parser = SSEUsageParser(trace)
        parser.feed(b'data: {"choices": [{"delta": {"content": "x"}}]}\n\n')
        parser.feed(b'data: {"choices": [], "usage": {"prompt_tokens": 7,')
        parser.feed(b' "completion_tokens": 2, "total_tokens": 9}}\n\ndata: [DONE]\n\n')
        parser.finish()
        trace.finish(200)
        await self.store.flush()

        row = self.store.fetch_requests()[0]
        self.assertEqual(row["total_tokens"], 9)
        self.assertEqual(row["usage_available"], 1)

    async def test_missing_or_incomplete_usage_remains_unobserved(self):
        trace = RequestTrace(self.store, kind="chat", stream=True)
        parser = SSEUsageParser(trace)
        parser.feed(b'data: {"choices":[{"delta":{"content":"not counted"}}]}\n\n')
        parser.feed(b'data: {"usage":{"completion_tokens":"unknown"}}\n\n')
        parser.finish()
        trace.finish(499, error="client-disconnect")
        await self.store.flush()

        row = self.store.fetch_requests()[0]
        self.assertIsNone(row["prompt_tokens"])
        self.assertIsNone(row["completion_tokens"])
        self.assertIsNone(row["total_tokens"])
        self.assertEqual(row["usage_available"], 0)

    async def test_cleanup_removes_only_rows_older_than_retention(self):
        self.store.retention_days = 1
        old = request_record("old", started_at=100.0)
        old["completed_at"] = 101.0
        recent = request_record("recent", started_at=100000.0)
        self.store.record_request(old)
        self.store.record_attempt(attempt_record("old"))
        self.store.record_request(recent)
        await self.store.flush()

        deleted = await self.store.cleanup(now_timestamp=100000.0 + 100)

        self.assertEqual(deleted, {"requests": 1, "attempts": 1})
        self.assertEqual([row["request_id"] for row in self.store.fetch_requests()], ["recent"])

    async def test_clear_preserves_schema_for_new_events(self):
        self.store.record_request(request_record("before-clear"))
        await self.store.flush()
        self.assertEqual(await self.store.clear(), {"requests": 1, "attempts": 0})
        self.store.record_request(request_record("after-clear"))
        await self.store.flush()
        self.assertEqual(self.store.request_count(), 1)


if __name__ == "__main__":
    unittest.main()
