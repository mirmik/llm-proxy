"""Persistent request statistics for llm-proxy.

The store deliberately records metadata and counters only.  Request/response
content, credentials, and session identifiers do not belong in this database.
"""

import asyncio
import json
import os
import sqlite3
import time
import uuid
from collections import defaultdict
from pathlib import Path


_STOP = object()

_REQUEST_COLUMNS = (
    "request_id",
    "started_at",
    "completed_at",
    "duration_ms",
    "kind",
    "client",
    "host",
    "requested_model",
    "resolved_model",
    "runtime",
    "route",
    "stream",
    "status",
    "outcome",
    "error",
    "client_input_bytes",
    "client_output_bytes",
    "upstream_input_bytes",
    "upstream_output_bytes",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "usage_available",
    "ttft_ms",
    "attempt_count",
)

_ATTEMPT_COLUMNS = (
    "request_id",
    "attempt_no",
    "started_at",
    "completed_at",
    "duration_ms",
    "kind",
    "model",
    "runtime",
    "route",
    "backend",
    "route_reason",
    "status",
    "outcome",
    "error",
    "input_bytes",
    "output_bytes",
)


class StatisticsStore:
    """Serialize statistics writes through one asyncio worker into SQLite."""

    def __init__(self, config):
        statistics = config.get("statistics") or {}
        self.enabled = bool(statistics.get("enabled", True))
        state_dir = os.path.expanduser(config["state_dir"])
        configured_path = statistics.get("database")
        self.path = os.path.expanduser(configured_path) if configured_path else os.path.join(
            state_dir, "statistics.sqlite3"
        )
        self.retention_days = float(statistics.get("retention_days", 30))
        self.cleanup_interval_seconds = float(statistics.get("cleanup_interval_hours", 24)) * 3600
        self._queue = None
        self._worker_task = None
        self._connection = None
        self._error = None
        self._accepting = False
        self._maintenance_task = None
        self._inflight = 0

    async def start(self):
        if not self.enabled or self._worker_task is not None:
            return
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=NORMAL")
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._initialize_schema()
        self._queue = asyncio.Queue()
        self._accepting = True
        self._worker_task = asyncio.create_task(self._run_writer())
        await self.cleanup()
        self._maintenance_task = asyncio.create_task(self._run_maintenance())

    async def stop(self):
        if self._worker_task is None:
            return
        self._accepting = False
        if self._maintenance_task is not None:
            self._maintenance_task.cancel()
            await asyncio.gather(self._maintenance_task, return_exceptions=True)
            self._maintenance_task = None
        await self.flush()
        await self._queue.put(_STOP)
        await self._worker_task
        self._worker_task = None
        self._queue = None
        self._connection.close()
        self._connection = None
        self._raise_writer_error()

    async def flush(self):
        if self._queue is not None:
            await self._queue.join()
        self._raise_writer_error()

    def record_request(self, values):
        """Queue one completed client request. Returns False while disabled/stopped."""
        return self._enqueue("request", values)

    def record_attempt(self, values):
        """Queue one completed upstream attempt. Returns False while disabled/stopped."""
        return self._enqueue("attempt", values)

    def request_count(self):
        return self._scalar("SELECT COUNT(*) FROM requests")

    def attempt_count(self):
        return self._scalar("SELECT COUNT(*) FROM backend_attempts")

    @property
    def inflight(self):
        return self._inflight

    def note_request_start(self):
        self._inflight += 1

    def note_request_end(self):
        self._inflight = max(0, self._inflight - 1)

    def fetch_requests(self):
        self._require_started()
        return [dict(row) for row in self._connection.execute(
            "SELECT * FROM requests ORDER BY started_at, request_id"
        )]

    def fetch_attempts(self):
        self._require_started()
        return [dict(row) for row in self._connection.execute(
            "SELECT * FROM backend_attempts ORDER BY started_at, request_id, attempt_no"
        )]

    async def cleanup(self, now_timestamp=None):
        """Delete detailed request and attempt rows older than retention_days."""
        if not self.enabled or self._connection is None or self.retention_days <= 0:
            return {"requests": 0, "attempts": 0}
        await self.flush()
        cutoff = (unix_time() if now_timestamp is None else float(now_timestamp)) - self.retention_days * 86400
        attempt_cursor = self._connection.execute(
            "DELETE FROM backend_attempts WHERE started_at < ?", (cutoff,)
        )
        request_cursor = self._connection.execute(
            "DELETE FROM requests WHERE started_at < ?", (cutoff,)
        )
        self._connection.commit()
        return {
            "requests": max(request_cursor.rowcount, 0),
            "attempts": max(attempt_cursor.rowcount, 0),
        }

    async def clear(self):
        """Remove statistics rows while preserving the schema."""
        if not self.enabled or self._connection is None:
            return {"requests": 0, "attempts": 0}
        await self.flush()
        attempts = self.attempt_count()
        requests = self.request_count()
        self._connection.execute("DELETE FROM backend_attempts")
        self._connection.execute("DELETE FROM requests")
        self._connection.commit()
        return {"requests": requests, "attempts": attempts}

    def overview(self, *, period_seconds, bucket_seconds, now_timestamp=None):
        """Build the control-panel aggregate payload from retained metadata."""
        self._require_started()
        now_timestamp = unix_time() if now_timestamp is None else float(now_timestamp)
        if period_seconds is None:
            first = self._connection.execute("SELECT MIN(started_at) FROM requests").fetchone()[0]
            cutoff = first if first is not None else now_timestamp
        else:
            cutoff = now_timestamp - float(period_seconds)
        request_rows = [dict(row) for row in self._connection.execute(
            "SELECT * FROM requests WHERE started_at >= ? ORDER BY started_at", (cutoff,)
        )]
        attempt_rows = [dict(row) for row in self._connection.execute(
            "SELECT * FROM backend_attempts WHERE started_at >= ? ORDER BY started_at", (cutoff,)
        )]

        durations = [row["duration_ms"] for row in request_rows]
        successful = sum(1 for row in request_rows if _is_success(row.get("status")))
        usage_rows = [row for row in request_rows if row.get("usage_available")]
        generation_rows = [
            row for row in usage_rows
            if row.get("completion_tokens") is not None
            and row.get("ttft_ms") is not None
            and row.get("duration_ms", 0) > row.get("ttft_ms", 0)
        ]
        generation_seconds = sum(
            (row["duration_ms"] - row["ttft_ms"]) / 1000 for row in generation_rows
        )
        summary = {
            "requests": len(request_rows),
            "successful": successful,
            "errors": len(request_rows) - successful,
            "inflight": self.inflight,
            "attempts": len(attempt_rows),
            "failovers": sum(1 for row in attempt_rows if row.get("route_reason") == "failover"),
            "client_input_bytes": _sum(request_rows, "client_input_bytes"),
            "client_output_bytes": _sum(request_rows, "client_output_bytes"),
            "upstream_input_bytes": _sum(attempt_rows, "input_bytes"),
            "upstream_output_bytes": _sum(attempt_rows, "output_bytes"),
            "prompt_tokens": _nullable_sum(usage_rows, "prompt_tokens"),
            "completion_tokens": _nullable_sum(usage_rows, "completion_tokens"),
            "total_tokens": _nullable_sum(usage_rows, "total_tokens"),
            "usage_coverage": (len(usage_rows) / len(request_rows)) if request_rows else 0.0,
            "tokens_per_second": (
                sum(row["completion_tokens"] for row in generation_rows) / generation_seconds
                if generation_seconds > 0 else None
            ),
            "latency_ms": {
                "avg": round(sum(durations) / len(durations), 1) if durations else None,
                "p50": _percentile(durations, 0.50),
                "p95": _percentile(durations, 0.95),
                "max": max(durations) if durations else None,
            },
        }
        return {
            "enabled": self.enabled,
            "retention_days": self.retention_days,
            "range": {"from": cutoff, "to": now_timestamp, "bucket_seconds": bucket_seconds},
            "summary": summary,
            "series": _time_series(request_rows, cutoff, now_timestamp, bucket_seconds),
            "breakdown": {
                "kind": _group_rows(request_rows, "kind", traffic="client"),
                "model": _group_rows(request_rows, "resolved_model", traffic="client"),
                "runtime": _group_rows(request_rows, "runtime", traffic="client"),
                "route": _group_rows(request_rows, "route", traffic="client"),
                "client": _group_rows(request_rows, "client", traffic="client"),
                "backend": _group_rows(attempt_rows, "backend", traffic="upstream"),
                "routing_reason": _group_rows(attempt_rows, "route_reason", traffic="upstream"),
            },
            "recent_errors": _recent(request_rows, errors_only=True),
            "slow_requests": _recent(sorted(request_rows, key=lambda row: row["duration_ms"]), limit=10, reverse=True),
        }

    def _enqueue(self, operation, values):
        if not self.enabled or not self._accepting or self._queue is None:
            return False
        self._raise_writer_error()
        self._queue.put_nowait((operation, dict(values)))
        return True

    async def _run_writer(self):
        while True:
            item = await self._queue.get()
            try:
                if item is _STOP:
                    return
                operation, values = item
                if operation == "request":
                    self._insert("requests", _REQUEST_COLUMNS, values)
                else:
                    self._insert("backend_attempts", _ATTEMPT_COLUMNS, values)
                self._connection.commit()
            except Exception as exc:
                if self._error is None:
                    self._error = exc
                self._connection.rollback()
            finally:
                self._queue.task_done()

    async def _run_maintenance(self):
        interval = max(self.cleanup_interval_seconds, 60.0)
        while True:
            await asyncio.sleep(interval)
            await self.cleanup()

    def _insert(self, table, columns, values):
        unknown = set(values) - set(columns)
        if unknown:
            raise ValueError(f"Unknown {table} fields: {sorted(unknown)}")
        selected = [column for column in columns if column in values]
        if not selected:
            raise ValueError(f"No values supplied for {table}")
        placeholders = ", ".join("?" for _ in selected)
        column_sql = ", ".join(selected)
        verb = "INSERT OR REPLACE" if table == "requests" else "INSERT"
        self._connection.execute(
            f"{verb} INTO {table} ({column_sql}) VALUES ({placeholders})",
            [values[column] for column in selected],
        )

    def _initialize_schema(self):
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS requests (
                request_id TEXT PRIMARY KEY,
                started_at REAL NOT NULL,
                completed_at REAL NOT NULL,
                duration_ms INTEGER NOT NULL,
                kind TEXT NOT NULL,
                client TEXT,
                host TEXT,
                requested_model TEXT,
                resolved_model TEXT,
                runtime TEXT,
                route TEXT,
                stream INTEGER NOT NULL DEFAULT 0,
                status INTEGER,
                outcome TEXT NOT NULL,
                error TEXT,
                client_input_bytes INTEGER NOT NULL DEFAULT 0,
                client_output_bytes INTEGER NOT NULL DEFAULT 0,
                upstream_input_bytes INTEGER NOT NULL DEFAULT 0,
                upstream_output_bytes INTEGER NOT NULL DEFAULT 0,
                prompt_tokens INTEGER,
                completion_tokens INTEGER,
                total_tokens INTEGER,
                usage_available INTEGER NOT NULL DEFAULT 0,
                ttft_ms INTEGER,
                attempt_count INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS backend_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT NOT NULL,
                attempt_no INTEGER NOT NULL,
                started_at REAL NOT NULL,
                completed_at REAL NOT NULL,
                duration_ms INTEGER NOT NULL,
                kind TEXT NOT NULL,
                model TEXT,
                runtime TEXT,
                route TEXT,
                backend TEXT,
                route_reason TEXT,
                status INTEGER,
                outcome TEXT NOT NULL,
                error TEXT,
                input_bytes INTEGER NOT NULL DEFAULT 0,
                output_bytes INTEGER NOT NULL DEFAULT 0,
                UNIQUE(request_id, attempt_no)
            );

            CREATE INDEX IF NOT EXISTS requests_started_at_idx
                ON requests(started_at);
            CREATE INDEX IF NOT EXISTS requests_kind_model_idx
                ON requests(kind, resolved_model, started_at);
            CREATE INDEX IF NOT EXISTS requests_route_idx
                ON requests(route, started_at);
            CREATE INDEX IF NOT EXISTS attempts_started_at_idx
                ON backend_attempts(started_at);
            CREATE INDEX IF NOT EXISTS attempts_route_backend_idx
                ON backend_attempts(route, backend, started_at);
            PRAGMA user_version=1;
            """
        )
        self._connection.commit()

    def _scalar(self, sql, parameters=()):
        self._require_started()
        return self._connection.execute(sql, parameters).fetchone()[0]

    def _require_started(self):
        if not self.enabled:
            raise RuntimeError("Statistics are disabled")
        if self._connection is None:
            raise RuntimeError("StatisticsStore is not started")
        self._raise_writer_error()

    def _raise_writer_error(self):
        if self._error is not None:
            raise RuntimeError("Statistics writer failed") from self._error


def unix_time():
    """Wall-clock timestamp helper kept here to make event construction consistent."""
    return time.time()


def error_label(error):
    """Return a low-cardinality, content-free error label."""
    if error is None:
        return None
    if isinstance(error, str):
        return error if error in {"client-disconnect", "rejected", "all-backends-failed"} else "error"
    return type(error).__name__


class RequestTrace:
    """Mutable per-request counters that emit immutable completion records."""

    def __init__(self, store, *, kind, client=None, host=None, requested_model=None,
                 stream=False, input_bytes=0):
        self.store = store
        self.request_id = uuid.uuid4().hex
        self.started_at = unix_time()
        self._started = time.perf_counter()
        self.kind = kind
        self.client = client
        self.host = host
        self.requested_model = requested_model
        self.resolved_model = None
        self.runtime = None
        self.route = None
        self.stream = bool(stream)
        self.client_input_bytes = int(input_bytes or 0)
        self.client_output_bytes = 0
        self.upstream_input_bytes = 0
        self.upstream_output_bytes = 0
        self.prompt_tokens = None
        self.completion_tokens = None
        self.total_tokens = None
        self.usage_available = False
        self.ttft_ms = None
        self._attempt_count = 0
        self._finished = False
        self.store.note_request_start()

    def start_attempt(self, *, model=None, runtime=None, route=None, backend=None,
                      route_reason=None, input_bytes=0):
        self._attempt_count += 1
        self.upstream_input_bytes += int(input_bytes or 0)
        return AttemptTrace(
            self,
            attempt_no=self._attempt_count,
            model=model,
            runtime=runtime,
            route=route,
            backend=backend,
            route_reason=route_reason,
            input_bytes=input_bytes,
        )

    def finish(self, status, *, outcome=None, error=None, output_bytes=0):
        if self._finished:
            return False
        self._finished = True
        self.store.note_request_end()
        self.client_output_bytes = int(output_bytes or 0)
        completed_at = unix_time()
        successful = status is not None and 200 <= int(status) < 400
        self.store.record_request({
            "request_id": self.request_id,
            "started_at": self.started_at,
            "completed_at": completed_at,
            "duration_ms": int((time.perf_counter() - self._started) * 1000),
            "kind": self.kind,
            "client": self.client,
            "host": self.host,
            "requested_model": self.requested_model,
            "resolved_model": self.resolved_model,
            "runtime": self.runtime,
            "route": self.route,
            "stream": self.stream,
            "status": status,
            "outcome": outcome or ("success" if successful else "error"),
            "error": error_label(error),
            "client_input_bytes": self.client_input_bytes,
            "client_output_bytes": self.client_output_bytes,
            "upstream_input_bytes": self.upstream_input_bytes,
            "upstream_output_bytes": self.upstream_output_bytes,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "usage_available": self.usage_available,
            "ttft_ms": self.ttft_ms,
            "attempt_count": self._attempt_count,
        })
        return True


class AttemptTrace:
    def __init__(self, request_trace, *, attempt_no, model, runtime, route, backend,
                 route_reason, input_bytes):
        self.request = request_trace
        self.attempt_no = attempt_no
        self.started_at = unix_time()
        self._started = time.perf_counter()
        self.model = model
        self.runtime = runtime
        self.route = route
        self.backend = backend
        self.route_reason = route_reason
        self.input_bytes = int(input_bytes or 0)
        self._finished = False

    def finish(self, status, *, outcome=None, error=None, output_bytes=0):
        if self._finished:
            return False
        self._finished = True
        output_bytes = int(output_bytes or 0)
        self.request.upstream_output_bytes += output_bytes
        successful = status is not None and 200 <= int(status) < 400
        self.request.store.record_attempt({
            "request_id": self.request.request_id,
            "attempt_no": self.attempt_no,
            "started_at": self.started_at,
            "completed_at": unix_time(),
            "duration_ms": int((time.perf_counter() - self._started) * 1000),
            "kind": self.request.kind,
            "model": self.model,
            "runtime": self.runtime,
            "route": self.route,
            "backend": self.backend,
            "route_reason": self.route_reason,
            "status": status,
            "outcome": outcome or ("success" if successful else "error"),
            "error": error_label(error),
            "input_bytes": self.input_bytes,
            "output_bytes": output_bytes,
        })
        return True


def apply_usage(trace, payload):
    """Copy an OpenAI-compatible usage object into a request trace."""
    if trace is None or not isinstance(payload, dict):
        return False
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return False
    fields = ("prompt_tokens", "completion_tokens", "total_tokens")
    observed = False
    for field in fields:
        value = usage.get(field)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            setattr(trace, field, value)
            observed = True
    trace.usage_available = observed
    return observed


class SSEUsageParser:
    """Incrementally inspect SSE data lines without retaining streamed content."""

    def __init__(self, trace):
        self.trace = trace
        self._buffer = b""

    def feed(self, chunk):
        self._buffer += chunk
        while b"\n" in self._buffer:
            line, self._buffer = self._buffer.split(b"\n", 1)
            self._consume_line(line.rstrip(b"\r"))

    def finish(self):
        if self._buffer:
            self._consume_line(self._buffer.rstrip(b"\r"))
            self._buffer = b""

    def _consume_line(self, line):
        if not line.startswith(b"data:"):
            return
        data = line[5:].strip()
        if not data or data == b"[DONE]":
            return
        try:
            payload = json.loads(data)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return
        apply_usage(self.trace, payload)


def _is_success(status):
    return status is not None and 200 <= int(status) < 400


def _sum(rows, field):
    return sum(int(row.get(field) or 0) for row in rows)


def _nullable_sum(rows, field):
    values = [row[field] for row in rows if row.get(field) is not None]
    return sum(values) if values else None


def _percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * fraction + 0.5)))
    return ordered[index]


def _group_rows(rows, field, *, traffic):
    groups = defaultdict(list)
    for row in rows:
        label = row.get(field)
        if label is not None and label != "":
            groups[str(label)].append(row)
    result = []
    total = len(rows)
    for label, group in groups.items():
        if traffic == "client":
            input_bytes = _sum(group, "client_input_bytes")
            output_bytes = _sum(group, "client_output_bytes")
        else:
            input_bytes = _sum(group, "input_bytes")
            output_bytes = _sum(group, "output_bytes")
        durations = [row["duration_ms"] for row in group]
        successes = sum(1 for row in group if _is_success(row.get("status")))
        result.append({
            "name": label,
            "count": len(group),
            "share": len(group) / total if total else 0.0,
            "errors": len(group) - successes,
            "input_bytes": input_bytes,
            "output_bytes": output_bytes,
            "p95_ms": _percentile(durations, 0.95),
        })
    return sorted(result, key=lambda row: (-row["count"], row["name"]))


def _time_series(rows, cutoff, now_timestamp, bucket_seconds):
    bucket_seconds = max(int(bucket_seconds), 1)
    bucket_count = max(1, int((now_timestamp - cutoff) / bucket_seconds) + 1)
    bucket_count = min(bucket_count, 1000)
    series = [{
        "timestamp": cutoff + index * bucket_seconds,
        "requests": 0,
        "errors": 0,
        "client_bytes": 0,
        "upstream_bytes": 0,
        "tokens": 0,
    } for index in range(bucket_count)]
    for row in rows:
        index = int((row["started_at"] - cutoff) / bucket_seconds)
        index = max(0, min(len(series) - 1, index))
        bucket = series[index]
        bucket["requests"] += 1
        if not _is_success(row.get("status")):
            bucket["errors"] += 1
        bucket["client_bytes"] += int(row.get("client_input_bytes") or 0) + int(row.get("client_output_bytes") or 0)
        bucket["upstream_bytes"] += int(row.get("upstream_input_bytes") or 0) + int(row.get("upstream_output_bytes") or 0)
        bucket["tokens"] += int(row.get("total_tokens") or 0)
    return series


def _recent(rows, *, errors_only=False, limit=10, reverse=False):
    selected = [row for row in rows if not errors_only or not _is_success(row.get("status"))]
    if errors_only:
        selected = selected[-limit:][::-1]
    elif reverse:
        selected = selected[-limit:][::-1]
    else:
        selected = selected[-limit:]
    fields = ("request_id", "started_at", "kind", "resolved_model", "runtime", "route",
              "status", "outcome", "error", "duration_ms", "attempt_count")
    return [{field: row.get(field) for field in fields} for row in selected]
