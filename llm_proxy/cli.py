"""Command-line client for the llm-proxy control plane."""

import argparse
import asyncio
import json
import os
import sys
from urllib.parse import quote

import aiohttp


def add_control_args(parser):
    parser.add_argument("--control-url", help="Control API base URL, e.g. http://127.0.0.1:8091")
    parser.add_argument("--host", help="Control host override")
    parser.add_argument("--port", type=int, help="Control port override")
    parser.add_argument("--token", help="Control Bearer token; defaults to LLM_PROXY_TOKEN")


def configure_subparsers(subparsers):
    status = subparsers.add_parser("status", help="Show runtime status")
    add_control_args(status)

    start = subparsers.add_parser("start", help="Start a runtime")
    start.add_argument("runtime", help="Runtime name")
    start.add_argument("--profile", help="Select profile before starting")
    add_control_args(start)

    stop = subparsers.add_parser("stop", help="Stop a runtime")
    stop.add_argument("runtime", help="Runtime name")
    add_control_args(stop)

    restart = subparsers.add_parser("restart", help="Restart a runtime")
    restart.add_argument("runtime", help="Runtime name")
    restart.add_argument("--profile", help="Select profile before restarting")
    add_control_args(restart)

    logs = subparsers.add_parser("logs", help="Show buffered runtime logs")
    logs.add_argument("runtime", help="Runtime name")
    logs.add_argument("-n", "--lines", type=int, default=80, help="Number of lines to show")
    add_control_args(logs)

    routes = subparsers.add_parser("routes", help="Show routing status")
    add_control_args(routes)

    cleanup = subparsers.add_parser("cleanup-kv", help="Run KV cache cleanup now")
    add_control_args(cleanup)


def is_control_command(command):
    return command in {"status", "start", "stop", "restart", "logs", "routes", "cleanup-kv"}


def control_url(args, config):
    if getattr(args, "control_url", None):
        return args.control_url.rstrip("/")

    listen = config.get("listen") or {}
    api_host = listen.get("host", "127.0.0.1") if isinstance(listen, dict) else "127.0.0.1"
    api_port = listen.get("port", 8090) if isinstance(listen, dict) else int(listen)
    control_cfg = config.get("control") or {}
    host = getattr(args, "host", None) or control_cfg.get("host") or api_host
    port = getattr(args, "port", None) or control_cfg.get("port") or (int(api_port) + 1)
    if host in ("0.0.0.0", "::", ""):
        host = "127.0.0.1"
    return f"http://{host}:{port}"


async def run_control_command(args, config):
    client = ControlClient(control_url(args, config), token=getattr(args, "token", None))
    try:
        if args.command == "status":
            data = await client.get("/v1/control/state")
            print_status(data)
            return 0
        if args.command == "start":
            if args.profile:
                await client.post(f"/v1/control/runtimes/{quote(args.runtime, safe='')}/select-profile", {"profile": args.profile})
            data = await client.post(f"/v1/control/runtimes/{quote(args.runtime, safe='')}/start")
            print_runtime(data.get("runtime"))
            return 0
        if args.command == "stop":
            data = await client.post(f"/v1/control/runtimes/{quote(args.runtime, safe='')}/stop")
            print_runtime(data.get("runtime"))
            return 0
        if args.command == "restart":
            await client.post(f"/v1/control/runtimes/{quote(args.runtime, safe='')}/stop")
            if args.profile:
                await client.post(f"/v1/control/runtimes/{quote(args.runtime, safe='')}/select-profile", {"profile": args.profile})
            data = await client.post(f"/v1/control/runtimes/{quote(args.runtime, safe='')}/start")
            print_runtime(data.get("runtime"))
            return 0
        if args.command == "logs":
            data = await client.get(f"/v1/control/runtimes/{quote(args.runtime, safe='')}/logs")
            for line in (data.get("logs") or [])[-max(args.lines, 0):]:
                print(line)
            return 0
        if args.command == "routes":
            data = await client.get("/v1/control/routes")
            print_routes(data.get("routes") or [])
            return 0
        if args.command == "cleanup-kv":
            data = await client.post("/v1/control/kv-cache/cleanup")
            print(
                "kv cleanup: "
                f"scanned={data.get('scanned', 0)} "
                f"deleted={data.get('deleted', 0)} "
                f"bytes_deleted={data.get('bytes_deleted', 0)}"
            )
            return 0
    except ControlError as e:
        print(f"llm-proxy: {e}", file=sys.stderr)
        return 1
    return 2


class ControlClient:
    def __init__(self, base_url, token=None):
        self.base_url = base_url.rstrip("/")
        self.token = token or os.environ.get("LLM_PROXY_TOKEN")

    async def get(self, path):
        return await self._request("GET", path)

    async def post(self, path, body=None):
        return await self._request("POST", path, body=body or {})

    async def _request(self, method, path, body=None):
        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        timeout = aiohttp.ClientTimeout(total=300)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as sess:
                kwargs = {"headers": headers}
                if body is not None:
                    kwargs["json"] = body
                async with sess.request(method, self.base_url + path, **kwargs) as resp:
                    text = await resp.text()
                    try:
                        data = json.loads(text) if text else {}
                    except json.JSONDecodeError:
                        data = {"error": text}
                    if resp.status >= 400:
                        msg = data.get("error") if isinstance(data, dict) else text
                        raise ControlError(f"HTTP {resp.status}: {msg}")
                    return data
        except aiohttp.ClientConnectorError as e:
            raise ControlError(f"cannot connect to {self.base_url}: {e}") from e
        except asyncio.TimeoutError as e:
            raise ControlError(f"request timed out: {self.base_url}{path}") from e


class ControlError(Exception):
    pass


def print_status(data):
    runtimes = data.get("runtimes") or []
    if not runtimes:
        print("(no runtimes)")
        return
    rows = [
        [
            rt.get("name", ""),
            rt.get("status", ""),
            rt.get("kind", ""),
            rt.get("profile", "") or "",
            str(rt.get("port") or ""),
            str(rt.get("slots") or rt.get("settings", {}).get("slots") or ""),
            str(rt.get("pid") or ""),
        ]
        for rt in runtimes
    ]
    print_table(["name", "status", "kind", "profile", "port", "slots", "pid"], rows)


def print_runtime(runtime):
    if not runtime:
        return
    print(
        f"{runtime.get('name')}: "
        f"{runtime.get('status')} "
        f"profile={runtime.get('profile') or '-'} "
        f"port={runtime.get('port') or '-'} "
        f"pid={runtime.get('pid') or '-'}"
    )


def print_routes(routes):
    if not routes:
        print("(no routes)")
        return
    rows = []
    for route in routes:
        backends = route.get("backends") or []
        healthy = sum(1 for item in backends if item.get("healthy"))
        rows.append([
            route.get("name", ""),
            route.get("kind", ""),
            str(len(backends)),
            f"{healthy}/{len(backends)}",
        ])
    print_table(["name", "kind", "backends", "healthy"], rows)


def print_table(headers, rows):
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(str(cell)))
    print("  ".join(h.ljust(widths[idx]) for idx, h in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(str(cell).ljust(widths[idx]) for idx, cell in enumerate(row)))
