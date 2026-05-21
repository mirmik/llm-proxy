"""Route manager — session-aware routing with health-aware failover.

Adapted from nemor-link's ServicePool (pool.py), made async with aiohttp.
"""

import asyncio
import time

import aiohttp

from llm_proxy.config import load_config_file, save_config_file

KIND_ENDPOINTS = {
    "llm": "/v1/chat/completions",
    "stt": "/stt",
    "tts": "/v1/audio/speech",
}

KIND_HEALTH_DEFAULTS = {
    "llm": "/health",
    "stt": "/health",
    "tts": "/health",
}


def _derive_health_url(base_url, health_path):
    if base_url.endswith("/"):
        base_url = base_url[:-1]
    if health_path.startswith("/"):
        return base_url + health_path
    return base_url + "/" + health_path


def _derive_endpoint_url(base_url, kind):
    if base_url.endswith("/"):
        base_url = base_url[:-1]
    suffix = KIND_ENDPOINTS.get(kind, "/v1/chat/completions")
    if suffix.startswith("/"):
        return base_url + suffix
    return base_url + "/" + suffix


class Route:
    """One named route with a priority-ordered list of backends and health state."""

    def __init__(self, name, backends, kind="llm", health_path="/health",
                 health_timeout=2.0, check_interval=5.0, hosts=None,
                 session_ttl=3600.0):
        self.name = name
        self.kind = kind
        self.backends = list(backends)
        self.health_path = health_path
        self.health_timeout = health_timeout
        self.check_interval = check_interval
        self.session_ttl = float(session_ttl)
        self.hosts = hosts or {}
        self._by_url = {}
        self._healthy = {}
        self._latency = {}
        self._last_checked = {}
        self._inflight = {}
        self._session_backend = {}
        self._session_seen = {}
        self._active_url = backends[0]["url"] if backends else None
        self._lock = asyncio.Lock()

        for b in self.backends:
            u = b["url"]
            self._by_url[u] = b
            self._healthy[u] = None
            self._latency[u] = None
            self._last_checked[u] = None
            self._inflight[u] = 0

    def _resolve_auth(self, backend):
        auth_name = backend.get("auth")
        if not auth_name or auth_name not in self.hosts:
            return {}
        return self.hosts[auth_name]

    def _auth_headers(self, backend):
        host = self._resolve_auth(backend)
        headers = {}
        token = host.get("token")
        if token:
            headers["Authorization"] = "Bearer " + token
        host_id = host.get("host_id")
        if host_id:
            headers["X-LLM-Proxy-Host-ID"] = host_id
        return headers

    def _best_available_locked(self, exclude=None):
        exclude = exclude or set()
        for b in self.backends:
            u = b["url"]
            if u in exclude:
                continue
            if self._healthy.get(u) is not False:
                return u
        for b in self.backends:
            if b["url"] not in exclude:
                return b["url"]
        return self.backends[0]["url"] if self.backends else None

    async def _mark(self, url, ok, latency=None):
        async with self._lock:
            self._healthy[url] = bool(ok)
            self._last_checked[url] = time.time()
            if latency is not None:
                self._latency[url] = latency
            old_active = self._active_url
            self._active_url = self._best_available_locked()
            changed = old_active != self._active_url
        return changed

    async def mark_unhealthy(self, url):
        """Eagerly mark a backend unhealthy (called on live request failure)."""
        async with self._lock:
            self._healthy[url] = False
            self._active_url = self._best_available_locked()
            self._drop_sessions_for_url_locked(url)

    async def probe(self, url):
        health_url = _derive_health_url(url, self.health_path)
        t0 = time.time()
        try:
            async with aiohttp.ClientSession() as sess:
                async with sess.get(
                    health_url,
                    timeout=aiohttp.ClientTimeout(total=self.health_timeout),
                ) as resp:
                    ok = resp.status < 500
                    latency = time.time() - t0
        except Exception:
            ok = False
            latency = None
        await self._mark(url, ok, latency)
        return ok, latency

    async def probe_all(self):
        results = []
        for b in self.backends:
            ok, lat = await self.probe(b["url"])
            results.append((b["url"], ok, lat))
        return results

    def failover_candidates(self, session_id=None):
        """Return backends ordered by sticky session, health, and current load."""
        self._prune_sessions()
        sticky_url = self._session_backend.get(session_id) if session_id else None
        if sticky_url and self._healthy.get(sticky_url) is not False and sticky_url in self._by_url:
            ordered_urls = [sticky_url]
            reason = "sticky"
        else:
            ordered_urls = []
            reason = "least-loaded" if session_id else "least-loaded-no-session"

        remaining = [
            b["url"] for b in self.backends
            if b["url"] not in ordered_urls
        ]
        remaining.sort(key=self._candidate_score)
        ordered_urls.extend(remaining)

        out = []
        for index, url in enumerate(ordered_urls):
            backend = self._enriched_backend(self._by_url[url])
            backend["_route_reason"] = reason if index == 0 else "failover"
            backend["_sticky_session"] = bool(session_id and self._session_backend.get(session_id) == url)
            out.append(backend)
        return out

    def note_request_start(self, session_id, url):
        self._inflight[url] = self._inflight.get(url, 0) + 1
        if session_id:
            self._session_backend[session_id] = url
            self._session_seen[session_id] = time.time()

    def note_request_end(self, session_id, url):
        self._inflight[url] = max(0, self._inflight.get(url, 0) - 1)
        if session_id and self._session_backend.get(session_id) == url:
            self._session_seen[session_id] = time.time()

    def _candidate_score(self, url):
        healthy_penalty = 0 if self._healthy.get(url) is not False else 1
        backend = self._by_url.get(url) or {}
        weight = max(float(backend.get("weight", 1) or 1), 0.001)
        load = self._inflight.get(url, 0) / weight
        latency = self._latency.get(url)
        latency_score = latency if latency is not None else 0.0
        priority = self._backend_index(url)
        return (healthy_penalty, load, latency_score, priority)

    def _backend_index(self, url):
        for index, backend in enumerate(self.backends):
            if backend["url"] == url:
                return index
        return len(self.backends)

    def _prune_sessions(self):
        if not self._session_seen or self.session_ttl <= 0:
            return
        cutoff = time.time() - self.session_ttl
        expired = [sid for sid, seen in self._session_seen.items() if seen < cutoff]
        for sid in expired:
            self._session_seen.pop(sid, None)
            self._session_backend.pop(sid, None)

    def _drop_sessions_for_url_locked(self, url):
        stale = [sid for sid, bound in self._session_backend.items() if bound == url]
        for sid in stale:
            self._session_backend.pop(sid, None)
            self._session_seen.pop(sid, None)

    def _enriched_backend(self, backend):
        d = dict(backend)
        d["_host"] = self._resolve_auth(backend)
        d["_healthy"] = self._healthy.get(backend["url"])
        d["_latency"] = self._latency.get(backend["url"])
        d["_inflight"] = self._inflight.get(backend["url"], 0)
        return d

    def active(self):
        if self._active_url and self._active_url in self._by_url:
            return self._enriched_backend(self._by_url[self._active_url])
        return None

    def active_url(self):
        return self._active_url

    def status(self):
        backends_status = []
        for b in self.backends:
            u = b["url"]
            backends_status.append({
                "url": u,
                "model": b.get("model", self.name),
                "auth": b.get("auth"),
                "healthy": self._healthy.get(u),
                "latency_ms": round(self._latency[u] * 1000, 1) if self._latency.get(u) else None,
                "last_checked": self._last_checked.get(u),
                "inflight": self._inflight.get(u, 0),
                "weight": b.get("weight", 1),
            })
        return {
            "name": self.name,
            "kind": self.kind,
            "active_url": self._active_url,
            "health_path": self.health_path,
            "check_interval": self.check_interval,
            "session_ttl": self.session_ttl,
            "sticky_sessions": len(self._session_backend),
            "backends": backends_status,
        }

    def add_backend(self, backend):
        self.backends.append(backend)
        u = backend["url"]
        self._by_url[u] = backend
        self._healthy[u] = None
        self._latency[u] = None
        self._last_checked[u] = None
        self._inflight[u] = 0
        if self._active_url is None:
            self._active_url = u

    def remove_backend(self, index):
        removed = self.backends.pop(index)
        u = removed["url"]
        self._by_url.pop(u, None)
        self._healthy.pop(u, None)
        self._latency.pop(u, None)
        self._last_checked.pop(u, None)
        self._inflight.pop(u, None)
        self._drop_sessions_for_url_locked(u)
        if self._active_url == u:
            self._active_url = self._best_available_locked()

    def to_config(self):
        return {
            "kind": self.kind,
            "health_path": self.health_path,
            "health_timeout": self.health_timeout,
            "check_interval": self.check_interval,
            "session_ttl": self.session_ttl,
            "backends": [
                {k: v for k, v in b.items() if not k.startswith("_")}
                for b in self.backends
            ],
        }


class RouteManager:
    """Manages all routes, config persistence, and background health monitoring."""

    def __init__(self, config):
        self._config = config
        self._routes = {}
        self._monitor_tasks = {}
        self._monitoring_started = False
        self._stop_event = asyncio.Event()
        self._load_from_config(config)

    def _load_from_config(self, config):
        hosts = config.get("hosts") or {}
        routes_cfg = config.get("routes") or {}
        self._routes.clear()
        for name, spec in routes_cfg.items():
            route = Route(
                name=name,
                backends=spec.get("backends", []),
                kind=spec.get("kind", "llm"),
                health_path=spec.get("health_path", KIND_HEALTH_DEFAULTS.get(spec.get("kind", "llm"), "/health")),
                health_timeout=spec.get("health_timeout", 2.0),
                check_interval=spec.get("check_interval", 5.0),
                session_ttl=spec.get("session_ttl", 3600.0),
                hosts=hosts,
            )
            self._routes[name] = route

    @property
    def routes(self):
        return dict(self._routes)

    def available_routes(self):
        return list(self._routes.keys())

    def resolve_model(self, model_name):
        routes = self.available_routes()
        if model_name in routes:
            return model_name, None
        matched = [m for m in routes if model_name in m]
        if len(matched) == 1:
            return matched[0], None
        if not matched:
            return None, None
        return None, f"Ambiguous model: {model_name}. Matches: {matched}"

    def get_route(self, name):
        return self._routes.get(name)

    def list_routes_status(self):
        return [r.status() for r in self._routes.values()]

    async def _monitor_loop(self, route):
        while not self._stop_event.is_set():
            for b in route.backends:
                if self._stop_event.is_set():
                    return
                await route.probe(b["url"])
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=route.check_interval,
                )
                return
            except asyncio.TimeoutError:
                pass

    async def start_monitoring(self):
        if self._monitoring_started:
            return
        self._stop_event.clear()
        self._monitoring_started = True
        self._monitor_tasks = {}
        for route in self._routes.values():
            if route.backends:
                self._monitor_tasks[route.name] = asyncio.create_task(
                    self._monitor_loop(route)
                )

    async def stop_monitoring(self):
        self._monitoring_started = False
        self._stop_event.set()
        tasks = list(self._monitor_tasks.values())
        for task in tasks:
            task.cancel()
        results = await asyncio.gather(*tasks, return_exceptions=True) if tasks else []
        self._monitor_tasks = {}
        return results

    async def _restart_monitoring_if_running(self):
        if not self._monitoring_started:
            return
        tasks = list(self._monitor_tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._monitor_tasks = {}
        for route in self._routes.values():
            if route.backends:
                self._monitor_tasks[route.name] = asyncio.create_task(
                    self._monitor_loop(route)
                )

    def _config_path(self):
        return self._config.get("_config_path")

    def _reload(self):
        path = self._config_path()
        if path:
            fresh = load_config_file(path)
            hosts = (fresh or {}).get("hosts") or (self._config.get("hosts") or {})
            routes_cfg = (fresh or {}).get("routes") or {}
            self._config["hosts"] = hosts
            self._config["routes"] = routes_cfg
            self._load_from_config(self._config)

    async def _save_and_reload(self, updater):
        path = self._config_path()
        if not path:
            self._apply_local(updater)
            return
        data = load_config_file(path)
        if data is None:
            data = {}
        updater(data)
        save_config_file(path, data)
        self._reload()

    def _apply_local(self, updater):
        data = {"hosts": self._config.get("hosts") or {},
                "routes": self._config.get("routes") or {}}
        updater(data)
        self._config["hosts"] = data.get("hosts") or {}
        self._config["routes"] = data.get("routes") or {}
        self._load_from_config(self._config)

    async def create_route(self, spec):
        if not isinstance(spec, dict):
            raise ValueError("route spec must be an object")
        name = spec.get("name")
        if not name:
            raise ValueError("route name is required")
        if name in self._routes:
            raise ValueError(f"Route '{name}' already exists")

        backends = spec.get("backends", [])
        if not isinstance(backends, list):
            raise ValueError("backends must be a list")
        for i, b in enumerate(backends):
            if not isinstance(b, dict):
                raise ValueError(f"backends[{i}] must be an object")
            if not b.get("url"):
                raise ValueError(f"backends[{i}].url is required")
        kind = spec.get("kind", "llm")
        if kind not in KIND_ENDPOINTS:
            raise ValueError(f"unknown route kind: {kind}")

        await self._save_and_reload(lambda data: _upsert_route(data, name, spec))
        await self._restart_monitoring_if_running()

    async def delete_route(self, name):
        if name not in self._routes:
            raise ValueError(f"Unknown route: {name}")

        await self._save_and_reload(lambda data: _delete_route(data, name))
        await self._restart_monitoring_if_running()

    async def add_backend(self, route_name, backend_spec):
        if not isinstance(backend_spec, dict):
            raise ValueError("backend must be an object")
        if not backend_spec.get("url"):
            raise ValueError("backend.url is required")
        route = self._routes.get(route_name)
        if not route:
            raise ValueError(f"Unknown route: {route_name}")

        await self._save_and_reload(
            lambda data: _add_backend(data, route_name, backend_spec)
        )
        await self._restart_monitoring_if_running()

    async def remove_backend(self, route_name, index):
        route = self._routes.get(route_name)
        if not route:
            raise ValueError(f"Unknown route: {route_name}")
        if index < 0 or index >= len(route.backends):
            raise ValueError(f"Invalid backend index: {index}")

        await self._save_and_reload(
            lambda data: _remove_backend(data, route_name, index)
        )
        await self._restart_monitoring_if_running()


def _upsert_route(data, name, spec):
    routes = data.setdefault("routes", {})
    entry = {
        "kind": spec.get("kind", "llm"),
        "backends": spec.get("backends", []),
    }
    if "health_path" in spec:
        entry["health_path"] = spec["health_path"]
    if "health_timeout" in spec:
        entry["health_timeout"] = spec["health_timeout"]
    if "check_interval" in spec:
        entry["check_interval"] = spec["check_interval"]
    if "session_ttl" in spec:
        entry["session_ttl"] = spec["session_ttl"]
    routes[name] = entry


def _delete_route(data, name):
    routes = data.get("routes") or {}
    routes.pop(name, None)
    if not routes:
        data.pop("routes", None)


def _add_backend(data, route_name, backend_spec):
    routes = data.get("routes") or {}
    route = routes.get(route_name)
    if not route:
        raise ValueError(f"Unknown route: {route_name}")
    route.setdefault("backends", []).append(backend_spec)


def _remove_backend(data, route_name, index):
    routes = data.get("routes") or {}
    route = routes.get(route_name)
    if not route:
        raise ValueError(f"Unknown route: {route_name}")
    backends = route.get("backends") or []
    if index < 0 or index >= len(backends):
        raise ValueError(f"Invalid backend index: {index}")
    backends.pop(index)
