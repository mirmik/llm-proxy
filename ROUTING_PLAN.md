# Routing Layer for llm-proxy

## Context

llm-proxy's routing layer lets a model name resolve to a list of remote or
local backends with health-aware failover. LLM routes are session-aware:
requests with the same extracted session key stay sticky to the same backend
while new sessions are placed on the least-loaded healthy backend.

## Approach

**Supplement, don't replace.** The RouteManager sits alongside RuntimeManager. If a model name matches a configured route → use the route's failover logic. If no route matches → fall through to the existing RuntimeManager flow. Backward compatible: with no routes configured, nothing changes.

### Routing Logic

- Backends are ordered by priority (index 0 = highest).
- Health state per URL: `None` (unknown, presumed usable), `True` (healthy), `False` (unhealthy).
- `failover_candidates(session_id=...)` returns a sticky backend first when the session is already assigned and healthy.
- New sessions are placed by effective load: `inflight / weight`, then latency and priority.
- Eager failure detection: a failed live request immediately marks the backend unhealthy.
- Sticky bindings have `session_ttl` and are removed when the backend is marked unhealthy.
- Background health monitor: one asyncio task per route, probes each backend at `check_interval`.

For qwen-code, the session key extractor understands DashScope metadata:
`metadata.promptId` shaped as `<session>#<subagent>#<turn>` becomes
`<session>#<subagent>`. That means parallel subagents can spread across
machines, while each subagent remains sticky after placement.

### Config (in `config.yaml`, new top-level keys)

```yaml
hosts:                    # shared auth configs (referenced by name)
  remote-cluster:
    token: "sk-xxx"
    host_id: "node-1"

routes:
  qwen3.5-122b:           # route name = model name
    kind: llm
    health_path: "/health"
    check_interval: 5.0
    session_ttl: 3600
    backends:
      - url: "http://192.168.1.100:8080"
        model: "qwen3.5-122b"       # overrides model in proxied request
        auth: "remote-cluster"
        weight: 2                   # receives roughly 2x load
      - url: "http://192.168.1.101:8080"   # no auth = unauthenticated
```

### New control API endpoints (on the control plane)

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/v1/control/routes` | List all routes with health status |
| POST | `/v1/control/routes` | Create a route |
| DELETE | `/v1/control/routes/{name}` | Delete a route |
| POST | `/v1/control/routes/{name}/backends` | Add a backend |
| DELETE | `/v1/control/routes/{name}/backends/{index}` | Remove a backend |

### UI (new "Routing" tab in `panel.html`)

- Navigation button alongside "Runtimes" / "Settings"
- Card per route showing: name, kind badge, backend table (priority #, URL, model override, health dot, latency, last checked)
- Forms for: create route, add backend, delete route/backend
- Health dots: green (healthy), red (unhealthy), grey (unknown)

### Data flow (chat example)

```
POST /v1/chat/completions {model: "qwen3.5-122b"}
  → _resolve_model() checks RuntimeManager → no match
  → checks RouteManager → match found
  → route.failover_candidates(session_id=...) → [{url: "http://192.168.1.100:8080", ...}, ...]
  → for each backend:
      POST to {url}/v1/chat/completions with auth headers
      success → return response
      failure → mark_unhealthy(url), try next
  → all failed → 503
```

## Files

### New: `llm_proxy/route_manager.py`
- `Route` class: one named route, manages backend list, health state, priority ordering
- `RouteManager` class: manages all routes, config persistence, background monitors

### Modified: `llm_proxy/router.py`
- `_resolve_model()` accepts optional `route_manager`, checks routes after RuntimeManager
- `handle_chat()` / `handle_stt_proxy()` / `handle_tts_proxy()`: add routing branch with failover loop
- `_proxy_json()` / `_proxy_stream()`: accept optional `extra_headers` parameter
- `handle_models()`: include route-defined models
- New control handlers: `handle_routes_list`, `handle_routes_create`, `handle_routes_delete`, `handle_routes_add_backend`, `handle_routes_remove_backend`

### Modified: `llm_proxy/main.py`
- Instantiate `RouteManager(config)`, pass to `create_api_app()` and `create_control_app()`
- Start/stop health monitoring in lifecycle

### Modified: `llm_proxy/config.py`
- Add `hosts: {}` and `routes: {}` to `DEFAULT_CONFIG`
- Add basic route validation

### Modified: `llm_proxy/panel.html`
- New "Routing" nav button, view section, JS functions (`refreshRoutes`, `renderRouteCard`, CRUD handlers)

## Implementation order

1. `route_manager.py` — core `Route` + `RouteManager` classes
2. `config.py` — schema defaults and validation
3. `main.py` — wiring and lifecycle
4. `router.py` — routing branch in handlers, new control endpoints, header passthrough
5. `panel.html` — routing UI

## Verification

1. Start llm-proxy with no routes configured → all existing flows work unchanged (chat, STT, TTS, panel)
2. Add a route via the control API pointing to a valid backend → requests for that model are proxied to the remote backend
3. Add a route pointing to a dead backend + a live one → failover works (dead skipped, live used)
4. Health monitoring updates backend status visible in `GET /v1/control/routes`
5. UI: create/delete routes, add/remove backends, see health status update
