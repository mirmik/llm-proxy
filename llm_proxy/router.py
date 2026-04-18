"""Router — OpenAI-compatible proxy with model routing and session awareness."""

import json

import aiohttp
from aiohttp import web


def create_app(runtime_manager, slot_manager, auth_manager=None):
    """Create aiohttp app with all routes."""
    app = web.Application()
    app["rm"] = runtime_manager
    app["sm"] = slot_manager
    app["auth"] = auth_manager

    app.router.add_post("/stt", handle_stt_proxy)
    app.router.add_get("/stt/health", handle_stt_health)
    app.router.add_post("/v1/audio/speech", handle_tts_proxy)
    app.router.add_get("/tts/health", handle_tts_health)
    app.router.add_post("/login", handle_login)
    app.router.add_post("/logout", handle_logout)
    app.router.add_get("/session", handle_session)
    app.router.add_post("/v1/chat/completions", handle_chat)
    app.router.add_get("/v1/models", handle_models)
    app.router.add_post("/v1/models/{name}/load", handle_load)
    app.router.add_post("/v1/models/{name}/unload", handle_unload)
    app.router.add_get("/v1/control/runtimes", handle_runtimes)
    app.router.add_post("/v1/control/runtimes/{name}/start", handle_runtime_start)
    app.router.add_post("/v1/control/runtimes/{name}/stop", handle_runtime_stop)
    app.router.add_post("/v1/control/runtimes/{name}/select-profile", handle_runtime_select_profile)
    app.router.add_post("/v1/control/runtimes/{name}/settings", handle_runtime_settings)
    app.router.add_get("/v1/control/runtimes/{name}/logs", handle_runtime_logs)
    app.router.add_get("/v1/control/state", handle_control_state)
    app.router.add_post("/v1/control/models/{name}/load", handle_load)
    app.router.add_post("/v1/control/models/{name}/unload", handle_unload)
    app.router.add_post("/v1/auth/bootstrap", handle_auth_bootstrap)
    app.router.add_get("/v1/control/whoami", handle_control_whoami)
    app.router.add_get("/", handle_panel)
    app.router.add_get("/health", handle_health)

    return app


def _resolve_model(rm, model_name):
    """Resolve model name, allowing partial matches. Returns (name, error_response)."""
    models = rm.available_models()
    if model_name in models:
        return model_name, None
    matched = [m for m in models if model_name in m]
    if len(matched) == 1:
        return matched[0], None
    if not matched:
        return None, web.json_response(
            {"error": f"Unknown model: {model_name}. Available: {models}"},
            status=404,
        )
    return None, web.json_response(
        {"error": f"Ambiguous model: {model_name}. Matches: {matched}"},
        status=400,
    )


async def _ensure_slot(rm, sm, model_name, session_id, port):
    """Assign a slot for the session if session_id is provided. Returns slot_id or None."""
    if not session_id:
        return None
    num_slots = rm.get_slots(model_name)
    if model_name not in sm._slots:
        sm.init_model(model_name, num_slots)
    return await sm.assign(model_name, session_id, port)


async def handle_chat(request):
    """Proxy /v1/chat/completions to the right llama-server backend."""
    rm = request.app["rm"]
    sm = request.app["sm"]

    body = await request.json()
    model_name = body.get("model", "")
    session_id = body.pop("session_id", None)
    stream = body.get("stream", False)

    model_name, err = _resolve_model(rm, model_name)
    if err:
        return err

    if not rm.is_running(model_name):
        return web.json_response(
            {"error": f"Model '{model_name}' is not loaded. Load it explicitly via control API or web UI."},
            status=503,
        )

    port = rm.get_port(model_name)
    if not port:
        return web.json_response(
            {"error": f"Model '{model_name}' is marked running but has no bound port."},
            status=503,
        )

    slot_id = await _ensure_slot(rm, sm, model_name, session_id, port)
    if slot_id is not None:
        body["id_slot"] = slot_id

    backend_url = f"http://127.0.0.1:{port}/v1/chat/completions"

    if stream:
        return await _proxy_stream(request, backend_url, body)
    else:
        return await _proxy_json(backend_url, body)


async def _proxy_json(url, body):
    """Forward request and return JSON response."""
    try:
        async with aiohttp.ClientSession() as sess:
            async with sess.post(url, json=body, timeout=aiohttp.ClientTimeout(total=300)) as resp:
                data = await resp.json()
                return web.json_response(data, status=resp.status)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=502)


async def _proxy_stream(request, url, body):
    """Forward streaming request, relay SSE chunks."""
    response = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
        },
    )
    await response.prepare(request)

    try:
        timeout = aiohttp.ClientTimeout(total=300)
        async with aiohttp.ClientSession() as sess:
            async with sess.post(url, json=body, timeout=timeout) as backend:
                async for chunk in backend.content.iter_any():
                    await response.write(chunk)
    except Exception as e:
        error_chunk = f"data: {json.dumps({'error': str(e)})}\n\n"
        await response.write(error_chunk.encode())

    await response.write_eof()
    return response


async def handle_models(request):
    """GET /v1/models — list available and loaded models."""
    rm = request.app["rm"]
    models = []
    for name in rm.available_models():
        models.append({
            "id": name,
            "object": "model",
            "created": 0,
            "owned_by": "local",
            "status": "loaded" if rm.is_running(name) else "available",
            "port": rm.get_port(name),
        })
    return web.json_response({"object": "list", "data": models})


async def handle_load(request):
    """POST /v1/models/{name}/load — explicitly load a model."""
    _require_control_auth(request)
    rm = request.app["rm"]
    name = request.match_info["name"]
    if name not in rm.available_models():
        return web.json_response({"error": f"Unknown model: {name}"}, status=404)
    try:
        await rm.start(name)
        return web.json_response({"status": "loaded", "model": name})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


async def handle_unload(request):
    """POST /v1/models/{name}/unload — stop a model."""
    _require_control_auth(request)
    rm = request.app["rm"]
    sm = request.app["sm"]
    name = request.match_info["name"]

    port = rm.get_port(name)
    if port:
        await sm.save_all(name, port)
        sm.remove_model(name)
    await rm.stop(name)
    return web.json_response({"status": "unloaded", "model": name})


async def handle_health(request):
    """GET /health — proxy health."""
    rm = request.app["rm"]
    return web.json_response({
        "status": "ok",
        "loaded_models": rm.loaded_models(),
        "loaded_runtimes": rm.loaded_runtimes(),
    })


def _resolve_stt_runtime(rm):
    preferred = "stt-main"
    stt_runtimes = rm.runtimes_by_kind("stt")
    if preferred in stt_runtimes:
        return preferred, None
    if len(stt_runtimes) == 1:
        return stt_runtimes[0], None
    if not stt_runtimes:
        return None, web.json_response({"error": "No STT runtime is configured"}, status=404)
    return None, web.json_response(
        {"error": f"Multiple STT runtimes are configured. Add a preferred routing rule. Available: {stt_runtimes}"},
        status=400,
    )


async def handle_stt_proxy(request):
    """Proxy STT client traffic to the active STT runtime."""
    _require_client_auth(request)
    rm = request.app["rm"]
    runtime_name, err = _resolve_stt_runtime(rm)
    if err:
        return err
    if not rm.is_running(runtime_name):
        return web.json_response(
            {"error": f"STT runtime '{runtime_name}' is not loaded. Start it explicitly via control API or web UI."},
            status=503,
        )
    port = rm.get_port(runtime_name)
    if not port:
        return web.json_response(
            {"error": f"STT runtime '{runtime_name}' is marked running but has no bound port."},
            status=503,
        )

    backend_url = f"http://127.0.0.1:{port}/stt"
    body = await request.read()
    headers = {}
    if request.content_type:
        headers["Content-Type"] = request.headers.get("Content-Type", request.content_type)
    auth_header = request.headers.get("Authorization")
    if auth_header:
        headers["Authorization"] = auth_header
    prompt = request.headers.get("X-Initial-Prompt")
    if prompt:
        headers["X-Initial-Prompt"] = prompt

    query_string = request.query_string
    if query_string:
        backend_url += "?" + query_string

    try:
        async with aiohttp.ClientSession() as sess:
            async with sess.post(
                backend_url,
                data=body,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=300),
            ) as resp:
                raw = await resp.read()
                content_type = resp.headers.get("Content-Type", "application/json")
                return web.Response(body=raw, status=resp.status, content_type=content_type.split(";")[0])
    except Exception as e:
        return web.json_response({"error": str(e)}, status=502)


def _resolve_tts_runtime(rm, requested=None):
    preferred = requested or "tts-main"
    tts_runtimes = rm.runtimes_by_kind("tts")
    if preferred in tts_runtimes:
        return preferred, None
    if len(tts_runtimes) == 1:
        return tts_runtimes[0], None
    if not tts_runtimes:
        return None, web.json_response({"error": "No TTS runtime is configured"}, status=404)
    return None, web.json_response(
        {"error": f"Multiple TTS runtimes are configured. Available: {tts_runtimes}"},
        status=400,
    )


async def handle_tts_proxy(request):
    """POST /v1/audio/speech — JSON in, binary audio out.

    Request body: {text, voice?, language?, speed?, temperature?}
    `voice` optionally selects the TTS runtime by name.
    """
    _require_client_auth(request)
    rm = request.app["rm"]
    body = await request.json()
    requested = body.pop("voice", None) if isinstance(body, dict) else None

    runtime_name, err = _resolve_tts_runtime(rm, requested)
    if err:
        return err
    if not rm.is_running(runtime_name):
        return web.json_response(
            {"error": f"TTS runtime '{runtime_name}' is not loaded. Start it explicitly via control API or web UI."},
            status=503,
        )
    port = rm.get_port(runtime_name)
    if not port:
        return web.json_response(
            {"error": f"TTS runtime '{runtime_name}' is marked running but has no bound port."},
            status=503,
        )

    backend_url = f"http://127.0.0.1:{port}/tts/generate"
    response = web.StreamResponse()
    try:
        async with aiohttp.ClientSession() as sess:
            async with sess.post(backend_url, json=body, timeout=aiohttp.ClientTimeout(total=300)) as backend:
                response.set_status(backend.status)
                for hk, hv in backend.headers.items():
                    if hk.lower() in ("content-length", "transfer-encoding", "connection"):
                        continue
                    response.headers[hk] = hv
                await response.prepare(request)
                async for chunk in backend.content.iter_any():
                    await response.write(chunk)
    except Exception as e:
        if not response.prepared:
            return web.json_response({"error": str(e)}, status=502)

    await response.write_eof()
    return response


async def handle_tts_health(request):
    _require_client_auth(request)
    rm = request.app["rm"]
    runtime_name, err = _resolve_tts_runtime(rm)
    if err:
        return err
    if not rm.is_running(runtime_name):
        return web.json_response(
            {"status": "stopped", "runtime": runtime_name, "loaded": False},
            status=503,
        )
    port = rm.get_port(runtime_name)
    backend_url = f"http://127.0.0.1:{port}/health"
    try:
        async with aiohttp.ClientSession() as sess:
            async with sess.get(backend_url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                data = await resp.json()
    except Exception as e:
        return web.json_response({"error": str(e), "runtime": runtime_name}, status=502)
    data["runtime"] = runtime_name
    data["loaded"] = True
    return web.json_response(data)


async def handle_stt_health(request):
    """Expose STT runtime health through the proxy."""
    _require_client_auth(request)
    rm = request.app["rm"]
    runtime_name, err = _resolve_stt_runtime(rm)
    if err:
        return err
    if not rm.is_running(runtime_name):
        return web.json_response(
            {"status": "stopped", "runtime": runtime_name, "loaded": False},
            status=503,
        )
    port = rm.get_port(runtime_name)
    backend_url = f"http://127.0.0.1:{port}/health"
    try:
        async with aiohttp.ClientSession() as sess:
            async with sess.get(backend_url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                data = await resp.json()
    except Exception as e:
        return web.json_response({"error": str(e), "runtime": runtime_name}, status=502)

    # Keep compatibility with the old client, which prints `model`.
    if "model" not in data and "model_size" in data:
        data["model"] = data["model_size"]
    data["runtime"] = runtime_name
    data["loaded"] = True
    return web.json_response(data)


def _require_control_auth(request):
    auth = request.app.get("auth")
    if auth is not None:
        return auth.require_control(request)
    return None


def _require_client_auth(request):
    auth = request.app.get("auth")
    if auth is not None:
        return auth.require_client(request)
    return None


async def handle_login(request):
    auth = request.app.get("auth")
    if auth is None or not auth.web_enabled:
        return web.json_response({"error": "web auth is disabled"}, status=400)
    body = await request.json()
    try:
        session_id = auth.web_login(body.get("username", ""), body.get("password", ""))
    except PermissionError as e:
        return web.json_response({"error": str(e)}, status=401)

    response = web.json_response({"ok": True, "username": body.get("username", "").strip()})
    response.set_cookie(
        auth.session_cookie_name(),
        session_id,
        httponly=True,
        samesite="Lax",
        secure=auth.secure_cookies(),
        max_age=int(float(auth.web_auth_cfg.get("session_ttl_hours", 168)) * 3600),
        path="/",
    )
    return response


async def handle_logout(request):
    auth = request.app.get("auth")
    if auth is not None:
        auth.web_logout(request)
    response = web.json_response({"ok": True})
    if auth is not None:
        response.del_cookie(auth.session_cookie_name(), path="/")
    return response


async def handle_session(request):
    auth = request.app.get("auth")
    username = auth.get_web_user(request) if auth is not None else None
    return web.json_response({"authenticated": bool(username), "username": username})


async def handle_auth_bootstrap(request):
    """POST /v1/auth/bootstrap — exchange invite/password for long-lived host token."""
    auth = request.app.get("auth")
    if auth is None or not auth.enabled:
        return web.json_response({"error": "auth is disabled"}, status=400)

    body = await request.json()
    try:
        result = auth.bootstrap(
            token=body.get("token", ""),
            password=body.get("password", ""),
            host_id=body.get("host_id", ""),
        )
    except ValueError as e:
        return web.json_response({"error": str(e)}, status=400)
    except PermissionError as e:
        return web.json_response({"error": str(e)}, status=403)

    return web.json_response({
        "ok": True,
        "host_id": result["host_id"],
        "token": result["token"],
    })


async def handle_control_whoami(request):
    """GET /v1/control/whoami — validate control credentials."""
    host_id = _require_control_auth(request)
    return web.json_response({"ok": True, "host_id": host_id})


async def handle_control_state(request):
    """GET /v1/control/state — current state for dashboard."""
    _require_control_auth(request)
    rm = request.app["rm"]
    runtimes = rm.list_state()
    models = [rt for rt in runtimes if rt["kind"] == "llm"]
    return web.json_response({"models": models, "runtimes": runtimes})


async def handle_runtimes(request):
    """GET /v1/control/runtimes — list runtimes."""
    _require_control_auth(request)
    rm = request.app["rm"]
    return web.json_response({"runtimes": rm.list_state()})


async def handle_runtime_start(request):
    """POST /v1/control/runtimes/{name}/start."""
    _require_control_auth(request)
    rm = request.app["rm"]
    name = request.match_info["name"]
    try:
        await rm.start(name)
        return web.json_response({"ok": True, "runtime": rm.runtime_state(name)})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


async def handle_runtime_stop(request):
    """POST /v1/control/runtimes/{name}/stop."""
    _require_control_auth(request)
    rm = request.app["rm"]
    name = request.match_info["name"]
    try:
        await rm.stop(name)
        return web.json_response({"ok": True, "runtime": rm.runtime_state(name)})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


async def handle_runtime_select_profile(request):
    """POST /v1/control/runtimes/{name}/select-profile."""
    _require_control_auth(request)
    rm = request.app["rm"]
    name = request.match_info["name"]
    body = await request.json()
    profile = body.get("profile")
    if not profile:
        return web.json_response({"error": "profile is required"}, status=400)
    try:
        await rm.select_profile(name, profile)
        return web.json_response({"ok": True, "runtime": rm.runtime_state(name)})
    except ValueError as e:
        return web.json_response({"error": str(e)}, status=400)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


async def handle_runtime_logs(request):
    """GET /v1/control/runtimes/{name}/logs."""
    _require_control_auth(request)
    rm = request.app["rm"]
    name = request.match_info["name"]
    if name not in rm.runtimes:
        return web.json_response({"error": f"Unknown runtime: {name}"}, status=404)
    return web.json_response({"runtime": name, "logs": rm.get_logs(name)})


async def handle_runtime_settings(request):
    """POST /v1/control/runtimes/{name}/settings."""
    _require_control_auth(request)
    rm = request.app["rm"]
    name = request.match_info["name"]
    body = await request.json()
    settings = body.get("settings")
    if not isinstance(settings, dict):
        return web.json_response({"error": "settings object is required"}, status=400)
    try:
        await rm.update_settings(name, settings)
        return web.json_response({"ok": True, "runtime": rm.runtime_state(name)})
    except ValueError as e:
        return web.json_response({"error": str(e)}, status=400)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


async def handle_panel(request):
    """Simple built-in control panel."""
    return web.Response(text=_panel_html(), content_type="text/html")


def _panel_html():
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>llm-proxy control</title>
  <style>
    :root {
      --bg: #f5efe2;
      --paper: #fffaf0;
      --ink: #1e1b18;
      --muted: #6d655d;
      --accent: #0f766e;
      --accent-2: #b45309;
      --line: #d8ccb8;
      --ok: #166534;
      --warn: #92400e;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Iosevka Aile", "IBM Plex Sans", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(180,83,9,0.15), transparent 24rem),
        linear-gradient(135deg, #efe2c6, var(--bg));
    }
    header {
      padding: 24px 28px 12px;
    }
    h1 {
      margin: 0;
      font-size: 2rem;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }
    .sub {
      color: var(--muted);
      margin-top: 6px;
    }
    .wrap {
      display: grid;
      grid-template-columns: 320px 1fr;
      gap: 20px;
      padding: 12px 28px 28px;
    }
    .card {
      background: rgba(255,250,240,0.9);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 18px;
      box-shadow: 0 12px 40px rgba(30,27,24,0.08);
      backdrop-filter: blur(8px);
    }
    .auth-grid {
      display: grid;
      gap: 10px;
    }
    .stack {
      display: grid;
      gap: 14px;
    }
    label {
      font-size: 0.85rem;
      text-transform: uppercase;
      color: var(--muted);
      letter-spacing: 0.08em;
    }
    input, select, button, textarea {
      width: 100%;
      padding: 10px 12px;
      border-radius: 12px;
      border: 1px solid var(--line);
      font: inherit;
      background: #fffdfa;
      color: var(--ink);
    }
    button {
      cursor: pointer;
      background: var(--ink);
      color: white;
      border: none;
      transition: transform 120ms ease, opacity 120ms ease;
    }
    button:hover { transform: translateY(-1px); }
    button.secondary { background: var(--accent); }
    button.warning { background: var(--accent-2); }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(270px, 1fr));
      gap: 16px;
      align-content: start;
    }
    .status {
      display: inline-block;
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 0.8rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      background: #ebe4d6;
      color: var(--warn);
    }
    .status.running {
      background: rgba(22,101,52,0.12);
      color: var(--ok);
    }
    .actions {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 8px;
      margin-top: 12px;
    }
    .settings-grid {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 8px;
      margin-top: 12px;
    }
    .settings-grid .full {
      grid-column: 1 / -1;
    }
    .meta {
      color: var(--muted);
      font-size: 0.9rem;
      margin-top: 8px;
      display: grid;
      gap: 4px;
    }
    pre {
      margin: 10px 0 0;
      background: #1c1917;
      color: #f8f5ef;
      padding: 12px;
      border-radius: 14px;
      max-height: 220px;
      overflow: auto;
      font-family: "Iosevka", monospace;
      font-size: 0.85rem;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .toolbar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 14px;
      gap: 10px;
    }
    .message {
      color: var(--muted);
      min-height: 1.4em;
      font-size: 0.92rem;
    }
    @media (max-width: 920px) {
      .wrap { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <h1>llm-proxy control</h1>
    <div class="sub">Unified local control plane for LLM, STT, and TTS runtimes.</div>
  </header>
  <div class="wrap">
    <section class="card stack">
      <div>
        <label>Username</label>
        <input id="username" placeholder="mirmik">
      </div>
      <div>
        <label>Password</label>
        <input id="password" type="password" placeholder="password">
      </div>
      <div class="auth-grid">
        <button id="loginBtn">Login</button>
        <button class="secondary" id="logoutBtn">Logout</button>
      </div>
      <div class="message" id="authMsg"></div>
      <div class="meta">
        <div>Browser UI uses a cookie-backed session.</div>
        <div>Machine clients continue to use host tokens.</div>
      </div>
    </section>
    <section class="stack">
      <div class="card">
        <div class="toolbar">
          <strong>Managed Runtimes</strong>
          <button class="secondary" id="refreshBtn" style="width:auto;padding:10px 18px;">Refresh</button>
        </div>
        <div class="message" id="stateMsg"></div>
      </div>
      <div class="grid" id="runtimes"></div>
    </section>
  </div>
  <script>
    const usernameInput = document.getElementById('username');
    const passwordInput = document.getElementById('password');
    const authMsg = document.getElementById('authMsg');
    const stateMsg = document.getElementById('stateMsg');
    const runtimesEl = document.getElementById('runtimes');

    async function api(path, options = {}) {
      const headers = Object.assign({}, options.headers || {});
      if (options.body && !headers['Content-Type']) {
        headers['Content-Type'] = 'application/json';
      }
      const resp = await fetch(path, Object.assign({}, options, {headers, credentials: 'same-origin'}));
      const data = await resp.json().catch(() => ({}));
      if (!resp.ok) {
        throw new Error(data.error || resp.statusText);
      }
      return data;
    }

    async function checkAuth() {
      try {
        const data = await api('/session');
        authMsg.textContent = data.authenticated ? ('Logged in as ' + data.username) : 'Not logged in.';
      } catch (e) {
        authMsg.textContent = 'Auth failed: ' + e.message;
      }
    }

    async function login() {
      try {
        const data = await api('/login', {
          method: 'POST',
          body: JSON.stringify({username: usernameInput.value.trim(), password: passwordInput.value}),
        });
        passwordInput.value = '';
        authMsg.textContent = 'Logged in as ' + data.username;
        await refreshState();
      } catch (e) {
        authMsg.textContent = 'Login failed: ' + e.message;
      }
    }

    async function logout() {
      try {
        await api('/logout', {method: 'POST'});
        authMsg.textContent = 'Logged out.';
        await refreshState();
      } catch (e) {
        authMsg.textContent = 'Logout failed: ' + e.message;
      }
    }

    function runtimeCard(runtime) {
      const running = runtime.status === 'running';
      const options = (runtime.profiles || []).map((profile) =>
        `<option value="${escapeHtml(profile)}" ${profile === runtime.profile ? 'selected' : ''}>${escapeHtml(profile)}</option>`
      ).join('');
      const llmAction = runtime.kind === 'llm'
        ? `<button data-action="${running ? 'unload-model' : 'load-model'}" data-name="${escapeHtml(runtime.name)}">${running ? 'Unload' : 'Load'}</button>`
        : `<button disabled>${escapeHtml(runtime.kind)}</button>`;
      const settings = runtime.settings || {};
      return `<article class="card">
        <div class="toolbar">
          <strong>${escapeHtml(runtime.name)}</strong>
          <span class="status ${running ? 'running' : ''}">${escapeHtml(runtime.status)}</span>
        </div>
        <div class="meta">
          <div>driver: ${escapeHtml(runtime.driver || '—')}</div>
          <div>kind: ${escapeHtml(runtime.kind || 'runtime')}</div>
          <div>profile: ${escapeHtml(runtime.profile || '—')}</div>
          <div>port: ${escapeHtml(runtime.port || '—')}</div>
          <div>gpu: ${escapeHtml(runtime.gpu || '—')}</div>
          <div>health: ${escapeHtml(runtime.health_url || '—')}</div>
        </div>
        ${runtime.profiles && runtime.profiles.length ? `
          <div style="margin-top:12px;">
            <label>Profile</label>
            <select data-profile-select="${escapeHtml(runtime.name)}">${options}</select>
          </div>` : ''}
        <div class="actions">
          <button class="secondary" data-action="start-runtime" data-name="${escapeHtml(runtime.name)}">Start</button>
          <button class="warning" data-action="stop-runtime" data-name="${escapeHtml(runtime.name)}">Stop</button>
          ${llmAction}
        </div>
        <div class="settings-grid">
          <div>
            <label>Autoload</label>
            <select data-setting="autoload" data-name="${escapeHtml(runtime.name)}">
              <option value="true" ${settings.autoload ? 'selected' : ''}>true</option>
              <option value="false" ${!settings.autoload ? 'selected' : ''}>false</option>
            </select>
          </div>
          <div>
            <label>Port</label>
            <input data-setting="port" data-name="${escapeHtml(runtime.name)}" value="${escapeHtml(settings.port ?? '')}">
          </div>
          ${settings.kind === 'llm' ? `
          <div>
            <label>Slots</label>
            <input data-setting="slots" data-name="${escapeHtml(runtime.name)}" value="${escapeHtml(settings.slots ?? '')}">
          </div>
          <div>
            <label>Startup Timeout (s)</label>
            <input data-setting="startup_timeout" data-name="${escapeHtml(runtime.name)}" value="${escapeHtml(settings.startup_timeout ?? '')}" placeholder="default 120">
          </div>
          ` : `
          <div>
            <label>Venv</label>
            <input data-setting="venv" data-name="${escapeHtml(runtime.name)}" value="${escapeHtml(settings.venv ?? '')}" placeholder="~/path/to/venv (resolves {python})">
          </div>
          <div>
            <label>Startup Timeout (s)</label>
            <input data-setting="startup_timeout" data-name="${escapeHtml(runtime.name)}" value="${escapeHtml(settings.startup_timeout ?? '')}" placeholder="default 120/240">
          </div>
          `}
          <div class="full">
            <label>Command</label>
            <textarea data-setting="command" data-name="${escapeHtml(runtime.name)}" rows="6" spellcheck="false" style="font-family:monospace;font-size:0.85rem;">${escapeHtml(settings.command ?? '')}</textarea>
          </div>
          <div class="full">
            <label>Env (KEY=value per line)</label>
            <textarea data-setting="env" data-name="${escapeHtml(runtime.name)}" rows="3" spellcheck="false" style="font-family:monospace;font-size:0.85rem;">${escapeHtml(formatEnv(settings.env))}</textarea>
          </div>
          <div class="full">
            <button class="secondary" data-action="save-settings" data-name="${escapeHtml(runtime.name)}">Save Settings</button>
          </div>
        </div>
        <button style="margin-top:10px;background:#44403c;" data-action="logs-runtime" data-name="${escapeHtml(runtime.name)}">Show Logs</button>
        <pre id="logs-${escapeHtml(runtime.name)}" hidden></pre>
      </article>`;
    }

    async function refreshState() {
      try {
        const data = await api('/v1/control/state');
        runtimesEl.innerHTML = data.runtimes.map(runtimeCard).join('');
        stateMsg.textContent = `runtimes: ${data.runtimes.length}, llm: ${data.models.length}`;
      } catch (e) {
        runtimesEl.innerHTML = '';
        stateMsg.textContent = 'Failed to load state: ' + e.message;
      }
    }

    async function showLogs(name) {
      const data = await api('/v1/control/runtimes/' + encodeURIComponent(name) + '/logs');
      const pre = document.getElementById('logs-' + name);
      pre.hidden = false;
      pre.textContent = (data.logs || []).join('\\n');
    }

    async function saveSettings(name) {
      const fields = document.querySelectorAll(`[data-setting][data-name="${CSS.escape(name)}"]`);
      const settings = {};
      for (const field of fields) {
        let value = field.value;
        const key = field.dataset.setting;
        if (key === 'autoload') {
          value = value === 'true';
        }
        settings[key] = value;
      }
      await api('/v1/control/runtimes/' + encodeURIComponent(name) + '/settings', {
        method: 'POST',
        body: JSON.stringify({settings}),
      });
    }

    function formatEnv(env) {
      if (!env || typeof env !== 'object') return '';
      return Object.entries(env).map(([k, v]) => `${k}=${v}`).join('\\n');
    }

    function escapeHtml(value) {
      return String(value)
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;');
    }

    document.getElementById('loginBtn').addEventListener('click', login);
    document.getElementById('logoutBtn').addEventListener('click', logout);
    document.getElementById('refreshBtn').addEventListener('click', refreshState);

    checkAuth();
    document.addEventListener('click', async (event) => {
      const target = event.target.closest('[data-action]');
      if (!target) return;
      const action = target.dataset.action;
      const name = target.dataset.name;
      try {
        if (action === 'load-model') {
          await api('/v1/control/models/' + encodeURIComponent(name) + '/load', {method: 'POST'});
        } else if (action === 'unload-model') {
          await api('/v1/control/models/' + encodeURIComponent(name) + '/unload', {method: 'POST'});
        } else if (action === 'start-runtime') {
          await api('/v1/control/runtimes/' + encodeURIComponent(name) + '/start', {method: 'POST'});
        } else if (action === 'stop-runtime') {
          await api('/v1/control/runtimes/' + encodeURIComponent(name) + '/stop', {method: 'POST'});
        } else if (action === 'logs-runtime') {
          await showLogs(name);
          return;
        } else if (action === 'save-settings') {
          await saveSettings(name);
        }
        await refreshState();
      } catch (e) {
        stateMsg.textContent = 'Action failed: ' + e.message;
      }
    });

    document.addEventListener('change', async (event) => {
      const select = event.target.closest('[data-profile-select]');
      if (!select) return;
      const name = select.dataset.profileSelect;
      try {
        await api('/v1/control/runtimes/' + encodeURIComponent(name) + '/select-profile', {
          method: 'POST',
          body: JSON.stringify({profile: select.value}),
        });
        await refreshState();
      } catch (e) {
        stateMsg.textContent = 'Profile switch failed: ' + e.message;
      }
    });

    refreshState();
  </script>
</body>
</html>"""
