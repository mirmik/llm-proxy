"""Router — OpenAI-compatible proxy with model routing and session awareness."""

import json
import os
from urllib.parse import urlsplit

import aiohttp
from aiohttp import web

from llm_proxy.certs import certificate_fingerprint, generate_server_cert
from llm_proxy.config import load_config_file, save_config_file
from llm_proxy.oplog import elapsed_ms, log_event, now

_SESSION_BODY_KEYS = (
    "session_id",
    "sessionId",
    "session",
    "conversation_id",
    "conversationId",
    "chat_id",
    "chatId",
)
_SESSION_METADATA_KEYS = (
    "session_id",
    "sessionId",
    "session",
    "conversation_id",
    "conversationId",
    "chat_id",
    "chatId",
)
_SESSION_HEADER_KEYS = (
    "X-LLM-Proxy-Session-ID",
    "X-Session-ID",
    "X-Session-Id",
    "X-Qwen-Session-ID",
    "X-Qwen-Session-Id",
    "X-Conversation-ID",
    "X-Conversation-Id",
    "X-Chat-ID",
    "X-Chat-Id",
)
_QWEN_MAIN_PROMPT_DELIMITER = "########"


def _setup_app(rm, sm, auth, routes=None):
    max_size = int((getattr(rm, "config", {}) or {}).get("client_max_size", 64 * 1024 * 1024))
    app = web.Application(client_max_size=max_size)
    app["rm"] = rm
    app["sm"] = sm
    app["auth"] = auth
    app["routes"] = routes
    return app


def _client_host(request):
    peer = request.headers.get("X-Forwarded-For", "").split(",", 1)[0].strip()
    if peer:
        return peer
    peername = request.transport.get_extra_info("peername") if request.transport else None
    if isinstance(peername, tuple) and peername:
        return peername[0]
    return "unknown"


def _backend_label(url):
    parsed = urlsplit(url)
    if parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    return url


def _session_label(session_id):
    if session_id is None:
        return "none"
    text = str(session_id)
    if len(text) <= 16:
        return text
    return f"{text[:8]}..{text[-4:]}"


def _coerce_session_value(value):
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (str, int, float)):
        text = str(value).strip()
        return text or None
    return None


def _qwen_prompt_session(prompt_id):
    """Derive a stable routing key from qwen-code metadata.promptId."""
    prompt_id = _coerce_session_value(prompt_id)
    if not prompt_id:
        return None

    delimiter_at = prompt_id.find(_QWEN_MAIN_PROMPT_DELIMITER)
    if delimiter_at > 0:
        return prompt_id[:delimiter_at]

    parts = prompt_id.split("#")
    if len(parts) >= 3 and parts[0] and parts[1] and parts[2].isdigit():
        return f"{parts[0]}#{parts[1]}"

    return None


def _extract_session_marker(body, headers=None):
    """Return (session_id, source) from common OpenAI-compatible extensions."""
    if isinstance(body, dict):
        for key in _SESSION_BODY_KEYS:
            session_id = _coerce_session_value(body.get(key))
            if session_id:
                return session_id, f"body.{key}"

        metadata = body.get("metadata")
        if isinstance(metadata, dict):
            session_id = _qwen_prompt_session(metadata.get("promptId"))
            if session_id:
                return session_id, "metadata.promptId:qwen"

            for key in _SESSION_METADATA_KEYS:
                session_id = _coerce_session_value(metadata.get(key))
                if session_id:
                    return session_id, f"metadata.{key}"

    if headers:
        for key in _SESSION_HEADER_KEYS:
            session_id = _coerce_session_value(headers.get(key))
            if session_id:
                return session_id, f"header.{key}"

    return None, None


def _strip_local_session_markers(body):
    """Remove llm-proxy session extensions before sending to local llama.cpp."""
    if not isinstance(body, dict):
        return body
    for key in _SESSION_BODY_KEYS:
        body.pop(key, None)
    metadata = body.get("metadata")
    if isinstance(metadata, dict):
        for key in _SESSION_METADATA_KEYS:
            metadata.pop(key, None)
        if not metadata:
            body.pop("metadata", None)
    return body


def _add_session_header(headers, session_id):
    if session_id and "X-LLM-Proxy-Session-ID" not in headers:
        headers["X-LLM-Proxy-Session-ID"] = session_id
    return headers


def create_api_app(runtime_manager, slot_manager, auth_manager=None, route_manager=None):
    """Public API surface: inference + discovery. Token-gated when auth.enabled."""
    app = _setup_app(runtime_manager, slot_manager, auth_manager, routes=route_manager)
    app.router.add_post("/stt", handle_stt_proxy)
    app.router.add_get("/stt/health", handle_stt_health)
    app.router.add_post("/v1/audio/speech", handle_tts_proxy)
    app.router.add_get("/tts/health", handle_tts_health)
    app.router.add_post("/v1/chat/completions", handle_chat)
    app.router.add_get("/v1/models", handle_models)
    app.router.add_get("/health", handle_health)
    return app


def create_control_app(runtime_manager, slot_manager, auth_manager=None, route_manager=None):
    """Control plane: web UI + runtime/token/password management.

    Open by default. Set a web password to require login.
    """
    app = _setup_app(runtime_manager, slot_manager, auth_manager, routes=route_manager)
    app.router.add_post("/login", handle_login)
    app.router.add_post("/logout", handle_logout)
    app.router.add_get("/session", handle_session)
    app.router.add_get("/v1/control/runtimes", handle_runtimes)
    app.router.add_post("/v1/control/runtimes", handle_runtime_create)
    app.router.add_post("/v1/control/runtimes/{name}/delete", handle_runtime_delete)
    app.router.add_post("/v1/control/runtimes/{name}/start", handle_runtime_start)
    app.router.add_post("/v1/control/runtimes/{name}/stop", handle_runtime_stop)
    app.router.add_post("/v1/control/runtimes/{name}/settings", handle_runtime_settings)
    app.router.add_get("/v1/control/runtimes/{name}/logs", handle_runtime_logs)
    app.router.add_get("/v1/control/state", handle_control_state)
    app.router.add_post("/v1/control/kv-cache/cleanup", handle_kv_cache_cleanup)
    app.router.add_post("/v1/control/models/{name}/load", handle_load)
    app.router.add_post("/v1/control/models/{name}/unload", handle_unload)
    app.router.add_get("/v1/control/whoami", handle_control_whoami)
    app.router.add_get("/v1/control/tokens", handle_tokens_list)
    app.router.add_post("/v1/control/tokens", handle_tokens_create)
    app.router.add_delete("/v1/control/tokens/{id}", handle_tokens_revoke)
    app.router.add_post("/v1/control/password", handle_password_set)
    app.router.add_delete("/v1/control/password", handle_password_clear)
    app.router.add_post("/v1/control/api-auth", handle_api_auth_set)
    app.router.add_get("/v1/control/routes", handle_routes_list)
    app.router.add_post("/v1/control/routes", handle_routes_create)
    app.router.add_delete("/v1/control/routes/{name}", handle_routes_delete)
    app.router.add_post("/v1/control/routes/{name}/backends", handle_routes_add_backend)
    app.router.add_delete("/v1/control/routes/{name}/backends/{index}", handle_routes_remove_backend)
    app.router.add_get("/v1/control/tls", handle_tls_status)
    app.router.add_post("/v1/control/tls", handle_tls_toggle)
    app.router.add_post("/v1/control/tls/generate-cert", handle_tls_generate_cert)
    app.router.add_get("/", handle_panel)
    return app


def _resolve_model(rm, model_name, route_manager=None):
    """Resolve model name, allowing partial matches. Returns (name, error_response)."""
    models = rm.available_models()
    if model_name in models:
        return model_name, None
    if route_manager and route_manager.get_route(model_name):
        return model_name, None
    matched = [m for m in models if model_name in m]
    if len(matched) == 1:
        return matched[0], None

    if route_manager:
        route_name, route_err = route_manager.resolve_model(model_name)
        if route_err:
            return None, web.json_response({"error": route_err}, status=400)
        if route_name:
            return route_name, None

    if not matched:
        all_models = list(models)
        if route_manager:
            all_models.extend(route_manager.available_routes())
        return None, web.json_response(
            {"error": f"Unknown model: {model_name}. Available: {all_models}"},
            status=404,
        )
    return None, web.json_response(
        {"error": f"Ambiguous model: {model_name}. Matches: {matched}"},
        status=400,
    )


async def _acquire_slot(rm, sm, model_name, session_id, port):
    """Acquire an exclusive slot lease for this local backend request."""
    num_slots = rm.get_slots(model_name)
    if model_name not in sm._slots:
        sm.init_model(model_name, num_slots)
    return await sm.acquire(model_name, session_id, port)


async def handle_chat(request):
    """Proxy /v1/chat/completions to the right backend (routed or local)."""
    request_started = now()
    _require_client_auth(request)
    rm = request.app["rm"]
    sm = request.app["sm"]
    route_manager = request.app.get("routes")

    body = await request.json()
    model_name = body.get("model", "")
    session_id, session_source = _extract_session_marker(body, request.headers)
    stream = body.get("stream", False)
    host = _client_host(request)
    log_event(
        "request received",
        kind="chat",
        host=host,
        model=model_name,
        session=_session_label(session_id),
        session_source=session_source,
        stream=stream,
    )

    model_name, err = _resolve_model(rm, model_name, route_manager=route_manager)
    if err:
        log_event("request rejected", kind="chat", host=host, model=model_name, ms=elapsed_ms(request_started))
        return err

    if route_manager:
        route = route_manager.get_route(model_name)
        if route:
            log_event("request route", kind="chat", host=host, model=model_name, route=route.name)
            return await _proxy_routed(
                request,
                route,
                body,
                stream,
                request_started,
                session_id=session_id,
                session_source=session_source,
            )

    if not rm.is_running(model_name):
        if rm.on_demand(model_name):
            try:
                ensure_started = now()
                log_event("runtime ensure", runtime=model_name, reason="on-demand")
                await rm.ensure(model_name)
                log_event("runtime ensured", runtime=model_name, ms=elapsed_ms(ensure_started))
            except Exception as e:
                log_event("runtime ensure failed", runtime=model_name, ms=elapsed_ms(ensure_started), error=e)
                log_event("request failed", kind="chat", host=host, model=model_name, status=503, ms=elapsed_ms(request_started), error=e)
                return web.json_response(
                    {"error": f"Failed to start {model_name} on demand: {e}"},
                    status=503,
                )
        else:
            log_event("request rejected", kind="chat", host=host, model=model_name, status=503, reason="model-not-loaded", ms=elapsed_ms(request_started))
            return web.json_response(
                {"error": f"Model '{model_name}' is not loaded. Load it explicitly via control API or web UI, or enable on_demand."},
                status=503,
            )

    port = rm.get_port(model_name)
    if not port:
        log_event("request rejected", kind="chat", host=host, model=model_name, status=503, reason="missing-port", ms=elapsed_ms(request_started))
        return web.json_response(
            {"error": f"Model '{model_name}' is marked running but has no bound port."},
            status=503,
        )

    slot_id = await _acquire_slot(rm, sm, model_name, session_id, port)
    _strip_local_session_markers(body)
    if slot_id is not None:
        body["id_slot"] = slot_id

    backend_url = f"http://127.0.0.1:{port}/v1/chat/completions"
    log_event(
        "request forward",
        kind="chat",
        host=host,
        model=model_name,
        backend=_backend_label(backend_url),
        slot=slot_id,
        session=_session_label(session_id),
        session_source=session_source,
        stream=stream,
    )

    try:
        if stream:
            return await _proxy_stream(
                request,
                backend_url,
                body,
                op={"kind": "chat", "host": host, "model": model_name, "started": request_started},
            )
        else:
            return await _proxy_json(
                backend_url,
                body,
                op={"kind": "chat", "host": host, "model": model_name, "started": request_started},
            )
    finally:
        if slot_id is not None:
            await sm.release(model_name, slot_id)


async def _proxy_routed(
    request,
    route,
    body,
    stream,
    request_started=None,
    session_id=None,
    session_source=None,
):
    """Forward request through a Route's failover chain."""
    from llm_proxy.route_manager import _derive_endpoint_url

    candidates = route.failover_candidates()
    last_err = None
    for backend in candidates:
        url = _derive_endpoint_url(backend["url"], route.kind)
        body["model"] = backend.get("model", route.name)
        headers = route._auth_headers(backend)
        _add_session_header(headers, session_id)
        host = _client_host(request)
        log_event(
            "request forward",
            kind=route.kind,
            host=host,
            model=body["model"],
            route=route.name,
            backend=_backend_label(url),
            session=_session_label(session_id),
            session_source=session_source,
            stream=stream,
        )
        try:
            if stream:
                return await _proxy_stream_inline(
                    request,
                    url,
                    body,
                    headers,
                    op={"kind": route.kind, "host": host, "model": body["model"], "started": request_started or now()},
                )
            else:
                return await _proxy_json_inline(
                    url,
                    body,
                    headers,
                    op={"kind": route.kind, "host": host, "model": body["model"], "started": request_started or now()},
                )
        except Exception as e:
            await route.mark_unhealthy(backend["url"])
            log_event(
                "request backend failed",
                kind=route.kind,
                host=host,
                model=body["model"],
                route=route.name,
                backend=_backend_label(url),
                error=e,
            )
            last_err = str(e)
            continue

    log_event("request failed", kind=route.kind, route=route.name, ms=elapsed_ms(request_started or now()))
    return web.json_response(
        {"error": f"All backends failed for '{route.name}'. Last error: {last_err}"},
        status=503,
    )


async def _proxy_json_inline(url, body, headers, op=None):
    """Forward request; raises on failure (for failover callers)."""
    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession() as sess:
        async with sess.post(url, json=body, headers=headers, timeout=timeout) as resp:
            data = await resp.json()
            if resp.status >= 500:
                raise aiohttp.ClientResponseError(
                    resp.request_info, resp.history, status=resp.status,
                    message=data.get("error", f"HTTP {resp.status}"),
                    headers=resp.headers,
                )
            _log_proxy_done(op, resp.status)
            return web.json_response(data, status=resp.status)


async def _proxy_stream_inline(request, url, body, headers, op=None):
    """Forward streaming request; raises on connection failure (enables failover)."""
    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession() as sess:
        async with sess.post(url, json=body, headers=headers, timeout=timeout) as backend:
            if backend.status >= 500:
                text = await backend.text()
                raise aiohttp.ClientResponseError(
                    backend.request_info, backend.history, status=backend.status,
                    message=text or f"HTTP {backend.status}",
                    headers=backend.headers,
                )
            status = backend.status
            response = web.StreamResponse(
                status=status,
                headers={
                    "Content-Type": "text/event-stream",
                    "Cache-Control": "no-cache",
                },
            )
            await response.prepare(request)
            async for chunk in backend.content.iter_any():
                await response.write(chunk)
    await response.write_eof()
    _log_proxy_done(op, status)
    return response


async def _proxy_json(url, body, extra_headers=None, op=None):
    """Forward request and return JSON response."""
    try:
        async with aiohttp.ClientSession() as sess:
            headers = extra_headers or {}
            async with sess.post(url, json=body, headers=headers,
                                timeout=aiohttp.ClientTimeout(total=300)) as resp:
                data = await resp.json()
                _log_proxy_done(op, resp.status)
                return web.json_response(data, status=resp.status)
    except Exception as e:
        _log_proxy_done(op, 502, error=e)
        return web.json_response({"error": str(e)}, status=502)


async def _proxy_stream(request, url, body, extra_headers=None, op=None):
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
            headers = extra_headers or {}
            async with sess.post(url, json=body, headers=headers, timeout=timeout) as backend:
                status = backend.status
                async for chunk in backend.content.iter_any():
                    await response.write(chunk)
    except Exception as e:
        status = 502
        error_chunk = f"data: {json.dumps({'error': str(e)})}\n\n"
        await response.write(error_chunk.encode())
        _log_proxy_done(op, status, error=e)
    else:
        _log_proxy_done(op, status)

    await response.write_eof()
    return response


def _log_proxy_done(op, status, error=None):
    if not op:
        return
    log_event(
        "request complete",
        kind=op.get("kind"),
        host=op.get("host"),
        model=op.get("model"),
        status=status,
        ms=elapsed_ms(op.get("started", now())),
        error=error,
    )


async def handle_models(request):
    """GET /v1/models — list available and loaded models (local + routed)."""
    rm = request.app["rm"]
    route_manager = request.app.get("routes")
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
    if route_manager:
        for name in route_manager.available_routes():
            route = route_manager.get_route(name)
            models.append({
                "id": name,
                "object": "model",
                "created": 0,
                "owned_by": "routed",
                "status": "routed",
                "route_kind": route.kind if route else None,
            })
    return web.json_response({"object": "list", "data": models})


async def handle_load(request):
    """POST /v1/control/models/{name}/load — explicitly load a model."""
    started = now()
    _require_control_auth(request)
    rm = request.app["rm"]
    name = request.match_info["name"]
    log_event("model load requested", host=_client_host(request), model=name)
    if name not in rm.available_models():
        log_event("model load rejected", host=_client_host(request), model=name, status=404, ms=elapsed_ms(started))
        return web.json_response({"error": f"Unknown model: {name}"}, status=404)
    try:
        await rm.start(name)
        log_event("model load complete", host=_client_host(request), model=name, status=200, ms=elapsed_ms(started))
        return web.json_response({"status": "loaded", "model": name})
    except Exception as e:
        log_event("model load failed", host=_client_host(request), model=name, status=500, ms=elapsed_ms(started), error=e)
        return web.json_response({"error": str(e)}, status=500)


async def handle_unload(request):
    """POST /v1/control/models/{name}/unload — stop a model."""
    started = now()
    _require_control_auth(request)
    rm = request.app["rm"]
    sm = request.app["sm"]
    name = request.match_info["name"]
    log_event("model unload requested", host=_client_host(request), model=name)

    port = rm.get_port(name)
    if port:
        await sm.save_all(name, port)
        sm.remove_model(name)
    await rm.stop(name)
    log_event("model unload complete", host=_client_host(request), model=name, status=200, ms=elapsed_ms(started))
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
    """Proxy STT client traffic to the active STT runtime or route."""
    request_started = now()
    _require_client_auth(request)
    rm = request.app["rm"]
    route_manager = request.app.get("routes")
    host = _client_host(request)
    log_event("request received", kind="stt", host=host)

    if route_manager:
        stt_routes = [n for n in route_manager.available_routes()
                      if route_manager.get_route(n).kind == "stt"]
        if stt_routes:
            route_name = stt_routes[0]
            route = route_manager.get_route(route_name)
            from llm_proxy.route_manager import _derive_endpoint_url
            body = await request.read()
            candidates = route.failover_candidates()
            last_err = None
            for backend in candidates:
                url = _derive_endpoint_url(backend["url"], "stt")
                headers = route._auth_headers(backend)
                if request.content_type:
                    headers["Content-Type"] = request.headers.get("Content-Type", request.content_type)
                prompt = request.headers.get("X-Initial-Prompt")
                if prompt:
                    headers["X-Initial-Prompt"] = prompt
                query_string = request.query_string
                if query_string:
                    url += "?" + query_string
                log_event("request forward", kind="stt", host=host, route=route.name, backend=_backend_label(url))
                try:
                    async with aiohttp.ClientSession() as sess:
                        async with sess.post(
                            url, data=body, headers=headers,
                            timeout=aiohttp.ClientTimeout(total=300),
                        ) as resp:
                            if resp.status >= 500:
                                text = await resp.text()
                                raise aiohttp.ClientResponseError(
                                    resp.request_info, resp.history, status=resp.status,
                                    message=text or f"HTTP {resp.status}",
                                    headers=resp.headers,
                                )
                            raw = await resp.read()
                            content_type = resp.headers.get("Content-Type", "application/json")
                            log_event(
                                "request complete",
                                kind="stt",
                                host=host,
                                route=route.name,
                                status=resp.status,
                                ms=elapsed_ms(request_started),
                            )
                            return web.Response(body=raw, status=resp.status,
                                               content_type=content_type.split(";")[0])
                except Exception as e:
                    await route.mark_unhealthy(backend["url"])
                    log_event("request backend failed", kind="stt", host=host, route=route.name, backend=_backend_label(url), error=e)
                    last_err = str(e)
                    continue
            log_event("request failed", kind="stt", host=host, route=route.name, ms=elapsed_ms(request_started))
            return web.json_response(
                {"error": f"All STT backends failed. Last error: {last_err}"},
                status=503,
            )

    runtime_name, err = _resolve_stt_runtime(rm)
    if err:
        return err
    if not rm.is_running(runtime_name):
        if rm.on_demand(runtime_name):
            try:
                ensure_started = now()
                log_event("runtime ensure", runtime=runtime_name, reason="on-demand")
                await rm.ensure(runtime_name)
                log_event("runtime ensured", runtime=runtime_name, ms=elapsed_ms(ensure_started))
            except Exception as e:
                log_event("runtime ensure failed", runtime=runtime_name, ms=elapsed_ms(ensure_started), error=e)
                log_event("request failed", kind="stt", host=host, runtime=runtime_name, status=503, ms=elapsed_ms(request_started), error=e)
                return web.json_response(
                    {"error": f"Failed to start {runtime_name} on demand: {e}"},
                    status=503,
                )
        else:
            log_event("request rejected", kind="stt", host=host, runtime=runtime_name, status=503, reason="runtime-not-loaded", ms=elapsed_ms(request_started))
            return web.json_response(
                {"error": f"STT runtime '{runtime_name}' is not loaded. Start it explicitly via control API or web UI, or enable on_demand."},
                status=503,
            )
    port = rm.get_port(runtime_name)
    if not port:
        log_event("request rejected", kind="stt", host=host, runtime=runtime_name, status=503, reason="missing-port", ms=elapsed_ms(request_started))
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
    log_event("request forward", kind="stt", host=host, runtime=runtime_name, backend=_backend_label(backend_url))

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
                log_event(
                    "request complete",
                    kind="stt",
                    host=host,
                    runtime=runtime_name,
                    status=resp.status,
                    ms=elapsed_ms(request_started),
                )
                return web.Response(body=raw, status=resp.status, content_type=content_type.split(";")[0])
    except Exception as e:
        log_event("request complete", kind="stt", host=host, runtime=runtime_name, status=502, ms=elapsed_ms(request_started), error=e)
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
    request_started = now()
    _require_client_auth(request)
    rm = request.app["rm"]
    route_manager = request.app.get("routes")
    body = await request.json()
    requested = body.pop("voice", None) if isinstance(body, dict) else None
    host = _client_host(request)
    log_event("request received", kind="tts", host=host, runtime=requested)

    if route_manager and requested:
        route = route_manager.get_route(requested)
        if route and route.kind == "tts":
            from llm_proxy.route_manager import _derive_endpoint_url
            candidates = route.failover_candidates()
            last_err = None
            for backend in candidates:
                url = _derive_endpoint_url(backend["url"], "tts")
                headers = route._auth_headers(backend)
                log_event("request forward", kind="tts", host=host, route=route.name, backend=_backend_label(url))
                try:
                    async with aiohttp.ClientSession() as sess:
                        async with sess.post(url, json=body, headers=headers,
                                            timeout=aiohttp.ClientTimeout(total=300)) as backend_resp:
                            if backend_resp.status >= 500:
                                text = await backend_resp.text()
                                raise aiohttp.ClientResponseError(
                                    backend_resp.request_info,
                                    backend_resp.history,
                                    status=backend_resp.status,
                                    message=text or f"HTTP {backend_resp.status}",
                                    headers=backend_resp.headers,
                                )
                            response = web.StreamResponse()
                            response.set_status(backend_resp.status)
                            for hk, hv in backend_resp.headers.items():
                                if hk.lower() in ("content-length", "transfer-encoding", "connection"):
                                    continue
                                response.headers[hk] = hv
                            await response.prepare(request)
                            async for chunk in backend_resp.content.iter_any():
                                await response.write(chunk)
                            await response.write_eof()
                            log_event(
                                "request complete",
                                kind="tts",
                                host=host,
                                route=route.name,
                                status=backend_resp.status,
                                ms=elapsed_ms(request_started),
                            )
                            return response
                except Exception as e:
                    await route.mark_unhealthy(backend["url"])
                    log_event("request backend failed", kind="tts", host=host, route=route.name, backend=_backend_label(url), error=e)
                    last_err = str(e)
                    continue
            log_event("request failed", kind="tts", host=host, route=route.name, ms=elapsed_ms(request_started))
            return web.json_response(
                {"error": f"All TTS backends failed. Last error: {last_err}"},
                status=503,
            )

    runtime_name, err = _resolve_tts_runtime(rm, requested)
    if err:
        return err
    if not rm.is_running(runtime_name):
        if rm.on_demand(runtime_name):
            try:
                ensure_started = now()
                log_event("runtime ensure", runtime=runtime_name, reason="on-demand")
                await rm.ensure(runtime_name)
                log_event("runtime ensured", runtime=runtime_name, ms=elapsed_ms(ensure_started))
            except Exception as e:
                log_event("runtime ensure failed", runtime=runtime_name, ms=elapsed_ms(ensure_started), error=e)
                log_event("request failed", kind="tts", host=host, runtime=runtime_name, status=503, ms=elapsed_ms(request_started), error=e)
                return web.json_response(
                    {"error": f"Failed to start {runtime_name} on demand: {e}"},
                    status=503,
                )
        else:
            log_event("request rejected", kind="tts", host=host, runtime=runtime_name, status=503, reason="runtime-not-loaded", ms=elapsed_ms(request_started))
            return web.json_response(
                {"error": f"TTS runtime '{runtime_name}' is not loaded. Start it explicitly via control API or web UI, or enable on_demand."},
                status=503,
            )
    port = rm.get_port(runtime_name)
    if not port:
        log_event("request rejected", kind="tts", host=host, runtime=runtime_name, status=503, reason="missing-port", ms=elapsed_ms(request_started))
        return web.json_response(
            {"error": f"TTS runtime '{runtime_name}' is marked running but has no bound port."},
            status=503,
        )

    backend_url = f"http://127.0.0.1:{port}/tts/generate"
    log_event("request forward", kind="tts", host=host, runtime=runtime_name, backend=_backend_label(backend_url))
    response = web.StreamResponse()
    status = None
    try:
        async with aiohttp.ClientSession() as sess:
            async with sess.post(backend_url, json=body, timeout=aiohttp.ClientTimeout(total=300)) as backend:
                status = backend.status
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
            log_event("request complete", kind="tts", host=host, runtime=runtime_name, status=502, ms=elapsed_ms(request_started), error=e)
            return web.json_response({"error": str(e)}, status=502)
        log_event("request complete", kind="tts", host=host, runtime=runtime_name, status=502, ms=elapsed_ms(request_started), error=e)
        await response.write_eof()
        return response

    await response.write_eof()
    log_event("request complete", kind="tts", host=host, runtime=runtime_name, status=status, ms=elapsed_ms(request_started))
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
        return web.json_response({"error": "no password is set"}, status=400)
    body = await request.json()
    try:
        session_id = auth.web_login(body.get("password", ""))
    except PermissionError as e:
        return web.json_response({"error": str(e)}, status=401)

    response = web.json_response({"ok": True})
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
    web_auth_enabled = bool(auth.web_enabled) if auth is not None else False
    api_auth_enabled = bool(auth.enabled) if auth is not None else False
    return web.json_response({
        "authenticated": bool(username),
        "username": username,
        "web_auth_enabled": web_auth_enabled,
        "api_auth_enabled": api_auth_enabled,
    })


async def handle_control_whoami(request):
    """GET /v1/control/whoami — validate control credentials."""
    identity = _require_control_auth(request)
    return web.json_response({"ok": True, "identity": identity})


async def handle_tokens_list(request):
    """GET /v1/control/tokens — list issued tokens (no secrets)."""
    _require_control_auth(request)
    auth = request.app.get("auth")
    if auth is None:
        return web.json_response({"tokens": []})
    return web.json_response({"tokens": auth.list_tokens()})


async def handle_tokens_create(request):
    """POST /v1/control/tokens — create a token. Plain value returned ONCE."""
    _require_control_auth(request)
    auth = request.app.get("auth")
    if auth is None:
        return web.json_response({"error": "auth is disabled"}, status=400)
    body = await request.json()
    try:
        record = auth.create_token(body.get("name", ""))
    except ValueError as e:
        return web.json_response({"error": str(e)}, status=400)
    return web.json_response({"ok": True, "token": record})


async def handle_api_auth_set(request):
    """POST /v1/control/api-auth — toggle API token requirement."""
    _require_control_auth(request)
    auth = request.app.get("auth")
    if auth is None:
        return web.json_response({"error": "auth subsystem not configured"}, status=400)
    body = await request.json()
    new_state = auth.set_api_auth_enabled(bool(body.get("enabled")))
    return web.json_response({"ok": True, "api_auth_enabled": new_state})


async def handle_password_set(request):
    """POST /v1/control/password — set/change UI password.

    Body: {password, current_password?}. current_password required iff one is already set.
    Setting a password enables the login screen for the web UI.
    """
    _require_control_auth(request)
    auth = request.app.get("auth")
    if auth is None:
        return web.json_response({"error": "auth subsystem not configured"}, status=400)
    body = await request.json()
    try:
        auth.set_web_password(
            body.get("password", ""),
            current_password=body.get("current_password"),
        )
    except ValueError as e:
        return web.json_response({"error": str(e)}, status=400)
    except PermissionError as e:
        return web.json_response({"error": str(e)}, status=403)
    return web.json_response({"ok": True, "web_auth_enabled": auth.web_enabled})


async def handle_password_clear(request):
    """DELETE /v1/control/password — remove UI password (open access via web)."""
    _require_control_auth(request)
    auth = request.app.get("auth")
    if auth is None:
        return web.json_response({"error": "auth subsystem not configured"}, status=400)
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass
    try:
        cleared = auth.clear_web_password(current_password=body.get("current_password"))
    except PermissionError as e:
        return web.json_response({"error": str(e)}, status=403)
    return web.json_response({"ok": True, "cleared": cleared, "web_auth_enabled": auth.web_enabled})


async def handle_tokens_revoke(request):
    """DELETE /v1/control/tokens/{id} — revoke by id or name."""
    _require_control_auth(request)
    auth = request.app.get("auth")
    if auth is None:
        return web.json_response({"error": "auth is disabled"}, status=400)
    identifier = request.match_info.get("id", "")
    ok = auth.revoke_token(identifier)
    if not ok:
        return web.json_response({"error": f"no such token: {identifier}"}, status=404)
    return web.json_response({"ok": True, "revoked": identifier})


async def handle_control_state(request):
    """GET /v1/control/state — current state for dashboard."""
    _require_control_auth(request)
    rm = request.app["rm"]
    runtimes = rm.list_state()
    models = [rt for rt in runtimes if rt["kind"] == "llm"]
    return web.json_response({"models": models, "runtimes": runtimes})


async def handle_kv_cache_cleanup(request):
    """POST /v1/control/kv-cache/cleanup — run KV cache cleanup now."""
    _require_control_auth(request)
    manager = request.app.get("kv_cache")
    if manager is None:
        return web.json_response({"error": "kv cache manager is not available"}, status=500)
    result = await manager.cleanup_once(reason="manual")
    return web.json_response({
        "ok": True,
        "scanned": result.scanned,
        "deleted": result.deleted,
        "bytes_deleted": result.bytes_deleted,
    })


async def handle_runtimes(request):
    """GET /v1/control/runtimes — list runtimes."""
    _require_control_auth(request)
    rm = request.app["rm"]
    return web.json_response({"runtimes": rm.list_state()})


async def handle_runtime_create(request):
    """POST /v1/control/runtimes — create a new runtime.

    Body: {name, driver, gpu?, autoload?, on_demand?,
           port, command, slots?, venv?, python_bin?, env?, ...}
    env may be a dict or a 'KEY=value' string (one per line).
    """
    _require_control_auth(request)
    rm = request.app["rm"]
    body = await request.json()

    runtime_body = dict(body)
    profile = runtime_body.get("profile")
    if isinstance(profile, dict):
        for key, value in profile.items():
            if key != "name" and key not in runtime_body:
                runtime_body[key] = value

    if "env" in runtime_body:
        try:
            from llm_proxy.runtime_manager import _parse_env
            runtime_body["env"] = _parse_env(runtime_body["env"])
        except ValueError as e:
            return web.json_response({"error": f"invalid env: {e}"}, status=400)
        if not runtime_body["env"]:
            runtime_body.pop("env", None)

    try:
        await rm.create_runtime(runtime_body)
    except ValueError as e:
        msg = str(e)
        status = 409 if "already exists" in msg else 400
        return web.json_response({"error": msg}, status=status)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)

    name = body.get("name")
    return web.json_response({"ok": True, "runtime": rm.runtime_state(name)})


async def handle_runtime_delete(request):
    """POST /v1/control/runtimes/{name}/delete — remove a runtime."""
    _require_control_auth(request)
    rm = request.app["rm"]
    name = request.match_info["name"]
    try:
        await rm.delete_runtime(name)
    except ValueError as e:
        msg = str(e)
        status = 404 if "Unknown runtime" in msg else 400
        return web.json_response({"error": msg}, status=status)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)
    return web.json_response({"ok": True, "deleted": name})


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


async def handle_routes_list(request):
    """GET /v1/control/routes — list all routes with health status."""
    _require_control_auth(request)
    route_manager = request.app.get("routes")
    if route_manager is None:
        return web.json_response({"routes": []})
    return web.json_response({"routes": route_manager.list_routes_status()})


async def handle_routes_create(request):
    """POST /v1/control/routes — create a new route."""
    _require_control_auth(request)
    route_manager = request.app.get("routes")
    if route_manager is None:
        return web.json_response({"error": "route manager not available"}, status=500)
    body = await request.json()
    try:
        await route_manager.create_route(body)
    except ValueError as e:
        msg = str(e)
        status = 409 if "already exists" in msg else 400
        return web.json_response({"error": msg}, status=status)
    name = body.get("name")
    route = route_manager.get_route(name)
    return web.json_response({"ok": True, "route": route.status() if route else None})


async def handle_routes_delete(request):
    """DELETE /v1/control/routes/{name} — delete a route."""
    _require_control_auth(request)
    route_manager = request.app.get("routes")
    if route_manager is None:
        return web.json_response({"error": "route manager not available"}, status=500)
    name = request.match_info["name"]
    try:
        await route_manager.delete_route(name)
    except ValueError as e:
        msg = str(e)
        status = 404 if "Unknown route" in msg else 400
        return web.json_response({"error": msg}, status=status)
    return web.json_response({"ok": True, "deleted": name})


async def handle_routes_add_backend(request):
    """POST /v1/control/routes/{name}/backends — add a backend to a route."""
    _require_control_auth(request)
    route_manager = request.app.get("routes")
    if route_manager is None:
        return web.json_response({"error": "route manager not available"}, status=500)
    name = request.match_info["name"]
    body = await request.json()
    try:
        await route_manager.add_backend(name, body)
    except ValueError as e:
        return web.json_response({"error": str(e)}, status=400)
    route = route_manager.get_route(name)
    return web.json_response({"ok": True, "route": route.status() if route else None})


async def handle_routes_remove_backend(request):
    """DELETE /v1/control/routes/{name}/backends/{index} — remove a backend by index."""
    _require_control_auth(request)
    route_manager = request.app.get("routes")
    if route_manager is None:
        return web.json_response({"error": "route manager not available"}, status=500)
    name = request.match_info["name"]
    try:
        index = int(request.match_info["index"])
    except ValueError:
        return web.json_response({"error": "backend index must be an integer"}, status=400)
    try:
        await route_manager.remove_backend(name, index)
    except ValueError as e:
        return web.json_response({"error": str(e)}, status=400)
    route = route_manager.get_route(name)
    return web.json_response({"ok": True, "route": route.status() if route else None, "removed_index": index})


async def handle_tls_status(request):
    _require_control_auth(request)
    config = request.app.get("config") or {}
    tls_cfg = config.get("tls") or {}
    cert_file = os.path.expanduser(tls_cfg.get("cert_file", ""))
    cert_exists = os.path.isfile(cert_file)
    fingerprint = None
    if cert_exists:
        try:
            fingerprint = certificate_fingerprint(cert_file)
        except Exception:
            pass
    return web.json_response({
        "enabled": bool(tls_cfg.get("enabled", False)),
        "cert_file": cert_file,
        "key_file": os.path.expanduser(tls_cfg.get("key_file", "")),
        "cert_exists": cert_exists,
        "fingerprint": fingerprint,
    })


async def handle_tls_toggle(request):
    _require_control_auth(request)
    config = request.app.get("config") or {}
    body = await request.json()
    desired = bool(body.get("enabled", False))

    config_path = config.get("_config_path")
    if not config_path:
        config_path = os.path.expanduser("~/.config/llm-proxy/config.yaml")

    current = load_config_file(config_path)
    current.setdefault("tls", {})["enabled"] = desired
    save_config_file(config_path, current)

    config["tls"] = config.get("tls") or {}
    config["tls"]["enabled"] = desired

    return web.json_response({
        "ok": True,
        "enabled": desired,
        "restart_needed": True,
    })


async def handle_tls_generate_cert(request):
    _require_control_auth(request)
    config = request.app.get("config") or {}
    body = await request.json()
    host = body.get("host", "llm-proxy")
    ip = body.get("ip")

    state_dir = config.get("state_dir", "~/.config/llm-proxy")
    generated = generate_server_cert(state_dir, host=host, ip=ip)

    config_path = config.get("_config_path")
    if not config_path:
        config_path = os.path.expanduser("~/.config/llm-proxy/config.yaml")

    current = load_config_file(config_path)
    current.setdefault("tls", {})["cert_file"] = generated["cert"]
    current.setdefault("tls", {})["key_file"] = generated["key"]
    current.setdefault("tls", {})["enabled"] = True
    save_config_file(config_path, current)

    config["tls"] = config.get("tls") or {}
    config["tls"]["cert_file"] = generated["cert"]
    config["tls"]["key_file"] = generated["key"]
    config["tls"]["enabled"] = True

    return web.json_response({
        "ok": True,
        "cert_file": generated["cert"],
        "key_file": generated["key"],
        "fingerprint": generated["fingerprint"],
        "enabled": True,
        "restart_needed": True,
    })


async def handle_panel(request):
    """Simple built-in control panel."""
    return web.Response(text=_panel_html(), content_type="text/html")


def _panel_html():
    path = os.path.join(os.path.dirname(__file__), "panel.html")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
