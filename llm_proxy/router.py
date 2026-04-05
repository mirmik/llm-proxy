"""Router — OpenAI-compatible proxy with model routing and session awareness."""

import json

import aiohttp
from aiohttp import web


def create_app(process_manager, slot_manager):
    """Create aiohttp app with all routes."""
    app = web.Application()
    app["pm"] = process_manager
    app["sm"] = slot_manager

    app.router.add_post("/v1/chat/completions", handle_chat)
    app.router.add_get("/v1/models", handle_models)
    app.router.add_post("/v1/models/{name}/load", handle_load)
    app.router.add_post("/v1/models/{name}/unload", handle_unload)
    app.router.add_get("/health", handle_health)

    return app


def _resolve_model(pm, model_name):
    """Resolve model name, allowing partial matches. Returns (name, error_response)."""
    if model_name in pm.models:
        return model_name, None
    matched = [m for m in pm.models if model_name in m]
    if len(matched) == 1:
        return matched[0], None
    if not matched:
        return None, web.json_response(
            {"error": f"Unknown model: {model_name}. Available: {pm.available_models()}"},
            status=404,
        )
    return None, web.json_response(
        {"error": f"Ambiguous model: {model_name}. Matches: {matched}"},
        status=400,
    )


async def _ensure_slot(pm, sm, model_name, session_id, port):
    """Assign a slot for the session if session_id is provided. Returns slot_id or None."""
    if not session_id:
        return None
    num_slots = pm.models[model_name].get("slots", 1)
    if model_name not in sm._slots:
        sm.init_model(model_name, num_slots)
    return await sm.assign(model_name, session_id, port)


async def handle_chat(request):
    """Proxy /v1/chat/completions to the right llama-server backend."""
    pm = request.app["pm"]
    sm = request.app["sm"]

    body = await request.json()
    model_name = body.get("model", "")
    session_id = body.pop("session_id", None)
    stream = body.get("stream", False)

    model_name, err = _resolve_model(pm, model_name)
    if err:
        return err

    try:
        port = await pm.ensure(model_name)
    except Exception as e:
        return web.json_response({"error": f"Failed to start {model_name}: {e}"}, status=503)

    slot_id = await _ensure_slot(pm, sm, model_name, session_id, port)
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
    pm = request.app["pm"]
    models = []
    for name in pm.available_models():
        models.append({
            "id": name,
            "object": "model",
            "created": 0,
            "owned_by": "local",
            "status": "loaded" if pm.is_running(name) else "available",
            "port": pm.get_port(name),
        })
    return web.json_response({"object": "list", "data": models})


async def handle_load(request):
    """POST /v1/models/{name}/load — explicitly load a model."""
    pm = request.app["pm"]
    name = request.match_info["name"]
    if name not in pm.models:
        return web.json_response({"error": f"Unknown model: {name}"}, status=404)
    try:
        await pm.start(name)
        return web.json_response({"status": "loaded", "model": name})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


async def handle_unload(request):
    """POST /v1/models/{name}/unload — stop a model."""
    pm = request.app["pm"]
    sm = request.app["sm"]
    name = request.match_info["name"]

    port = pm.get_port(name)
    if port:
        await sm.save_all(name, port)
        sm.remove_model(name)
    await pm.stop(name)
    return web.json_response({"status": "unloaded", "model": name})


async def handle_health(request):
    """GET /health — proxy health."""
    pm = request.app["pm"]
    return web.json_response({
        "status": "ok",
        "loaded_models": pm.loaded_models(),
    })
