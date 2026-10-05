"""Authenticated tokenization without starting runtimes or allocating KV slots."""

import asyncio
import json

import aiohttp
from aiohttp import web

from llm_proxy.route_manager import backend_client_kwargs


async def handle_tokenize(request):
    # Router owns the shared authentication and model-resolution contracts.
    from llm_proxy.router import _require_client_auth, _resolve_model

    _require_client_auth(request)
    try:
        body = await request.json()
    except (ValueError, UnicodeDecodeError):
        return web.json_response({"error": "Request body must be valid JSON"}, status=400)
    if (not isinstance(body, dict)
            or not isinstance(body.get("model"), str) or not body["model"].strip()
            or not isinstance(body.get("content"), str)):
        return web.json_response(
            {"error": "model must be a non-empty string and content must be a string"},
            status=400,
        )
    rm = request.app["rm"]
    routes = request.app.get("routes")
    name, error = _resolve_model(rm, body["model"].strip(), route_manager=routes)
    if error is not None:
        return error
    route = routes.get_route(name) if routes else None
    if route is not None:
        if route.kind != "llm":
            return web.json_response({"error": "Tokenization requires an LLM"}, status=400)
        candidates = route.failover_candidates()
    else:
        port = rm.get_port(name) if rm.is_running(name) else None
        if not port:
            return web.json_response({"error": "Model is not loaded"}, status=503)
        candidates = [{"url": f"http://127.0.0.1:{port}", "model": name}]

    for backend in candidates:
        payload = {"model": backend.get("model", name), "content": body["content"]}
        headers = route._auth_headers(backend) if route else {}
        try:
            async with aiohttp.ClientSession(**backend_client_kwargs(backend)) as client:
                async with client.post(
                    backend["url"].rstrip("/") + "/tokenize",
                    json=payload, headers=headers,
                    timeout=aiohttp.ClientTimeout(total=5), allow_redirects=False,
                ) as response:
                    raw = await response.read()
                    try:
                        data = json.loads(raw)
                    except (ValueError, UnicodeDecodeError):
                        data = {"error": "Backend does not provide a JSON tokenizer response"}
                    if response.status >= 500:
                        continue
                    if response.status == 200 and (
                        not isinstance(data, dict) or not isinstance(data.get("tokens"), list)
                    ):
                        return web.json_response({"error": "Invalid tokenizer response"}, status=502)
                    return web.json_response(data, status=response.status)
        except (aiohttp.ClientError, asyncio.TimeoutError):
            # An auxiliary endpoint failing must not disable a healthy chat model.
            continue
    return web.json_response({"error": "Model tokenizer unavailable"}, status=503)
