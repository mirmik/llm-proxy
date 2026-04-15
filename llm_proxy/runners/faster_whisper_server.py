#!/usr/bin/env python3
"""Managed faster-whisper HTTP server for llm-proxy."""

import os

import numpy as np
from aiohttp import web
from faster_whisper import WhisperModel


HOST = "127.0.0.1"
PORT = int(os.environ.get("LLM_PROXY_FW_PORT", "5055"))
MODEL_SIZE = os.environ.get("LLM_PROXY_FW_MODEL_SIZE", "large-v3")
DEVICE = os.environ.get("LLM_PROXY_FW_DEVICE", "cuda")
COMPUTE_TYPE = os.environ.get("LLM_PROXY_FW_COMPUTE_TYPE", "float16")
SAMPLE_RATE = int(os.environ.get("LLM_PROXY_FW_SAMPLE_RATE", "16000"))
LANGUAGE = os.environ.get("LLM_PROXY_FW_LANGUAGE") or None
TOKEN = os.environ.get("LLM_PROXY_FW_TOKEN") or None


print(f"[faster-whisper] Loading model={MODEL_SIZE} device={DEVICE} compute={COMPUTE_TYPE}")
MODEL = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
print("[faster-whisper] Ready")


def _check_auth(request):
    if not TOKEN:
        return True
    header = request.headers.get("Authorization", "")
    return header.removeprefix("Bearer ").strip() == TOKEN


async def handle_stt(request):
    if not _check_auth(request):
        return web.json_response({"error": "unauthorized"}, status=401)

    if request.content_type and request.content_type.startswith("multipart/"):
        data = await request.post()
        if "audio" not in data:
            return web.json_response({"error": "missing audio file"}, status=400)
        raw = data["audio"].file.read()
        audio = np.frombuffer(raw, dtype=np.float32)
    else:
        raw = await request.read()
        audio = np.frombuffer(raw, dtype=np.float32)

    duration = len(audio) / SAMPLE_RATE if SAMPLE_RATE else 0.0
    if duration < 0.3:
        return web.json_response({"text": "", "duration": duration})

    prompt = request.query.get("prompt", "") or request.headers.get("X-Initial-Prompt", "")
    segments, info = MODEL.transcribe(
        audio,
        language=LANGUAGE,
        beam_size=5,
        initial_prompt=prompt or None,
    )
    text = " ".join(segment.text.strip() for segment in segments).strip()
    return web.json_response({
        "text": text,
        "duration": duration,
        "language": info.language,
    })


async def handle_health(request):
    return web.json_response({
        "status": "ok",
        "kind": "stt",
        "model_size": MODEL_SIZE,
        "language": LANGUAGE,
    })


def main():
    app = web.Application(client_max_size=64 * 1024 * 1024)
    app.router.add_post("/stt", handle_stt)
    app.router.add_get("/health", handle_health)
    web.run_app(app, host=HOST, port=PORT, handle_signals=True)


if __name__ == "__main__":
    main()
