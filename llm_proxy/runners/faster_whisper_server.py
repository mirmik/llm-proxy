#!/usr/bin/env python3
"""Managed faster-whisper HTTP server for llm-proxy.

Parameters are taken from CLI flags first, env vars as fallback.
CLI flags (each has a matching LLM_PROXY_FW_* env var):
    --host                   (LLM_PROXY_FW_HOST,         default 0.0.0.0)
    --port                   (LLM_PROXY_FW_PORT,         default 5055)
    --model-size             (LLM_PROXY_FW_MODEL_SIZE,   default large-v3)
    --device                 (LLM_PROXY_FW_DEVICE,       default cuda)
    --compute-type           (LLM_PROXY_FW_COMPUTE_TYPE, default float16)
    --sample-rate            (LLM_PROXY_FW_SAMPLE_RATE,  default 16000)
    --language               (LLM_PROXY_FW_LANGUAGE,     optional)
    --token                  (LLM_PROXY_FW_TOKEN,        optional bearer)
"""

import argparse
import os

import numpy as np
from aiohttp import web
from faster_whisper import WhisperModel

from llm_proxy.stt_protocol import InitialPromptEncodingError, decode_initial_prompt


def _env_or(name, default=None):
    value = os.environ.get(name)
    return value if value is not None and value != "" else default


def parse_args():
    p = argparse.ArgumentParser(description="faster-whisper HTTP server")
    p.add_argument("--host", default=_env_or("LLM_PROXY_FW_HOST", "0.0.0.0"))
    p.add_argument("--port", type=int, default=int(_env_or("LLM_PROXY_FW_PORT", "5055")))
    p.add_argument("--model-size", default=_env_or("LLM_PROXY_FW_MODEL_SIZE", "large-v3"))
    p.add_argument("--device", default=_env_or("LLM_PROXY_FW_DEVICE", "cuda"))
    p.add_argument("--compute-type", default=_env_or("LLM_PROXY_FW_COMPUTE_TYPE", "float16"))
    p.add_argument("--sample-rate", type=int, default=int(_env_or("LLM_PROXY_FW_SAMPLE_RATE", "16000")))
    p.add_argument("--language", default=_env_or("LLM_PROXY_FW_LANGUAGE"))
    p.add_argument("--token", default=_env_or("LLM_PROXY_FW_TOKEN"))
    return p.parse_args()


def build_app(args, model):
    def check_auth(request):
        if not args.token:
            return True
        header = request.headers.get("Authorization", "")
        return header.removeprefix("Bearer ").strip() == args.token

    async def handle_stt(request):
        if not check_auth(request):
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

        duration = len(audio) / args.sample_rate if args.sample_rate else 0.0
        if duration < 0.3:
            return web.json_response({"text": "", "duration": duration})

        try:
            header_prompt = decode_initial_prompt(request.headers)
        except InitialPromptEncodingError as exc:
            return web.json_response({"error": str(exc)}, status=400)
        prompt = request.query.get("prompt", "") or header_prompt or ""
        segments, info = model.transcribe(
            audio,
            language=args.language,
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
            "model_size": args.model_size,
            "language": args.language,
        })

    app = web.Application(client_max_size=64 * 1024 * 1024)
    app.router.add_post("/stt", handle_stt)
    app.router.add_get("/health", handle_health)
    return app


def main():
    args = parse_args()
    print(f"[faster-whisper] Loading model={args.model_size} device={args.device} compute={args.compute_type}")
    model = WhisperModel(args.model_size, device=args.device, compute_type=args.compute_type)
    print("[faster-whisper] Ready")
    app = build_app(args, model)
    web.run_app(app, host=args.host, port=args.port, handle_signals=True)


if __name__ == "__main__":
    main()
