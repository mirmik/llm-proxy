#!/usr/bin/env python3
"""Managed XTTS HTTP server for llm-proxy.

Parameters are taken from CLI flags first, env vars as fallback.
CLI flags (each has a matching LLM_PROXY_XTTS_* env var):
    --host            (LLM_PROXY_XTTS_HOST,           default 0.0.0.0)
    --port            (LLM_PROXY_XTTS_PORT,           default 5050)
    --device          (LLM_PROXY_XTTS_DEVICE,         default cuda)
    --model-name      (LLM_PROXY_XTTS_MODEL_NAME,     default xtts_v2)
    --reference-wav   (LLM_PROXY_XTTS_REFERENCE_WAV,  required)
    --language        (LLM_PROXY_XTTS_LANGUAGE,       default ru)
    --sample-rate     (LLM_PROXY_XTTS_SAMPLE_RATE,    default 24000)
Note: Coqui TOS is accepted via COQUI_TOS_AGREED env var (as before).
"""

import argparse
import os

import numpy as np
import soundfile as sf
import torch
import torchaudio
from aiohttp import web
from TTS.api import TTS


_original_load = torch.load
torch.load = lambda *args, **kwargs: _original_load(*args, **{**kwargs, "weights_only": False})


def _soundfile_load(path, frame_offset=0, num_frames=-1, normalize=True, channels_first=True, format=None, buffer_size=4096):
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if frame_offset:
        audio = audio[frame_offset:]
    if num_frames is not None and num_frames >= 0:
        audio = audio[:num_frames]
    tensor = torch.from_numpy(audio.T if channels_first else audio)
    return tensor, sample_rate


torchaudio.load = _soundfile_load


DEFAULT_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"


def _env_or(name, default=None):
    value = os.environ.get(name)
    return value if value is not None and value != "" else default


def parse_args():
    p = argparse.ArgumentParser(description="XTTS HTTP server")
    p.add_argument("--host", default=_env_or("LLM_PROXY_XTTS_HOST", "0.0.0.0"))
    p.add_argument("--port", type=int, default=int(_env_or("LLM_PROXY_XTTS_PORT", "5050")))
    p.add_argument("--device", default=_env_or("LLM_PROXY_XTTS_DEVICE", "cuda"))
    p.add_argument("--model-name", default=_env_or("LLM_PROXY_XTTS_MODEL_NAME", DEFAULT_MODEL))
    p.add_argument("--reference-wav", default=_env_or("LLM_PROXY_XTTS_REFERENCE_WAV"))
    p.add_argument("--language", default=_env_or("LLM_PROXY_XTTS_LANGUAGE", "ru"))
    p.add_argument("--sample-rate", type=int, default=int(_env_or("LLM_PROXY_XTTS_SAMPLE_RATE", "24000")))
    args = p.parse_args()
    if not args.reference_wav:
        p.error("--reference-wav is required (or set LLM_PROXY_XTTS_REFERENCE_WAV)")
    args.reference_wav = os.path.expanduser(args.reference_wav)
    return args


def synthesize_factory(xtts_model, gpt_cond_latent, speaker_embedding):
    def synthesize(text, language, speed=1.0, temperature=0.75):
        out = xtts_model.inference(
            text=text,
            language=language,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            speed=speed,
            temperature=temperature,
        )
        wav = out["wav"]
        if hasattr(wav, "cpu"):
            return wav.cpu().numpy().squeeze().astype(np.float32)
        return np.asarray(wav, dtype=np.float32).squeeze()

    return synthesize


def build_app(args, synthesize):
    async def handle_tts(request):
        data = await request.json()
        text = data.get("text")
        if not text:
            return web.json_response({"error": "missing 'text' field"}, status=400)
        language = data.get("language", args.language)
        speed = float(data.get("speed", 1.0))
        temperature = float(data.get("temperature", 0.75))
        audio_np = synthesize(text=text, language=language, speed=speed, temperature=temperature)
        return web.json_response({
            "status": "ok",
            "samples": int(len(audio_np)),
            "sample_rate": args.sample_rate,
        })

    async def handle_tts_generate(request):
        data = await request.json()
        text = data.get("text")
        if not text:
            return web.json_response({"error": "missing 'text' field"}, status=400)
        language = data.get("language", args.language)
        speed = float(data.get("speed", 1.0))
        temperature = float(data.get("temperature", 0.75))
        audio_np = synthesize(text=text, language=language, speed=speed, temperature=temperature)
        return web.Response(body=audio_np.tobytes(), content_type="application/octet-stream")

    async def handle_health(request):
        return web.json_response({
            "status": "ok",
            "kind": "tts",
            "model_name": args.model_name,
            "reference_wav": args.reference_wav,
            "language": args.language,
        })

    app = web.Application(client_max_size=16 * 1024 * 1024)
    app.router.add_post("/tts", handle_tts)
    app.router.add_post("/tts/generate", handle_tts_generate)
    app.router.add_get("/health", handle_health)
    return app


def main():
    args = parse_args()
    print(f"[xtts] Loading {args.model_name} on {args.device}")
    tts_api = TTS(args.model_name).to(args.device)
    xtts_model = tts_api.synthesizer.tts_model
    print(f"[xtts] Computing conditioning from {args.reference_wav}")
    gpt_cond_latent, speaker_embedding = xtts_model.get_conditioning_latents(audio_path=[args.reference_wav])
    print("[xtts] Ready")
    synthesize = synthesize_factory(xtts_model, gpt_cond_latent, speaker_embedding)
    app = build_app(args, synthesize)
    web.run_app(app, host=args.host, port=args.port, handle_signals=True)


if __name__ == "__main__":
    main()
