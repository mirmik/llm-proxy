#!/usr/bin/env python3
"""Managed XTTS HTTP server for llm-proxy."""

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

HOST = "127.0.0.1"
PORT = int(os.environ.get("LLM_PROXY_XTTS_PORT", "5050"))
DEVICE = os.environ.get("LLM_PROXY_XTTS_DEVICE", "cuda")
MODEL_NAME = os.environ.get("LLM_PROXY_XTTS_MODEL_NAME", "tts_models/multilingual/multi-dataset/xtts_v2")
REFERENCE_WAV = os.path.expanduser(os.environ["LLM_PROXY_XTTS_REFERENCE_WAV"])
DEFAULT_LANGUAGE = os.environ.get("LLM_PROXY_XTTS_LANGUAGE", "ru")
SAMPLE_RATE = int(os.environ.get("LLM_PROXY_XTTS_SAMPLE_RATE", "24000"))

print(f"[xtts] Loading {MODEL_NAME} on {DEVICE}")
TTS_API = TTS(MODEL_NAME).to(DEVICE)
XTTS_MODEL = TTS_API.synthesizer.tts_model
print(f"[xtts] Computing conditioning from {REFERENCE_WAV}")
GPT_COND_LATENT, SPEAKER_EMBEDDING = XTTS_MODEL.get_conditioning_latents(audio_path=[REFERENCE_WAV])
print("[xtts] Ready")


def synthesize_cached(text, language, speed=1.0, temperature=0.75):
    out = XTTS_MODEL.inference(
        text=text,
        language=language,
        gpt_cond_latent=GPT_COND_LATENT,
        speaker_embedding=SPEAKER_EMBEDDING,
        speed=speed,
        temperature=temperature,
    )
    wav = out["wav"]
    if hasattr(wav, "cpu"):
        return wav.cpu().numpy().squeeze().astype(np.float32)
    return np.asarray(wav, dtype=np.float32).squeeze()


async def handle_tts(request):
    data = await request.json()
    text = data.get("text")
    if not text:
        return web.json_response({"error": "missing 'text' field"}, status=400)
    language = data.get("language", DEFAULT_LANGUAGE)
    speed = float(data.get("speed", 1.0))
    temperature = float(data.get("temperature", 0.75))
    audio_np = synthesize_cached(text=text, language=language, speed=speed, temperature=temperature)
    return web.json_response({
        "status": "ok",
        "samples": int(len(audio_np)),
        "sample_rate": SAMPLE_RATE,
    })


async def handle_tts_generate(request):
    data = await request.json()
    text = data.get("text")
    if not text:
        return web.json_response({"error": "missing 'text' field"}, status=400)
    language = data.get("language", DEFAULT_LANGUAGE)
    speed = float(data.get("speed", 1.0))
    temperature = float(data.get("temperature", 0.75))
    audio_np = synthesize_cached(text=text, language=language, speed=speed, temperature=temperature)
    return web.Response(body=audio_np.tobytes(), content_type="application/octet-stream")


async def handle_health(request):
    return web.json_response({
        "status": "ok",
        "kind": "tts",
        "model_name": MODEL_NAME,
        "reference_wav": REFERENCE_WAV,
        "language": DEFAULT_LANGUAGE,
    })


def main():
    app = web.Application(client_max_size=16 * 1024 * 1024)
    app.router.add_post("/tts", handle_tts)
    app.router.add_post("/tts/generate", handle_tts_generate)
    app.router.add_get("/health", handle_health)
    web.run_app(app, host=HOST, port=PORT, handle_signals=True)


if __name__ == "__main__":
    main()
