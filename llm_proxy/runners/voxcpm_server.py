#!/usr/bin/env python3
"""VoxCPM2: mono float32 little-endian PCM, native sample rate in headers.

The GPU runs in one producer thread. Health stays responsive during inference;
competing requests get 503 instead of silently queuing. A disconnected consumer
cancels generation at the next chunk. Torch is imported only in the engine.
"""
import argparse
import asyncio
import concurrent.futures
import logging
import os
import threading

from aiohttp import web

LOG = logging.getLogger("voxcpm")


class VoxCPMEngine:
    def __init__(self, args):
        import torch
        # Initialize the selected CUDA runtime before optional audio libraries.
        # On the 3090 host, importing those first breaks lazy CUDA initialization.
        if args.device.startswith("cuda"):
            torch.cuda.init()
        from voxcpm import VoxCPM

        self.torch = torch
        torch.set_num_threads(args.threads)
        self.model = VoxCPM.from_pretrained(args.model_path, device=args.device,
                                           load_denoiser=False, optimize=False)
        with torch.inference_mode():
            self.cache = self.model.tts_model.build_prompt_cache(reference_wav_path=args.reference_wav)
        self.sample_rate = self.model.tts_model.sample_rate
        self.steps = args.steps
        # Warm both the generator and the incremental waveform decoder.
        for _ in self.generate("Проверка синтеза речи."):
            pass

    def generate(self, text):
        import numpy as np
        from transformers import set_seed

        set_seed(42)
        with self.torch.inference_mode():
            generator = self.model.tts_model.generate_with_prompt_cache_streaming(
                target_text=text, prompt_cache=self.cache, cfg_value=2.0,
                inference_timesteps=self.steps, max_len=1400, retry_badcase=False)
            try:
                for audio, _, _ in generator:
                    pcm = audio.squeeze(0).cpu().numpy().astype('<f4', copy=False)
                    if pcm.ndim != 1 or not np.isfinite(pcm).all():
                        raise RuntimeError("Model returned invalid audio")
                    if pcm.size:
                        yield pcm.tobytes()
            finally:
                generator.close()


def build_app(args, engine):
    busy = threading.Lock()
    jobs = set()
    cancellations = set()

    async def health(request):
        return web.json_response(dict(status="ok", kind="tts", model="VoxCPM2",
                                      busy=busy.locked(), sample_rate=engine.sample_rate,
                                      channels=1, format="f32le", streaming=True))

    async def synthesize(request):
        try:
            body = await request.json()
        except (ValueError, UnicodeDecodeError):
            return web.json_response({"error": "Expected JSON object"}, status=400)
        if not isinstance(body, dict):
            return web.json_response({"error": "Expected JSON object"}, status=400)
        if set(body) - {"text", "stream", "language"}:
            return web.json_response({"error": "Supported fields: text, stream, language"}, status=400)
        text = body.get("text")
        if not isinstance(text, str) or not text.strip():
            return web.json_response({"error": "Non-empty text is required"}, status=400)
        text = " ".join(text.split())
        if len(text) > args.max_text_length:
            return web.json_response({"error": f"Text exceeds {args.max_text_length} characters; split into sentences"}, status=400)
        streaming = body.get("stream", False)
        if not isinstance(streaming, bool):
            return web.json_response({"error": "stream must be boolean"}, status=400)
        # VoxCPM2 derives pronunciation/language from text, not a language token.
        if not isinstance(body.get("language", "ru"), str):
            return web.json_response({"error": "language must be a string"}, status=400)
        if not busy.acquire(blocking=False):
            return web.json_response({"error": "TTS is busy"}, status=503, headers={"Retry-After": "1"})

        loop = asyncio.get_running_loop()
        queue = asyncio.Queue(maxsize=4)
        stop = threading.Event()
        cancellations.add(stop)

        def put(item):
            if stop.is_set():
                return False
            pending = asyncio.run_coroutine_threadsafe(queue.put(item), loop)
            while True:
                try:
                    pending.result(timeout=.1)
                    return True
                except concurrent.futures.TimeoutError:
                    if stop.is_set():
                        pending.cancel()
                        return False

        def produce():
            generator = None
            try:
                generator = engine.generate(text)
                for chunk in generator:
                    if stop.is_set() or not put(chunk):
                        break
                put(None)
            except Exception as exc:
                LOG.exception("Synthesis failed")
                put(exc)
            finally:
                try:
                    if generator is not None:
                        generator.close()
                finally:
                    busy.release()

        job = asyncio.create_task(asyncio.to_thread(produce))
        jobs.add(job)
        job.add_done_callback(jobs.discard)
        response = None
        headers = {"X-Audio-Sample-Rate": str(engine.sample_rate),
                   "X-Audio-Channels": "1", "X-Audio-Format": "f32le"}
        try:
            pieces = []
            while True:
                chunk = await queue.get()
                if isinstance(chunk, Exception):
                    raise chunk
                if chunk is None:
                    break
                if streaming:
                    if response is None:
                        response = web.StreamResponse(headers=headers)
                        response.content_type = "application/octet-stream"
                        await response.prepare(request)
                    await response.write(chunk)
                else:
                    pieces.append(chunk)
            if streaming:
                if response is None:
                    raise RuntimeError("Model returned no audio")
                await response.write_eof()
                return response
            if not pieces:
                raise RuntimeError("Model returned no audio")
            return web.Response(body=b"".join(pieces), headers=headers,
                                content_type="application/octet-stream")
        except asyncio.CancelledError:
            raise
        except Exception:
            if response is not None and response.prepared:
                # Never report a truncated waveform as a successful complete body.
                if request.transport:
                    request.transport.close()
                return response
            return web.json_response({"error": "TTS synthesis failed; see runtime logs"}, status=502)
        finally:
            stop.set()
            # Keep the GPU reserved until the producer actually leaves inference.
            try:
                await asyncio.shield(job)
            finally:
                cancellations.discard(stop)

    async def cleanup(app):
        for stop in cancellations:
            stop.set()
        if jobs:
            await asyncio.gather(*list(jobs), return_exceptions=True)

    app = web.Application(client_max_size=64 * 1024)
    app.router.add_get("/health", health)
    app.router.add_post("/tts/generate", synthesize)
    app.on_cleanup.append(cleanup)
    return app


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=5052)
    p.add_argument('--model-path', required=True)
    p.add_argument('--reference-wav', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--steps', type=int, default=10)
    p.add_argument('--max-text-length', type=int, default=500)
    args = p.parse_args()
    for field in ['model_path', 'reference_wav']:
        setattr(args, field, os.path.expanduser(getattr(args, field)))
    if min(args.steps, args.threads, args.max_text_length) <= 0:
        p.error('steps, threads and max-text-length must be positive')
    return args


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    engine = VoxCPMEngine(args)
    LOG.info("Ready: %s Hz mono f32le", engine.sample_rate)
    web.run_app(build_app(args, engine), host=args.host, port=args.port,
                handler_cancellation=True, shutdown_timeout=10)


if __name__ == '__main__':
    main()
