# Managed speech synthesis

`voxcpm2` runs on AionShad / RTX 5090 (192.168.0.90). It is optional:
`autoload: false`, `on_demand: false`. Start and stop it through the existing
control panel or CLI; inference requests never load this model automatically.
XTTS remains a separate `tts-main` runtime.

## Commands on AionShad

```bash
llm-proxy start voxcpm2 --port 8081
llm-proxy stop voxcpm2 --port 8081
llm-proxy logs voxcpm2 --port 8081
llm-proxy status --port 8081
```

Panel: http://192.168.0.90:8081/

The runner binds to loopback. Clients use the proxy's authenticated inference
port, not the runner port. If the proxy or model is unavailable, a voice-agent
client should skip speech for that answer and keep the text. It should not
queue old answers for playback when the model returns. That client integration
is separate from this runtime addition.

## API and PCM contract

`POST /v1/audio/speech`:

```json
{"voice":"voxcpm2","text":"Дважды два — четыре.","stream":true}
```

- `voice` selects the runtime; omitting it retains the existing `tts-main` default.
- `text`: non-empty string, up to 500 characters after whitespace normalization.
  Longer answers should be split into sentences by the client.
- `stream`: boolean, default false. True sends audio as it is generated; false
  buffers the same incremental synthesis and returns the complete PCM.
- Optional `language`: string, accepted for the existing TTS interface;
  VoxCPM2 derives language from the text itself.
- Other fields are rejected with 400. XTTS speed/temperature/time-stretch
  settings are not VoxCPM2 controls.

The response is **raw mono little-endian float32 PCM at 48000 Hz**, not WAV,
MP3, or the 24000 Hz PCM produced by XTTS. Headers:

```text
Content-Type: application/octet-stream
X-Audio-Sample-Rate: 48000
X-Audio-Channels: 1
X-Audio-Format: f32le
```

HTTP chunks are arbitrary byte boundaries; clients must carry incomplete
four-byte samples across reads. Use the response sample-rate header for playback.
The API uses `text` and a runtime selector, not the OpenAI `input`/`model` schema.

`GET /tts/health?voice=voxcpm2` checks this runtime. It returns 503 when stopped,
200 with `busy: true` during synthesis. Health never waits for GPU generation.
One generation runs at a time; competing requests receive 503 with Retry-After.
No request backlog is retained. A disconnect cancels at the next model chunk;
the GPU lock stays held until the generator has actually stopped. A failure
before audio produces 502; a mid-stream failure closes the transfer, so a
truncated response is not reported as successfully complete.

## Configuration and environment

The main proxy does not import Torch or VoxCPM. The runner uses the existing
Grim environment: Python 3.12.7, voxcpm 2.0.3, torch 2.10.0, transformers 4.57.3,
aiohttp 3.14.3. Its pinned package list is
`~/project/grim/experiments/voxcpm2/requirements.lock.txt`.

```yaml
runtimes:
  voxcpm2:
    driver: voxcpm
    gpu: local
    autoload: false
    on_demand: false
    port: 5052
    startup_timeout: 240
    venv: ~/project/grim/experiments/voxcpm2/.venv
    command: >-
      {python} {runners_dir}/voxcpm_server.py
      --host 127.0.0.1 --port {port}
      --model-path ~/project/grim/experiments/voxcpm2/models/hub/models--openbmb--VoxCPM2/snapshots/32279effe8c19989596f05d353d1447f51d9e915
      --reference-wav ~/project/for-tts-xtts/reference.wav
      --device cuda --steps 10 --threads 4
    env:
      HF_HUB_OFFLINE: "1"
      TOKENIZERS_PARALLELISM: "false"
      PYTHONUNBUFFERED: "1"
```

No downloads occur during startup. The reference voice is prepared once and
reused. CFG=2, ten diffusion steps, seed=42 per request; no torch.compile,
denoising, text normalization model, or automatic badcase retries. Health
becomes available only after model loading, voice preparation, and a short
streaming warmup. Stopping the runtime exits its process and releases its GPU
allocations. Existing STT/LLM runtimes are not evicted to make room; insufficient
memory surfaces as a startup/inference error.

## Tests

`python -m unittest discover -s tests` includes CPU-only HTTP tests for selection,
stopped models without automatic loading, streaming before completion, PCM
headers, busy rejection, responsive health, invalid requests, cancellation,
and failures before/after the first chunk. Platform measurements are recorded
in `TTS_VERIFICATION.md` after the actual GPU smoke test.
