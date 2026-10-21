# Speech runtimes

Speech recognition and synthesis run in separate processes and Python
environments. The proxy forwards their PCM protocol; the main service does not
import Torch or model packages.

These are custom interfaces. `/stt` does not accept OpenAI multipart audio
uploads, and `/v1/audio/speech` uses `text` and `voice`, rather than the OpenAI
`input` and `model` schema.

## Speech recognition

Install `faster-whisper`, `numpy`, and this project in a dedicated environment,
with the GPU libraries required by the selected backend. Add a runtime:

```yaml
runtimes:
  stt-main:
    driver: faster_whisper
    gpu: local
    autoload: false
    on_demand: false
    port: 5055
    venv: ~/.venvs/whisper
    command: >-
      {python} {runners_dir}/faster_whisper_server.py
      --host 127.0.0.1 --port {port}
      --model-size large-v3 --device cuda --compute-type float16
      --sample-rate 16000
```

The runner may download model weights on first use. Start it explicitly, then
send raw mono float32 little-endian PCM at the configured sample rate:

```bash
llm-proxy -c config.yaml start stt-main
curl http://127.0.0.1:8090/stt \
  -H 'Content-Type: application/octet-stream' \
  --data-binary @speech.f32le
```

The response is JSON containing `text`, `language`, and `duration`.
`GET /stt/health` checks availability. Select a specific runtime with the
`X-STT-Runtime` header. The optional `X-Initial-Prompt-Encoded` header carries
`v1:` followed by base64url-encoded UTF-8 prompt text; plain ASCII prompts may
use `X-Initial-Prompt`.

## VoxCPM2 synthesis

Prepare a separate environment with `voxcpm`, its compatible Torch stack, and
`aiohttp`. Download the model and prepare a reference WAV before launching.
Then configure the local model directory and voice sample:

```yaml
runtimes:
  voxcpm2:
    driver: voxcpm
    gpu: local
    autoload: false
    on_demand: false
    port: 5052
    startup_timeout: 240
    venv: ~/.venvs/voxcpm
    command: >-
      {python} {runners_dir}/voxcpm_server.py
      --host 127.0.0.1 --port {port}
      --model-path ~/models/VoxCPM2
      --reference-wav ~/voices/reference.wav
      --device cuda --steps 10 --threads 4
    env:
      HF_HUB_OFFLINE: "1"
      TOKENIZERS_PARALLELISM: "false"
      PYTHONUNBUFFERED: "1"
```

Start it with `llm-proxy -c config.yaml start voxcpm2`. Read logs with
`llm-proxy -c config.yaml logs voxcpm2`. The runner becomes healthy after model
loading, reference preparation, and streaming warmup. The configured offline
mode requires all weights to be available locally.

```bash
curl http://127.0.0.1:8090/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"voice":"voxcpm2","text":"Hello from a local model.","stream":true}' \
  --output speech.f32le
```

- `voice` selects the runtime; when omitted, the default name is `tts-main`.
- `text` must be non-empty and at most 500 characters after whitespace
  normalization with the runner's default limit. Split longer text in the client.
- `stream` is a boolean, defaulting to false. Streaming sends audio as generated;
  buffered mode collects the same synthesis before responding.
- Optional `language` is accepted, but VoxCPM2 derives language from text.
- Other fields are rejected by the VoxCPM2 runner.

VoxCPM2 returns raw mono little-endian float32 PCM at its native 48000 Hz:

```text
Content-Type: application/octet-stream
X-Audio-Sample-Rate: 48000
X-Audio-Channels: 1
X-Audio-Format: f32le
```

Always use the response sample-rate header. HTTP chunks can split four-byte
samples, so retain partial samples between reads. The payload is not WAV or MP3.

`GET /tts/health?voice=voxcpm2` checks that runtime. A stopped runtime returns
503 without loading it when `on_demand` is false; an unknown voice returns 404.
During synthesis health remains responsive with `busy: true`. One synthesis
runs at a time; concurrent requests receive 503 with `Retry-After`. No backlog
is retained. Disconnects cancel generation at the next model chunk. Failure
before audio returns 502; failure after streaming begins truncates the transfer.

## XTTS synthesis

XTTS uses its own environment with `TTS`, `torch`, `torchaudio`, `numpy`,
`soundfile`, and `aiohttp`, with versions compatible with the selected model.
Add an `xtts` runtime pointing to `xtts_server.py`:

```yaml
runtimes:
  tts-main:
    driver: xtts
    gpu: local
    autoload: false
    on_demand: false
    port: 5050
    venv: ~/.venvs/xtts
    command: >-
      {python} {runners_dir}/xtts_server.py
      --host 127.0.0.1 --port {port}
      --model-name tts_models/multilingual/multi-dataset/xtts_v2
      --reference-wav ~/voices/reference.wav
      --language en --device cuda --sample-rate 24000
```

Review and accept the model's license and terms before use. XTTS returns a
completed waveform as mono float32 PCM (24000 Hz in this example), rather than
incremental generation. Use the response audio headers for playback.

## Verification

`python -m unittest discover -s tests` includes CPU-only HTTP tests for runtime
selection, stopped models, streaming, PCM headers, busy rejection, health,
validation, cancellation, and partial failures. Live synthesis requires the
selected model environment and suitable hardware; unit tests do not measure
audio quality or GPU performance.
