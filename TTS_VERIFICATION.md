# VoxCPM2 platform verification

16 September 2026, AionShad / NVIDIA RTX 5090. Task #2412.

Installed driver and runner in the live llm-proxy checkout. Registered runtime
`voxcpm2` through the control API on port 8081, with `autoload=false` and
`on_demand=false`. Inference passed through the authenticated API on port 8080;
a temporary test token was revoked in the test's finally block. No authentication
was disabled. Existing XTTS registration remains separate.

## Results

| Request | First PCM bytes | Entire response | Audio duration |
|---|---:|---:|---:|
| «Дважды два — четыре», streaming | 0.056 s | 0.584 s | 2.08 s |
| Two-sentence time/reminder reply, streaming | 0.065 s | 2.058 s | 7.52 s |
| «Слева балка жёстко закреплена», buffered | 0.527 s | 0.527 s | 1.92 s |
| «Модель снова готова к работе», after restart | 0.053 s | 0.481 s | 1.76 s |

This is local HTTP timing through the proxy, not time to audible sound from
AtomHome speakers. The first PCM may contain leading silence. Generation was
roughly 3.6 times faster than playback. Configuration: cached reference,
CFG 2, ten diffusion steps, seed 42, native 48000 Hz, no torch.compile.

- First start including load, reference preparation and warmup: 12.03 s.
  Second start: 9.03 s (proxy log).
- Health during active synthesis: 200, busy=true, 0.0021 s.
- Concurrent synthesis: 503, no queued job.
- Stopped synthesis and health: 503, completed in about 0.003–0.004 s combined;
  no model loading was triggered.
- Successful start → synthesis → stop → start → synthesis → stop.
- Device-wide GPU memory before / loaded / stopped: 1554 / 8420 / 1588 MiB.
  The model process exited; its allocations were released. The small baseline
  difference is device-wide accounting, not evidence of retained model weights.
- Float32 little-endian, mono, 48000 Hz headers survived proxy forwarding.
  Audio payloads were non-empty, finite and non-silent; saved listening copies
  are PCM16 WAV. Human listening quality was not evaluated in this smoke.
- All 49 repository tests passed, including ten new TTS HTTP/driver tests.
  Those cover partial-response failure, responsive health, cancellation,
  unknown voice selection, input validation and stopped-model behavior.

## State after the runtime smoke

VoxCPM2 was stopped after the runtime smoke, ready for manual start. Whisper was restored after the
proxy restart, then explicitly disabled at the user's request. Its `stt-main`
runtime is stopped with autoload=false and on_demand=false. The primary voice
STT remains GigaAM on Aurora (.61); no STT model was started for this TTS smoke.
The existing local GigaAM configuration was not changed.

## AtomHome integration (later on 2026-09-16, Nemor #2413)

The AtomHome (.76) client now uses this authenticated API with native 48 kHz
PCM streaming and optional-speech fallback; `tts_enabled=true`. Only locally
submitted microphone turns are spoken, correlated through external_turn_id
and the server's acknowledged turn_id. Concurrent typed chat stays silent.
VoxCPM2 was manually started for the live output check; autoload and on_demand
remain false. Whisper remains disabled.

All 24 voice-client tests passed on .76. Full recorded-input replay reached
GigaAM, Nemor, VoxCPM2 and ALSA null, with first PCM in 73–84 ms from the TTS
request. With the runtime stopped, the availability check skipped synthesis
in 11 ms. ALSA default accepted 48 kHz float32 silence. The user subsequently confirmed
audible speech; live logs show completed playback and successful return to
wakeword for consecutive requests (22:37–22:38). See nemor-voice-input/HOME_VOICE.md for
configuration, limitations and rollback, and
`~/project/nemor-voice-input/experiments/tts/runs/atomhome-20260916/` for results.

Artifacts and repeatable smoke script:
`~/project/nemor-voice-input/experiments/tts/runs/voxcpm-proxy-20260916/`.
The smoke script is an operational test that starts/stops the runtime and
creates/revokes a temporary API token; it must not run concurrently with
another user of that runtime.

Before deployment, live config was backed up as
`~/.config/llm-proxy/config.yaml.bak-voxcpm-20260916`.
Existing manual/automatic start preferences were preserved during the proxy
restart; the subsequent Whisper-disable instruction was applied separately.
