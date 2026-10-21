# Runtime Architecture

`RuntimeManager` owns managed inference processes. Drivers describe how to
validate, launch, and check each type of runtime.

## Concepts

- `runtime`: a managed neural runtime instance
- `driver`: a module that knows how to start and validate a runtime type

## Current Status

Implemented:

- `RuntimeManager` as the unified process owner
- `llama_cpp`, `faster_whisper`, `xtts`, and `voxcpm` drivers
- `llama_cpp_embedding` driver and OpenAI-compatible `/v1/embeddings`
- config section `runtimes:`
- control API for listing runtimes, starting/stopping runtimes, and reading logs
- CLI control commands backed by the same control API

- dedicated in-repo runner modules for STT/TTS
- manual or on-demand loading, independently configured per runtime
- TTS PCM forwarding, including streaming VoxCPM2 output

Embedding runtime configuration and API behavior are documented in
[Embedding runtimes](embeddings.md).

## Driver Contract

Each driver provides:

- `validate_runtime(runtime)`
- `build_command(runtime_name, runtime, global_config)`
- `build_env(runtime_name, runtime, global_config)`
- `resolve_health_url(runtime_name, runtime, global_config)`
- `stop_signal(runtime_name, runtime)`
- `startup_timeout(runtime_name, runtime)`
- `runtime_meta(runtime_name, runtime, global_config)`

## Python Environments

Runtimes may specify either:

- `python_bin`
- `venv`

If `venv` is set, `llm-proxy` resolves the interpreter as:

```text
<venv>/bin/python
```

This allows speech runtimes to use different PyTorch builds while still being
started and supervised directly by `llm-proxy`. Install each runner's
dependencies in its selected environment. When using the faster-whisper
runner, also install this package into that environment (`python -m pip
install -e /path/to/llm-proxy`): it imports the shared STT protocol module.

## CLI Control

The `llm-proxy` command can manage a running proxy through the control port:

```bash
llm-proxy status --port 8091
llm-proxy start chat --port 8091
llm-proxy stop chat --port 8091
llm-proxy restart chat --port 8091
llm-proxy logs chat -n 120 --port 8091
llm-proxy routes --port 8091
llm-proxy cleanup-kv --port 8091
```

If `--control-url` is omitted, the client derives it from config and uses
`control.port` or `listen.port + 1`. A control token can be passed with
`--token` or `LLM_PROXY_TOKEN`.

## Legacy Config Compatibility

The loader still understands the old `profiles/default` shape and collapses it
into the flat runtime format in memory. New writes use the flat format:

```yaml
runtimes:
  qwen:
    driver: llama_cpp
    gpu: local
    autoload: false
    port: 8082
    slots: 1
    command: >
      {defaults.llama_bin}
      --model ~/models/qwen.gguf
      --host 127.0.0.1 --port {port}
```

## TTS runtimes

XTTS and VoxCPM2 are managed processes with their own Python environments.
Use `autoload: false` and `on_demand: false` for models that should only occupy
GPU memory when explicitly started. A stopped TTS returns 503 without loading.
`voice` selects the runtime for `/v1/audio/speech`; `/tts/health?voice=NAME`
checks that specific runtime. Unknown names return 404, not a different voice.

See [speech runtimes](speech.md) for setup, PCM formats, and streaming.
The proxy forwards TTS bytes; synthesis behavior depends on the runner. XTTS
returns a completed waveform, while VoxCPM2 can return incremental PCM.
