# Runtime Architecture

`llm-proxy` is moving toward a single runtime host model.

## Concepts

- `runtime`: a managed neural runtime instance
- `driver`: a module that knows how to start and validate a runtime type
- `profile`: a named set of parameters for a runtime

## Current Status

Implemented:

- `RuntimeManager` as the unified process owner
- `llama_cpp` driver
- config section `runtimes:`
- control API for listing runtimes, selecting profiles, starting/stopping runtimes, and reading logs
- CLI control commands backed by the same control API

Legacy compatibility:

- old `models:` config is still accepted and mapped to `runtimes`

Not migrated yet:

- `faster_whisper` driver
- `xtts` driver
- dedicated in-repo runner modules for STT/TTS

## Driver Contract

Each driver provides:

- `validate_profile(profile)`
- `build_command(runtime_name, profile, global_config)`
- `build_env(runtime_name, profile, global_config)`
- `resolve_health_url(runtime_name, profile, global_config)`
- `stop_signal(runtime_name, profile)`
- `startup_timeout(runtime_name, profile)`
- `runtime_meta(runtime_name, profile, global_config)`

## Python Environments

Runtime profiles may specify either:

- `python_bin`
- `venv`

If `venv` is set, `llm-proxy` resolves the interpreter as:

```text
<venv>/bin/python
```

This allows STT, TTS, diffusion, and other runtimes to use different PyTorch builds while still being started and supervised directly by `llm-proxy`.

## CLI Control

The `llm-proxy` command can manage a running proxy through the control port:

```bash
llm-proxy status --port 8091
llm-proxy start qwen3.6-27b --port 8091
llm-proxy start qwen3.6-27b --profile large --port 8091
llm-proxy stop qwen3.6-27b --port 8091
llm-proxy restart qwen3.6-27b --port 8091
llm-proxy logs qwen3.6-27b -n 120 --port 8091
llm-proxy routes --port 8091
llm-proxy cleanup-kv --port 8091
```

If `--control-url` is omitted, the client derives it from config and uses
`control.port` or `listen.port + 1`. A control token can be passed with
`--token` or `LLM_PROXY_TOKEN`.

## Next Step

The next concrete step is to add:

- `llm_proxy/drivers/faster_whisper.py`
- `llm_proxy/drivers/xtts.py`
- in-repo runner modules so STT/TTS parameters are controlled directly by `llm-proxy`
