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

## Next Step

The next concrete step is to add:

- `llm_proxy/drivers/faster_whisper.py`
- `llm_proxy/drivers/xtts.py`
- in-repo runner modules so STT/TTS parameters are controlled directly by `llm-proxy`
