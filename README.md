# llm-proxy

[![Tests](https://github.com/mirmik/llm-proxy/actions/workflows/tests.yml/badge.svg)](https://github.com/mirmik/llm-proxy/actions/workflows/tests.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![MIT License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A self-hosted inference gateway for local models and remote backends. Run
multiple models behind one API, manage their processes, and keep chat sessions
attached to their backend and llama.cpp KV cache.

The proxy itself needs only Python, aiohttp, and PyYAML. Model servers run as
separate processes, with their own binaries or Python environments.

## Features

- OpenAI-compatible chat completions with streaming, and dense embeddings.
- Managed llama.cpp, faster-whisper, XTTS, and VoxCPM2 runtimes.
- Manual, startup, or on-demand model loading.
- Routing to remote backends with health checks, weighted load distribution,
  sticky sessions, and failover.
- Exclusive llama.cpp slot leases and persistent session KV checkpoints.
- A browser control panel and CLI for runtime management, routing, logs, and
  usage statistics.
- Bearer tokens, optional HTTPS for inference, and a separate control listener.

This is an evolving Linux-oriented project. Chat and embeddings use
OpenAI-compatible endpoints; speech uses a custom PCM protocol. It does not
implement the full OpenAI API. Model weights, inference binaries, and GPU
frameworks are installed separately.

## Quick start

Requirements: Python 3.10+ on Linux, and a working `llama-server` binary plus a
GGUF model for the chat example.

```bash
git clone https://github.com/mirmik/llm-proxy.git
cd llm-proxy
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
cp config.example.yaml config.yaml
```

Edit `config.yaml`: set `defaults.llama_bin` to your `llama-server` binary and
`defaults.chat_model` to your GGUF file. Adjust the context size and GPU options
for your hardware. Then start the proxy:

```bash
llm-proxy -c config.yaml serve
```

Open the control panel at **http://127.0.0.1:8091/**. In another terminal with
the virtual environment activated, start the example runtime:

```bash
llm-proxy -c config.yaml start chat
curl http://127.0.0.1:8090/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"chat","messages":[{"role":"user","content":"Hello!"}],"stream":true}'
```

The example binds both listeners to loopback and starts no model automatically.
You can explore the panel without installing a model server. Set `autoload:
true` to load a runtime at startup, or `on_demand: true` to load it on its first
inference request. Both are disabled in the example.

## Configuration

Pass `-c PATH` before the subcommand to select a config explicitly. Otherwise,
the proxy reads `~/.config/llm-proxy/config.yaml`, then `./config.yaml`, falling
back to built-in defaults if neither exists. Keep your deployment config local;
`config.yaml` is ignored by git.

The [example config](config.example.yaml) defines a chat runtime and an optional
embedding runtime. You can also use routes without managing any local process.
See [configuration and routing](docs/configuration.md) for details.

## API

The example exposes inference on port **8090** and control on **8091**.

| Method | Inference endpoint | Purpose |
| --- | --- | --- |
| `POST` | `/v1/chat/completions` | Chat completions, including SSE streaming |
| `POST` | `/v1/embeddings` | Dense embeddings |
| `POST` | `/tokenize` | Backend tokenization; requires backend support |
| `GET` | `/v1/models` | Chat model discovery |
| `GET` | `/v1/models/{name}/health` | Readiness of one exact local runtime |
| `GET` | `/v1/capabilities` | Service and protocol discovery |
| `GET` | `/health` | Public proxy liveness, without model details |
| `POST` | `/stt` | Speech recognition from raw PCM |
| `POST` | `/v1/audio/speech` | Speech synthesis using the custom PCM contract |

When API authentication is enabled, inference, `/v1/models`, and per-model
health require a bearer token. `/health` and `/v1/capabilities` remain public.
See the [readiness and routing contract](docs/configuration.md#model-readiness).

For persistent local chat sessions, send an `X-LLM-Proxy-Session-ID` header.
See [KV cache persistence](docs/kv-cache.md) for checkpoint behavior and recovery
limits. Speech details are in [speech runtimes](docs/speech.md).

## Runtime management

```bash
llm-proxy -c config.yaml status
llm-proxy -c config.yaml start chat
llm-proxy -c config.yaml stop chat
llm-proxy -c config.yaml restart chat
llm-proxy -c config.yaml logs chat -n 120
llm-proxy -c config.yaml routes
llm-proxy -c config.yaml cleanup-kv
```

Use `--control-url` to target a different control listener and `LLM_PROXY_TOKEN`
to supply its bearer token. The panel provides the same runtime controls,
configuration forms, and local usage statistics.

For a Linux user service, run `./install.sh`. It installs into a dedicated
virtual environment, preserves an existing user config, and enables the
[systemd unit](deploy/systemd/llm-proxy.service). Edit
`~/.config/llm-proxy/config.yaml` before starting it with
`systemctl --user start llm-proxy`. See [deployment](docs/deployment.md).

## Authentication

The example disables authentication for local use. Before exposing the service,
configure [API tokens and control-panel access](docs/authentication.md).
Inference authentication and control access are separate: enabling API tokens
does not by itself protect the control panel. Built-in TLS covers only the
inference listener; the control listener uses HTTP.

## Documentation

- [Configuration and routing](docs/configuration.md)
- [Runtime architecture and drivers](docs/runtimes.md)
- [Authentication and TLS](docs/authentication.md)
- [Embedding runtimes](docs/embeddings.md)
- [Speech runtimes and PCM protocols](docs/speech.md)
- [KV cache persistence](docs/kv-cache.md)
- [Usage statistics](docs/statistics.md)
- [Deployment](docs/deployment.md)

## Development

```bash
python -m pip install -e .
python -m unittest discover -s tests
```

Tests use temporary local HTTP servers and mocked model engines; no GPU or
model weights are needed. See [CONTRIBUTING.md](CONTRIBUTING.md) for reporting
issues and submitting changes.

## License

[MIT](LICENSE). Model weights and third-party runtimes have their own licenses.
