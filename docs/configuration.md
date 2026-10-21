# Configuration and routing

Start with [config.example.yaml](../config.example.yaml). Copy it to a local
`config.yaml` or to `~/.config/llm-proxy/config.yaml`.

`llm-proxy -c PATH serve` selects an explicit config. Without `-c`, the loader
checks the user config before `./config.yaml`, then uses built-in defaults.
The public example explicitly sets loopback addresses; built-in defaults bind
both listeners to `0.0.0.0`.

## Listeners and state

```yaml
listen:
  host: 127.0.0.1
  port: 8090
control:
  host: 127.0.0.1
  port: 8091
cache_dir: ~/.cache/llm-proxy/kv
state_dir: ~/.config/llm-proxy
```

If `control.port` is unset, it defaults to `listen.port + 1`. Inference and
control access have separate authentication rules; see
[authentication](authentication.md).

`state_dir` contains authentication settings, token records, and the default
statistics database. KV checkpoints live under `cache_dir`. Use a persistent,
writable directory for both. Control-panel configuration changes write to the
selected YAML file, so it must also be writable.

## Managed runtimes

Each `runtimes` entry needs a driver, port, and command. Commands are explicit:
the driver does not add model flags for you.

| Driver | Kind | Backend |
| --- | --- | --- |
| `llama_cpp` | `llm` | llama.cpp chat server |
| `llama_cpp_embedding` | `embedding` | llama.cpp embedding server |
| `faster_whisper` | `stt` | Bundled faster-whisper runner |
| `xtts` | `tts` | Bundled XTTS runner |
| `voxcpm` | `tts` | Bundled VoxCPM2 runner |

`autoload` starts a runtime when the proxy starts. `on_demand` permits an
inference request to start a stopped runtime. Set both explicitly; use false
for manual loading and a 503 response while stopped. `slots` must match the
backend's parallel-slot setting for llama.cpp.

Commands support `{name}`, `{port}`, `{slots}`, `{python}`, `{runners_dir}`,
`{cache_dir}`, `{state_dir}`, `{runtime_kv}`, and values such as
`{defaults.llama_bin}` from `defaults`. Commands are split into arguments,
not executed by a shell. Quote paths with spaces inside the command template.
Tilde-prefixed paths expand to the current user's home directory.

For Python runners, set `venv` or `python_bin` to select their interpreter.
The runner's packages belong in that environment, not the core proxy install.
See [runtime architecture](runtimes.md) and [speech](speech.md).

## Remote routes

Routes expose another inference service under a local name. They can be used
without any local runtimes. Each backend URL is the service root, without
`/v1` or the endpoint suffix:

```yaml
hosts:
  inference-node:
    token: "REPLACE_WITH_BACKEND_TOKEN"

routes:
  remote-chat:
    kind: llm
    health_path: /health
    check_interval: 5
    session_ttl: 3600
    backends:
      - url: https://inference-a.example.com
        model: chat
        auth: inference-node
        weight: 2
      - url: https://inference-b.example.com
        model: chat
        auth: inference-node
        weight: 1
```

Send `"model":"remote-chat"` to the local chat endpoint. `model` on a backend
rewrites the upstream model name. `auth` references a `hosts` entry whose token
is sent as a bearer credential. Config values are literal; the loader does not
expand environment variables in tokens.

New sessions are placed according to current in-flight load divided by backend
weight, with latency and backend order as tie breakers. Healthy sticky sessions
remain on their backend. Failed requests mark a backend unhealthy, and eligible
requests try remaining candidates. Background probes refresh health. Once audio
or response streaming has begun, a failed transfer cannot transparently restart
on another backend.

Route kinds are `llm`, `embedding`, `stt`, and `tts`. A backend's `/health` must
be reachable, or `health_path` must be adjusted. A healthy remote proxy does not
necessarily mean the selected model is loaded; configure remote availability
and on-demand loading accordingly.

Use the Routing tab or `llm-proxy routes` to inspect routes. The control API
also supports creating/deleting routes and editing their backend lists under
`/v1/control/routes`.

## Timeouts and persistence

`inference.request_timeout_seconds` defaults to 300. It limits connection and
read inactivity, so active streams may run longer; long prefills that emit no
data may need a larger value.

See [KV cache persistence](kv-cache.md) for checkpoint settings and
[usage statistics](statistics.md) for retention and client identity settings.
