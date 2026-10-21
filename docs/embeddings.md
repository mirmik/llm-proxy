# Embedding runtimes

`llm-proxy` exposes local dense embedding models through the OpenAI-compatible
endpoint `POST /v1/embeddings`. Embedding processes have their own
`embedding` runtime kind and never use chat slots, session markers, KV-cache,
or chat model discovery.

## Runtime configuration

Use the dedicated `llama_cpp_embedding` driver. The command is explicit: the
driver does not add model flags implicitly.

```yaml
runtimes:
  bge-m3:
    driver: llama_cpp_embedding
    gpu: local
    autoload: false
    on_demand: true
    port: 8084
    slots: 2
    command: >
      {defaults.llama_bin}
      --model ~/models/bge-m3-Q8_0.gguf
      --host 127.0.0.1 --port {port}
      --embedding --pooling cls --parallel {slots}
      -c 16384 -b 2048 -ub 2048
```

BGE-M3 dense embeddings use CLS pooling. Do not substitute mean pooling unless
the selected model explicitly requires it. `llama-server` L2-normalizes pooled
vectors; the proxy forwards them without further transformation.

`llama-server` divides the configured context among parallel slots. The example
uses `-c 16384` so each of its two slots retains BGE-M3's 8192-token capacity.
On current llama.cpp builds device placement may be automatic; verify reported
RAM/VRAM use and coexistence with generation runtimes on the target machine.

The tested artifact is `lm-kit/bge-m3-gguf` revision
`9379ce497e8814b200f2dc0d18eb4045426dcb8c`, file `bge-m3-Q8_0.gguf`:

```text
size:   634553856 bytes
sha256: c98c7c907bb3ab8c3be4c8ad827cc934c7c1a7e014d1c999e0c45c8027c158bb
```

## Request

```bash
curl http://127.0.0.1:8090/v1/embeddings \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer <token>' \
  -d '{"model":"bge-m3","input":["первый текст","second text"]}'
```

`input` is forwarded unchanged. Validation of input types, token limits, and
batch limits is left to the backend. Model resolution only considers embedding
runtimes and routes; chat, STT, and TTS runtimes cannot be selected through
this endpoint.

An embedding backend can be exposed through the routing layer instead of being
managed as a local runtime:

```yaml
routes:
  bge-m3:
    kind: embedding
    backends:
      - url: https://embedding-host:8080
        model: bge-m3
        auth: embedding-host
```

Routed embedding requests use the same health monitoring, authentication,
TLS, statistics, and failover machinery as routed chat requests.

Error statuses are:

- `400` for malformed JSON, missing model, or an ambiguous partial name;
- `404` for an unknown embedding model;
- `503` for a stopped runtime or an on-demand startup failure;
- `502` when the local backend cannot be reached or does not return JSON.

Successful and error responses from a reachable backend retain its HTTP
status. Statistics contain kind, model/runtime names, status, timings, byte
counts, and standard usage counters, but never input text or vectors.

From the repository root, run the repeatable multilingual check with:

```bash
LLM_PROXY_TOKEN='<token>' python scripts/embedding_quality_smoke.py
```

The token may be omitted when API authentication is disabled. The script
checks response shape, vector order and dimension, finite values, normalization,
and relative Russian/English ranking without relying on a brittle absolute
similarity threshold.

## Index compatibility

An embedding index must record the exact GGUF checksum, model identifier,
vector dimension, pooling mode, normalization, and preprocessing convention.
Changing any of these requires rebuilding the index.
