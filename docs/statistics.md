# Usage statistics

View statistics in the control panel or through
`GET /v1/control/statistics` on the control listener.

`llm-proxy` stores local operational statistics in
`<state_dir>/statistics.sqlite3`. Statistics are enabled by default and can be
disabled or relocated in `config.yaml`:

```yaml
statistics:
  enabled: true
  database: null                 # defaults to <state_dir>/statistics.sqlite3
  retention_days: 30             # 0 keeps detailed rows indefinitely
  cleanup_interval_hours: 24
  client_identity: token         # token name, or "none"
  include_host: false            # do not persist client IP/host by default
```

The database contains completed client requests and separate upstream backend
attempts. This distinction matters when failover sends one client request to
more than one backend. Byte counters describe application payload bytes, not
TCP, HTTP-header, or TLS wire overhead. Client-side and upstream byte counts
are kept separately.

For LLM responses, token counts come only from an OpenAI-compatible `usage`
object. Missing usage remains unknown; the proxy does not estimate tokens from
text length. Time to first token is represented by the first payload chunk
actually relayed to the client.

The statistics database never stores prompts, generated responses, audio
content, authorization headers, cookies, backend credentials, URL query
strings, or raw session identifiers. API token names can be retained as a
client label; set `client_identity: none` to aggregate all clients as
`anonymous`. Client network addresses are retained only when `include_host` is
explicitly enabled.

No pricing, tariff, or monetary estimates are collected: the service and its
models are locally hosted.
