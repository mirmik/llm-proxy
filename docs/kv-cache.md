# KV cache persistence

Send a stable `X-LLM-Proxy-Session-ID` header with each chat request to reuse
the same local session. Keep sending the full conversation so the backend can
recover when live or saved state is unavailable. Requests without a session ID
use transient slots.

`llm-proxy` persists llama.cpp slot state for requests that carry a session ID.
The llama.cpp slot API writes a complete checkpoint, so saving after every
successful request can rewrite hundreds of megabytes or several gigabytes.

Checkpoint writes are throttled by default:

```yaml
kv_cache:
  checkpoint_interval_minutes: 5
  checkpoint_max_requests: 5
```

The first successful request for a new or restored session is saved
immediately. After that, a new full checkpoint is written when either limit is
reached at the end of a successful request. There is no background write for an
idle slot. Clean deferred state is always saved before slot eviction and during
a graceful runtime or proxy shutdown.

An abrupt process or machine failure can therefore leave the disk checkpoint
up to four successful requests behind with the defaults above. A client that
sends its full prompt can rebuild that tail on its next request; a client that
relies exclusively on server-side session state can lose those latest turns.

Failed or disconnected requests are never persisted over the last good disk
checkpoint. Their live state remains in the llama.cpp slot for prompt recovery;
if that slot is evicted or the process stops, the last good checkpoint is kept.

Set either trigger to `null` to disable it. With both triggers disabled,
checkpoints are written only for the initial/restored successful request and
when clean state must be flushed on eviction or shutdown. Set
`checkpoint_interval_minutes: 0` or `checkpoint_max_requests: 1` to restore
save-after-every-success behavior.

Successful llama.cpp save and restore responses include their exact token and
byte counts in the operation log as `tokens=` and `bytes=` fields.
