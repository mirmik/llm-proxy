# Control API Auth

`llm-proxy` now has a separate control-plane auth flow for management operations such as model load/unload.
The intended deployment is HTTPS with trust-on-first-use fingerprint pinning, similar to SSH.

## Config

In `config.yaml`:

```yaml
tls:
  enabled: true
  cert_file: ~/.config/llm-proxy/certs/server.crt
  key_file: ~/.config/llm-proxy/certs/server.key

auth:
  enabled: true
  bootstrap_password: null
```

If `auth.enabled` is `false`, control endpoints stay open.
If `tls.enabled` is `false`, the API still works over plain HTTP, but that is only suitable for trusted local networks.

## HTTPS Certificate

Generate a self-signed certificate:

```bash
llm-proxy cert init --host llm-proxy --ip 192.168.1.10
```

Then enable `tls.enabled: true` in `config.yaml`.

## Bootstrap

Create an invite on the server:

```bash
llm-proxy auth invite my-host
```

Or store a bootstrap password hash:

```bash
llm-proxy auth set-password change-me
```

Then bootstrap a long-lived host token and pin the server fingerprint:

```bash
llm-proxy auth login \
  --server https://192.168.1.10:8090 \
  --token INVITE_TOKEN
```

Password-based bootstrap:

```bash
llm-proxy auth login \
  --server https://192.168.1.10:8090 \
  --password change-me \
  --host-id my-host
```

The login stores:

- `server`
- `host_id`
- long-lived control token
- pinned HTTPS fingerprint

in `state_dir/client_auth.json`.

Raw bootstrap by HTTP API is still available:

```bash
curl -s -X POST https://192.168.1.10:8090/v1/auth/bootstrap \
  -H 'Content-Type: application/json' \
  -d '{"token":"INVITE_TOKEN"}'
```

## Protected Requests

Control requests must send:

- `Authorization: Bearer <TOKEN>`
- `X-LLM-Proxy-Host-ID: <HOST_ID>`

Example:

```bash
curl -s -X POST http://127.0.0.1:8090/v1/models/qwen3.5-35b-a3b/load \
  -H 'Authorization: Bearer YOUR_TOKEN' \
  -H 'X-LLM-Proxy-Host-ID: my-host'
```

Validation endpoint:

```bash
curl -s http://127.0.0.1:8090/v1/control/whoami \
  -H 'Authorization: Bearer YOUR_TOKEN' \
  -H 'X-LLM-Proxy-Host-ID: my-host'
```

## Stored State

Auth files live under `state_dir`:

- `auth_invites.json`
- `auth_tokens.json`
- `auth_bootstrap.json`
- `client_auth.json`
- `certs/server.crt`
- `certs/server.key`

This is the same practical pattern as Nemor: self-signed HTTPS, TOFU fingerprint pinning, bootstrap via invite/password, then long-lived host token.
