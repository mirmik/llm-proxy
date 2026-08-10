# API Tokens

`llm-proxy` uses one mechanism for API auth: long-lived bearer tokens that you
create on the server and give to a client. Same idea as a GitHub personal
access token. `nemor-link` stores the token in its machine-managed connection
state; users do not need to write a client config.

## Config

In `config.yaml`:

```yaml
auth:
  enabled: false   # default: false (open). Set true to require a token.

web_auth:
  enabled: false   # default: false. Browser login for the control panel.
  users:
    - { username: admin, password: change-me }
```

If you sit the proxy behind a reverse proxy that already enforces auth (basic-auth, mTLS, OIDC, etc.), keep both flags `false`.

## Tokens

Created on the server, plain value shown **once** at creation time. Only `sha256(secret)` is persisted in `state_dir/auth_tokens.json`, so a leaked tokens file does not let anyone authenticate.

Token format: `lp_<id>_<secret>` — the `id` is a short opaque identifier the server uses to find the record, the `secret` is the high-entropy half.

### CLI

```bash
# Create — copy the printed token immediately, it cannot be recovered later.
llm-proxy auth create aionshad

# List — id, name, created_at, last_used_at (no secrets).
llm-proxy auth list

# Revoke by id or by name.
llm-proxy auth revoke aionshad
llm-proxy auth revoke 85e586879208
```

### HTTP API

Same operations, requires control auth (token or web session) once at least one token exists:

```bash
curl -X POST http://127.0.0.1:8080/v1/control/tokens \
  -H 'Authorization: Bearer lp_xxx' \
  -H 'Content-Type: application/json' \
  -d '{"name": "aionshad"}'

curl http://127.0.0.1:8080/v1/control/tokens \
  -H 'Authorization: Bearer lp_xxx'

curl -X DELETE http://127.0.0.1:8080/v1/control/tokens/<id-or-name> \
  -H 'Authorization: Bearer lp_xxx'
```

## Using a token

For an integrated `nemor-link` utility:

```console
commit --set-token lp_<id>_<secret>
```

The selection is shared by `commit`, `ask`, `summarize`, and other clients of
the same nemor-link state.

Clients send a single header:

```
Authorization: Bearer lp_<id>_<secret>
```

The same token works for inference (`/v1/chat/completions`, `/v1/audio/speech`, `/stt`). Model load/unload is control-plane only (`/v1/control/models/.../load`, `/v1/control/models/.../unload`).

## State files

In `state_dir`:

- `auth_tokens.json` — token records (no plain secrets, only sha256 hashes).

Legacy files from older versions (`auth_invites.json`, `auth_bootstrap.json`, `client_auth.json`) are no longer used and can be deleted. If `auth_tokens.json` was in the old `{host_id: {token, issued_at}}` format, it will be migrated on first load — but the old plain tokens are not recoverable, so you must re-create them with `llm-proxy auth create <name>`.
