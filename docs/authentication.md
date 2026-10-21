# Authentication and TLS

Inference tokens and control-panel login are separate settings. The public
example binds both listeners to loopback with authentication disabled.

## Inference tokens

Enable token checks in your config:

```yaml
auth:
  enabled: true
```

Create a client token locally using the same config and `state_dir` as the
running server:

```bash
llm-proxy -c config.yaml auth create my-client
llm-proxy -c config.yaml auth list
llm-proxy -c config.yaml auth revoke my-client
```

Copy the plain token at creation time; it is shown once. Clients send:

```text
Authorization: Bearer lp_<id>_<secret>
```

Records in `<state_dir>/auth_tokens.json` store a SHA-256 hash of each secret.
Tokens are long-lived and remain valid until revoked. Treat them as credentials;
they can also authorize control operations.

The panel's API-auth switch persists to `<state_dir>/api_auth.json` and takes
precedence over `auth.enabled` in YAML. Check the panel setting if editing the
config does not change the effective state.

## Control-panel access

The control listener is open until a web password is set. Open the panel over
loopback and set a password in its settings before exposing control access.
The password hash is persisted in `<state_dir>/web_password.json`.
Setting `auth.enabled: true` only protects inference; it does not enable web
login. The legacy `web_auth.enabled` and `web_auth.users` YAML fields do not
configure the current password mechanism.

Once a password is set, control requests require either a logged-in browser
session or a valid bearer token. CLI control commands read `LLM_PROXY_TOKEN`,
or accept `--token` explicitly:

```bash
export LLM_PROXY_TOKEN='lp_<id>_<secret>'
llm-proxy -c config.yaml status
curl http://127.0.0.1:8091/v1/control/tokens \
  -H "Authorization: Bearer $LLM_PROXY_TOKEN"
```

`POST /v1/control/tokens` with `{"name":"my-client"}` creates a token;
`DELETE /v1/control/tokens/{id-or-name}` revokes one. These endpoints follow
the same control-access rules as other control operations.

Browser sessions are held in memory and expire after `web_auth.session_ttl_hours`
(168 hours by default). Restarting the proxy clears them. Set
`web_auth.secure_cookies: true` when the panel is served through HTTPS by a
reverse proxy.

## HTTPS

The built-in TLS option applies to the inference listener. Generate a
self-signed certificate using the same config as the server:

```bash
llm-proxy -c config.yaml cert init --host inference.example.com
```

This command requires the `openssl` executable. Then enable TLS and restart:

```yaml
tls:
  enabled: true
  cert_file: ~/.config/llm-proxy/certs/server.crt
  key_file: ~/.config/llm-proxy/certs/server.key
```

Use a trusted certificate or explicitly trust the generated certificate in
clients. The CLI prints its SHA-256 fingerprint.

The control listener always uses HTTP. Keep it on loopback, use an SSH tunnel,
or terminate HTTPS at a reverse proxy. The `listen.host` and `control.host`
settings are independent; configure both explicitly when changing exposure.
