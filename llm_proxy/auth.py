"""Server-side auth helpers for llm-proxy.

Tokens look like `lp_<id>_<secret>` where:
  - `id` is a short opaque identifier used to locate the token record;
  - `secret` is the high-entropy half. Only `sha256(secret)` is persisted, so
    a leaked `auth_tokens.json` does not let anyone authenticate.

The plain token is shown to the user exactly once at creation time and never
again — same model as GitHub personal-access tokens. To revoke, drop the record.
"""

import base64
import hashlib
import json
import os
import secrets
import string
from datetime import datetime, timedelta, timezone

from aiohttp import web


TOKEN_PREFIX = "lp_"
_ID_LEN = 12
_SECRET_LEN = 40


class AuthManager:
    def __init__(self, config):
        self.enabled = bool(((config.get("auth") or {}).get("enabled", False)))
        self.state_dir = os.path.expanduser(config["state_dir"])
        self.web_auth_cfg = config.get("web_auth") or {}
        self.web_enabled = bool(self.web_auth_cfg.get("enabled", False))
        self._sessions = {}
        self._session_cookie = "llm_proxy_session"

    # ---- token CRUD ----

    def create_token(self, name):
        name = (name or "").strip()
        if not name:
            raise ValueError("name is required")
        token_id = self._generate_hex(_ID_LEN)
        secret = self._generate_hex(_SECRET_LEN)
        plain = f"{TOKEN_PREFIX}{token_id}_{secret}"
        record = {
            "id": token_id,
            "name": name,
            "secret_hash": self._hash(secret),
            "created_at": self._now(),
            "last_used_at": None,
        }
        tokens = self._load_tokens()
        tokens.append(record)
        self._save_tokens(tokens)
        public = {k: v for k, v in record.items() if k != "secret_hash"}
        public["token"] = plain
        return public

    def list_tokens(self):
        return [
            {k: v for k, v in r.items() if k != "secret_hash"}
            for r in self._load_tokens()
        ]

    def revoke_token(self, identifier):
        identifier = (identifier or "").strip()
        if not identifier:
            return False
        tokens = self._load_tokens()
        before = len(tokens)
        tokens = [r for r in tokens if r.get("id") != identifier and r.get("name") != identifier]
        if len(tokens) == before:
            return False
        self._save_tokens(tokens)
        return True

    # ---- request guards ----

    def require_control(self, request):
        web_user = self.get_web_user(request)
        if web_user:
            return f"web:{web_user}"
        if not self.enabled:
            return None
        return self._require_token(request)

    def require_client(self, request):
        if not self.enabled:
            return None
        return self._require_token(request)

    # ---- web (cookie) session ----

    def get_web_user(self, request):
        if not self.web_enabled:
            return None
        session_id = request.cookies.get(self._session_cookie)
        if not session_id:
            return None
        session = self._sessions.get(session_id)
        if not session:
            return None
        expires_at = session.get("expires_at")
        if expires_at and datetime.now(timezone.utc) >= datetime.fromisoformat(expires_at):
            self._sessions.pop(session_id, None)
            return None
        return session.get("username")

    def web_login(self, username, password):
        username = (username or "").strip()
        password = password or ""
        if not username or not password:
            raise PermissionError("username and password are required")
        user = None
        for candidate in self.web_auth_cfg.get("users", []):
            if candidate.get("username") == username:
                user = candidate
                break
        if not user or not self._verify_web_password(user, password):
            raise PermissionError("invalid username or password")
        session_id = self._generate_hex(48)
        ttl_hours = float(self.web_auth_cfg.get("session_ttl_hours", 168))
        self._sessions[session_id] = {
            "username": username,
            "created_at": self._now(),
            "expires_at": (datetime.now(timezone.utc) + timedelta(hours=ttl_hours)).isoformat(),
        }
        return session_id

    def web_logout(self, request):
        session_id = request.cookies.get(self._session_cookie)
        if session_id:
            self._sessions.pop(session_id, None)

    def session_cookie_name(self):
        return self._session_cookie

    def secure_cookies(self):
        return bool(self.web_auth_cfg.get("secure_cookies", False))

    # ---- internals ----

    def _require_token(self, request):
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            raise web.HTTPUnauthorized(
                text=json.dumps({"error": "missing Bearer token"}),
                content_type="application/json",
            )
        token = auth_header[len("Bearer "):].strip()
        record = self._lookup_token(token)
        if record is None:
            raise web.HTTPForbidden(
                text=json.dumps({"error": "invalid or revoked token"}),
                content_type="application/json",
            )
        return record["name"]

    def _lookup_token(self, plain):
        if not plain or not plain.startswith(TOKEN_PREFIX):
            return None
        rest = plain[len(TOKEN_PREFIX):]
        if "_" not in rest:
            return None
        token_id, secret = rest.split("_", 1)
        if not token_id or not secret:
            return None
        tokens = self._load_tokens()
        for record in tokens:
            if record.get("id") != token_id:
                continue
            if not secrets.compare_digest(record.get("secret_hash", ""), self._hash(secret)):
                return None
            record["last_used_at"] = self._now()
            self._save_tokens(tokens)
            return record
        return None

    def _verify_web_password(self, user, password):
        plain = user.get("password")
        if plain is not None:
            return secrets.compare_digest(str(plain), password)
        salt_b64 = user.get("salt")
        password_hash_b64 = user.get("password_hash")
        if not salt_b64 or not password_hash_b64:
            return False
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(password_hash_b64)
        iterations = int(user.get("iterations", 600_000))
        actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
        return secrets.compare_digest(actual, expected)

    def _tokens_path(self):
        return os.path.join(self.state_dir, "auth_tokens.json")

    def _load_tokens(self):
        path = self._tokens_path()
        if not os.path.isfile(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        # Legacy migration: dict {host_id: {token, issued_at}} → list of records.
        # Old plain tokens cannot be recovered; mark them as migrated-but-invalid
        # so the user knows to re-issue.
        migrated = []
        if isinstance(data, dict):
            for host_id, payload in data.items():
                if not isinstance(payload, dict):
                    continue
                migrated.append({
                    "id": self._generate_hex(_ID_LEN),
                    "name": str(host_id),
                    "secret_hash": "__legacy_invalid__",
                    "created_at": payload.get("issued_at") or self._now(),
                    "last_used_at": None,
                    "note": "migrated from legacy host_id format; revoke and re-create",
                })
        if migrated:
            self._save_tokens(migrated)
            print(
                f"  [auth] migrated {len(migrated)} legacy token(s); "
                f"old tokens are no longer valid — re-create with "
                f"`llm-proxy auth create <name>`"
            )
        return migrated

    def _save_tokens(self, tokens):
        path = self._tokens_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(tokens, f, ensure_ascii=True, indent=2, sort_keys=True)
        os.replace(tmp, path)

    @staticmethod
    def _hash(secret):
        return "sha256:" + hashlib.sha256(secret.encode("utf-8")).hexdigest()

    @staticmethod
    def _generate_hex(length):
        alphabet = string.hexdigits[:16]  # lowercase + digits
        return "".join(secrets.choice(alphabet) for _ in range(length))

    @staticmethod
    def _now():
        return datetime.now(timezone.utc).isoformat()
