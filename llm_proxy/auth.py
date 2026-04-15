"""Server-side auth helpers for llm-proxy control API."""

import base64
import hashlib
import json
import os
import secrets
import string
from datetime import datetime, timedelta, timezone

from aiohttp import web


class AuthManager:
    """Invite/token/password auth modeled after Nemor's host bootstrap."""

    def __init__(self, config):
        self.enabled = bool(((config.get("auth") or {}).get("enabled", False)))
        self.state_dir = os.path.expanduser(config["state_dir"])
        self.auth_cfg = config.get("auth") or {}
        self.web_auth_cfg = config.get("web_auth") or {}
        self.web_enabled = bool(self.web_auth_cfg.get("enabled", False))
        self._sessions = {}
        self._session_cookie = "llm_proxy_session"

    def create_invite(self, host_id, ttl_hours=24.0):
        invites = self._load_json(self._invites_path())
        token = self._generate_token(32)
        invites[token] = {
            "host_id": host_id,
            "created_at": self._now(),
            "expires_at": (datetime.now(timezone.utc) + timedelta(hours=ttl_hours)).isoformat(),
            "used_at": None,
        }
        self._save_json(self._invites_path(), invites)
        return token, invites[token]

    def set_bootstrap_password(self, password, *, iterations=600_000):
        salt = secrets.token_bytes(16)
        digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
        payload = {
            "salt": base64.b64encode(salt).decode("ascii"),
            "password_hash": base64.b64encode(digest).decode("ascii"),
            "iterations": iterations,
            "updated_at": self._now(),
        }
        self._save_json(self._bootstrap_password_path(), payload)

    def clear_bootstrap_password(self):
        path = self._bootstrap_password_path()
        if os.path.isfile(path):
            os.remove(path)

    def bootstrap(self, *, token="", password="", host_id=""):
        token = (token or "").strip()
        password = password or ""
        host_id = (host_id or "").strip()
        if bool(token) == bool(password):
            raise ValueError("pass exactly one of token or password")

        if token:
            invite, error = self._consume_invite(token)
            if error:
                raise PermissionError(error)
            host_id = invite["host_id"]
        else:
            if not self._verify_bootstrap_password(password):
                raise PermissionError("invalid password")
            if not host_id:
                raise ValueError("host_id is required for password login")

        auth_token = self._issue_host_token(host_id)
        return {"host_id": host_id, "token": auth_token}

    def require_control(self, request):
        web_user = self.get_web_user(request)
        if web_user:
            return web_user
        if not self.enabled:
            return None
        return self._require_host_token(request)

    def require_client(self, request):
        if not self.enabled:
            return None
        return self._require_host_token(request)

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
        if not user:
            raise PermissionError("invalid username or password")
        if not self._verify_web_password(user, password):
            raise PermissionError("invalid username or password")
        session_id = self._generate_token(48)
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

    def _require_host_token(self, request):
        auth_header = request.headers.get("Authorization", "")
        host_id = (request.headers.get("X-LLM-Proxy-Host-ID") or "").strip()
        token = ""
        if auth_header.startswith("Bearer "):
            token = auth_header[len("Bearer "):].strip()
        if not token or not host_id:
            raise web.HTTPUnauthorized(
                text=json.dumps({"error": "control auth requires Bearer token and X-LLM-Proxy-Host-ID"}),
                content_type="application/json",
            )
        if not self._verify_host_token(host_id, token):
            raise web.HTTPForbidden(
                text=json.dumps({"error": f"token is not valid for host '{host_id}'"}),
                content_type="application/json",
            )
        return host_id

    def local_auth_path(self):
        return os.path.join(self.state_dir, "client_auth.json")

    def save_local_auth(self, *, server, host_id, token, server_fingerprint=None):
        payload = {
            "server": server,
            "host_id": host_id,
            "token": token,
        }
        if server_fingerprint:
            payload["server_fingerprint"] = server_fingerprint
        self._save_json(self.local_auth_path(), payload)

    def load_local_auth(self):
        path = self.local_auth_path()
        if not os.path.isfile(path):
            return None
        return self._load_json(path)

    def _consume_invite(self, token):
        invites = self._load_json(self._invites_path())
        invite = invites.get(token)
        if not invite:
            return None, "invalid token"
        if invite.get("used_at"):
            return None, "token already used"
        expires_at = invite.get("expires_at")
        if expires_at and datetime.now(timezone.utc) >= datetime.fromisoformat(expires_at):
            return None, "token expired"
        invite["used_at"] = self._now()
        invites[token] = invite
        self._save_json(self._invites_path(), invites)
        return invite, None

    def _issue_host_token(self, host_id):
        tokens = self._load_json(self._tokens_path())
        token = self._generate_token(48)
        tokens[host_id] = {
            "token": token,
            "issued_at": self._now(),
        }
        self._save_json(self._tokens_path(), tokens)
        return token

    def _verify_host_token(self, host_id, token):
        tokens = self._load_json(self._tokens_path())
        record = tokens.get(host_id) or {}
        return secrets.compare_digest(record.get("token", ""), token or "")

    def _verify_bootstrap_password(self, password):
        configured = self.auth_cfg.get("bootstrap_password") or ""
        if configured:
            return secrets.compare_digest(password, configured)

        path = self._bootstrap_password_path()
        if not os.path.isfile(path):
            return False
        payload = self._load_json(path)
        salt = base64.b64decode(payload["salt"])
        expected = base64.b64decode(payload["password_hash"])
        iterations = int(payload.get("iterations", 600_000))
        actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
        return secrets.compare_digest(actual, expected)

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

    def _invites_path(self):
        return os.path.join(self.state_dir, "auth_invites.json")

    def _tokens_path(self):
        return os.path.join(self.state_dir, "auth_tokens.json")

    def _bootstrap_password_path(self):
        return os.path.join(self.state_dir, "auth_bootstrap.json")

    def _load_json(self, path):
        if not os.path.isfile(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_json(self, path, payload):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2, sort_keys=True)
        os.replace(tmp, path)

    def _generate_token(self, length):
        alphabet = string.ascii_letters + string.digits
        return "".join(secrets.choice(alphabet) for _ in range(length))

    def _now(self):
        return datetime.now(timezone.utc).isoformat()
