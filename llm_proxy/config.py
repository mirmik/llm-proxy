"""Configuration loader."""

import copy
import os

import yaml


DEFAULT_CONFIG = {
    "listen": {"host": "0.0.0.0", "port": 8090},
    "cache_dir": "~/.cache/llm-proxy",
    "state_dir": "~/.config/llm-proxy",
    "tls": {
        "enabled": False,
        "cert_file": "~/.config/llm-proxy/certs/server.crt",
        "key_file": "~/.config/llm-proxy/certs/server.key",
    },
    "auth": {
        "enabled": False,
    },
    "control": {
        "host": "0.0.0.0",
        "port": None,  # defaults to listen.port + 1 if not set
    },
    "web_auth": {
        "enabled": False,
        "session_ttl_hours": 168,
        "secure_cookies": False,
        "users": [],
    },
    "defaults": {},
    "runtimes": {},
    "hosts": {},
    "routes": {},
    "statistics": {
        "enabled": True,
        "database": None,
        "retention_days": 30,
        "cleanup_interval_hours": 24,
        "client_identity": "token",
        "include_host": False,
    },
    "kv_cache": {
        "cleanup_enabled": True,
        "ttl_hours": 168,
        "cleanup_interval_minutes": 30,
        "max_total_gb": None,
        "max_per_runtime_gb": None,
        "keep_active_sessions": True,
        "checkpoint_interval_minutes": 5,
        "checkpoint_max_requests": 5,
    },
}


def load_config(path=None):
    if path is None:
        candidates = [
            os.path.expanduser("~/.config/llm-proxy/config.yaml"),
            "config.yaml",
        ]
        for c in candidates:
            if os.path.isfile(c):
                path = c
                break

    config = copy.deepcopy(DEFAULT_CONFIG)
    config["_config_path"] = os.path.expanduser(path) if path else None
    if path and os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
        _deep_update(config, user)
        print(f"  Config: {path}")
    else:
        print("  Config: defaults (no config file found)")

    config["cache_dir"] = os.path.expanduser(config["cache_dir"])
    config["state_dir"] = os.path.expanduser(config["state_dir"])

    tls_cfg = config.get("tls") or {}
    if "cert_file" in tls_cfg:
        tls_cfg["cert_file"] = os.path.expanduser(tls_cfg["cert_file"])
    if "key_file" in tls_cfg:
        tls_cfg["key_file"] = os.path.expanduser(tls_cfg["key_file"])

    defaults = config.get("defaults") or {}
    for key, value in list(defaults.items()):
        if isinstance(value, str):
            defaults[key] = os.path.expanduser(value)

    runtimes = config.get("runtimes") or {}
    for name, runtime in list(runtimes.items()):
        normalized = _normalize_runtime(runtime)
        runtimes[name] = normalized
        _validate_runtime(name, normalized)
        _expand_runtime_paths(normalized)

    for name, route in (config.get("routes") or {}).items():
        _validate_route(name, route)

    _validate_statistics(config.get("statistics") or {})
    _validate_kv_cache(config.get("kv_cache") or {})

    return config


def load_config_file(path):
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_config_file(path, payload):
    path = os.path.expanduser(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=False)
    os.replace(tmp, path)


def _validate_runtime(name, runtime):
    if not runtime.get("driver"):
        raise ValueError(f"Runtime '{name}' is missing 'driver'")
    if not runtime.get("command"):
        raise ValueError(f"Runtime '{name}' is missing 'command'")
    if not runtime.get("port"):
        raise ValueError(f"Runtime '{name}' is missing 'port'")


def _deep_update(base, override):
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v


def _validate_route(name, route):
    if not isinstance(route, dict):
        raise ValueError(f"Route '{name}' must be a dict")
    backends = route.get("backends", [])
    if not isinstance(backends, list):
        raise ValueError(f"Route '{name}' must have a 'backends' list")
    for i, b in enumerate(backends):
        if not isinstance(b, dict):
            raise ValueError(f"Route '{name}'.backends[{i}] must be a dict")
        if not b.get("url"):
            raise ValueError(f"Route '{name}'.backends[{i}] requires 'url'")
    kind = route.get("kind", "llm")
    if kind not in ("llm", "embedding", "stt", "tts"):
        raise ValueError(f"Route '{name}' has unknown kind: {kind}")


def _validate_statistics(statistics):
    try:
        retention_days = float(statistics.get("retention_days", 30))
        cleanup_hours = float(statistics.get("cleanup_interval_hours", 24))
    except (TypeError, ValueError) as exc:
        raise ValueError("statistics retention/cleanup values must be numbers") from exc
    if retention_days < 0:
        raise ValueError("statistics.retention_days must be >= 0")
    if cleanup_hours <= 0:
        raise ValueError("statistics.cleanup_interval_hours must be > 0")
    if statistics.get("client_identity", "token") not in ("token", "none"):
        raise ValueError("statistics.client_identity must be 'token' or 'none'")


def _validate_kv_cache(kv_cache):
    interval = kv_cache.get("checkpoint_interval_minutes", 5)
    if interval is not None:
        try:
            interval = float(interval)
        except (TypeError, ValueError) as exc:
            raise ValueError("kv_cache.checkpoint_interval_minutes must be a number or null") from exc
        if interval < 0:
            raise ValueError("kv_cache.checkpoint_interval_minutes must be >= 0")

    max_requests = kv_cache.get("checkpoint_max_requests", 5)
    if max_requests is not None:
        if isinstance(max_requests, bool):
            raise ValueError("kv_cache.checkpoint_max_requests must be a positive integer or null")
        try:
            parsed_number = float(max_requests)
        except (TypeError, ValueError) as exc:
            raise ValueError("kv_cache.checkpoint_max_requests must be a positive integer or null") from exc
        if not parsed_number.is_integer() or parsed_number < 1:
            raise ValueError("kv_cache.checkpoint_max_requests must be a positive integer or null")


def _normalize_runtime(runtime):
    """Collapse legacy profiles/default config into the flat runtime format."""
    if not isinstance(runtime, dict):
        return runtime
    if "profiles" not in runtime:
        return runtime

    profiles = runtime.get("profiles") or {}
    profile_name = "default"
    if profile_name not in profiles:
        configured = runtime.get("default_profile")
        if configured in profiles:
            profile_name = configured
        elif len(profiles) == 1:
            profile_name = next(iter(profiles))

    profile = profiles.get(profile_name) or {}
    base = runtime.get("base") or {}
    normalized = {
        key: copy.deepcopy(value)
        for key, value in runtime.items()
        if key not in {"profiles", "default_profile", "base"}
    }
    merged = copy.deepcopy(base)
    merged.update(copy.deepcopy(profile))
    normalized.update(merged)
    return normalized


def _expand_runtime_paths(runtime):
    for key, value in list(runtime.items()):
        if not isinstance(value, str):
            continue
        if key in {"reference_wav", "python_bin", "workdir", "venv"}:
            runtime[key] = os.path.expanduser(value)
