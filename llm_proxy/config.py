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

    for name, runtime in (config.get("runtimes") or {}).items():
        _validate_runtime(name, runtime)
        base = runtime.get("base") or {}
        _expand_profile_paths(base)
        for _, profile in (runtime.get("profiles") or {}).items():
            _expand_profile_paths(profile)

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
    profiles = runtime.get("profiles") or {}
    if not profiles:
        raise ValueError(f"Runtime '{name}' has no profiles")
    default_profile = runtime.get("default_profile")
    if default_profile and default_profile not in profiles:
        raise ValueError(
            f"Runtime '{name}' default_profile='{default_profile}' "
            f"not in profiles {list(profiles)}"
        )


def _deep_update(base, override):
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v


def _expand_profile_paths(profile):
    for key, value in list(profile.items()):
        if not isinstance(value, str):
            continue
        if key in {"reference_wav", "python_bin", "workdir", "venv"}:
            profile[key] = os.path.expanduser(value)
