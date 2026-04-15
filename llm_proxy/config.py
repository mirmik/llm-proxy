"""Configuration loader."""

import os
import yaml


DEFAULT_CONFIG = {
    "listen": {"host": "0.0.0.0", "port": 8090},
    "cache_dir": "~/.cache/llm-proxy/kv",
    "state_dir": "~/.config/llm-proxy",
    "llama_server": "llama-server",
    "tls": {
        "enabled": False,
        "cert_file": "~/.config/llm-proxy/certs/server.crt",
        "key_file": "~/.config/llm-proxy/certs/server.key",
    },
    "auth": {
        "enabled": False,
        "bootstrap_password": None,
    },
    "runtimes": {},
    "models": {},
    "services": {},
    "gpu_groups": {},
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

    config = dict(DEFAULT_CONFIG)
    if path and os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
        _deep_update(config, user)
        print(f"  Config: {path}")
    else:
        print("  Config: defaults (no config file found)")

    config["cache_dir"] = os.path.expanduser(config["cache_dir"])
    config["state_dir"] = os.path.expanduser(config["state_dir"])
    config["llama_server"] = os.path.expanduser(config["llama_server"])
    tls_cfg = config.get("tls") or {}
    if "cert_file" in tls_cfg:
        tls_cfg["cert_file"] = os.path.expanduser(tls_cfg["cert_file"])
    if "key_file" in tls_cfg:
        tls_cfg["key_file"] = os.path.expanduser(tls_cfg["key_file"])

    # Expand paths in model definitions
    for name, m in config["models"].items():
        if "gguf" in m:
            m["gguf"] = os.path.expanduser(m["gguf"])

    for _, runtime in config.get("runtimes", {}).items():
        base = runtime.get("base") or {}
        _expand_runtime_paths(base)
        profiles = runtime.get("profiles") or {}
        for _, profile in profiles.items():
            _expand_runtime_paths(profile)

    for _, svc in config.get("services", {}).items():
        _expand_service_paths(svc)

    return config


def _deep_update(base, override):
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v


def _expand_service_paths(service):
    for key in ("workdir", "health_url"):
        if key in service and isinstance(service[key], str) and key == "workdir":
            service[key] = os.path.expanduser(service[key])
    env = service.get("env") or {}
    for key, value in list(env.items()):
        if isinstance(value, str) and ("/" in value or value.startswith("~")):
            env[key] = os.path.expanduser(value)
    profiles = service.get("profiles") or {}
    for _, profile in profiles.items():
        _expand_service_paths(profile)


def _expand_runtime_paths(profile):
    for key, value in list(profile.items()):
        if not isinstance(value, str):
            continue
        if key in {"gguf", "reference_wav", "python_bin", "workdir"}:
            profile[key] = os.path.expanduser(value)
