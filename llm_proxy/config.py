"""Configuration loader."""

import os
import yaml


DEFAULT_CONFIG = {
    "listen": {"host": "0.0.0.0", "port": 8090},
    "cache_dir": "~/.cache/llm-proxy/kv",
    "llama_server": "llama-server",
    "models": {},
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
    config["llama_server"] = os.path.expanduser(config["llama_server"])

    # Expand paths in model definitions
    for name, m in config["models"].items():
        if "gguf" in m:
            m["gguf"] = os.path.expanduser(m["gguf"])

    return config


def _deep_update(base, override):
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
