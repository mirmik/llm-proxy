"""Shared helper utilities for runtime drivers."""

import os
import shlex
import sys


def resolve_python_bin(profile):
    """Resolve interpreter path from profile fields."""
    python_bin = profile.get("python_bin")
    if python_bin:
        return os.path.expanduser(str(python_bin))

    venv = profile.get("venv")
    if venv:
        return os.path.join(os.path.expanduser(str(venv)), "bin", "python")

    return sys.executable


def build_substitutions(runtime_name, profile, global_config):
    """Build the placeholder map available in command/env templates.

    Flat keys:
        {port}, {name}, {cache_dir}, {state_dir},
        {python}, {runtime_kv}, {slots}, {runners_dir}

    Nested via dotted access (using DottedDict):
        {defaults.llama_bin}, {defaults.any_key_from_config_defaults}
    """
    cache_dir = os.path.expanduser(str(global_config.get("cache_dir", "~/.cache/llm-proxy")))
    state_dir = os.path.expanduser(str(global_config.get("state_dir", "~/.config/llm-proxy")))
    runtime_kv = os.path.join(cache_dir, runtime_name)
    # drivers/utils.py → drivers/ → llm_proxy/ → runners/
    runners_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runners")

    subs = {
        "port": profile.get("port", ""),
        "name": runtime_name,
        "cache_dir": cache_dir,
        "state_dir": state_dir,
        "python": resolve_python_bin(profile),
        "runtime_kv": runtime_kv,
        "slots": profile.get("slots", 1),
        "runners_dir": runners_dir,
        "defaults": DottedDict(global_config.get("defaults") or {}),
    }
    return subs


def render_command(command_str, substitutions):
    """Expand {placeholders} in a command string, shlex.split, expanduser each token."""
    try:
        rendered = str(command_str).format_map(_DefaultingMap(substitutions))
    except KeyError as e:
        raise ValueError(f"Unknown placeholder {e} in command template")
    tokens = shlex.split(rendered)
    return [os.path.expanduser(t) for t in tokens]


def render_env(env_dict, substitutions):
    """Expand {placeholders} in each value of an env dict. Returns str→str dict."""
    if not env_dict:
        return {}
    out = {}
    for key, value in env_dict.items():
        if value is None:
            continue
        try:
            rendered = str(value).format_map(_DefaultingMap(substitutions))
        except KeyError as e:
            raise ValueError(f"Unknown placeholder {e} in env value for {key!r}")
        out[str(key)] = rendered
    return out


class DottedDict(dict):
    """Dict that exposes `.key` access so `{defaults.llama_bin}` works in str.format."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class _DefaultingMap(dict):
    """Raises KeyError on unknown placeholder (so render_command can report nicely)."""

    def __missing__(self, key):
        raise KeyError(key)
