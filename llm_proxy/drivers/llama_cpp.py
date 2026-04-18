"""llama.cpp runtime driver — thin pass-through.

The entire command (binary path + all flags) is in profile.command.
Available placeholders: {port}, {name}, {cache_dir}, {state_dir},
{python}, {runtime_kv}, {slots}, {defaults.*}.

The driver only supplies the `kind` and ensures {runtime_kv} parent
directory exists (used by --slot-save-path)."""

import os

from llm_proxy.drivers.base import RuntimeDriver


class LlamaCppDriver(RuntimeDriver):
    name = "llama_cpp"

    def kind(self):
        return "llm"

    def build_command(self, runtime_name, profile, global_config):
        # Pre-create the KV cache dir so --slot-save-path works out of the box.
        cache_dir = os.path.expanduser(str(global_config.get("cache_dir", "~/.cache/llm-proxy")))
        os.makedirs(os.path.join(cache_dir, runtime_name), exist_ok=True)
        return super().build_command(runtime_name, profile, global_config)

    def runtime_meta(self, runtime_name, profile, global_config):
        meta = super().runtime_meta(runtime_name, profile, global_config)
        meta["context"] = int(profile.get("context", 0)) or None
        return meta
