"""llama.cpp runtime driver."""

import glob
import os
import shlex

from llm_proxy.drivers.base import RuntimeDriver


class LlamaCppDriver(RuntimeDriver):
    name = "llama_cpp"

    def kind(self):
        return "llm"

    def validate_profile(self, profile):
        if not profile.get("gguf"):
            raise ValueError("llama_cpp profile requires 'gguf'")
        if not profile.get("port"):
            raise ValueError("llama_cpp profile requires 'port'")

    def build_command(self, runtime_name, profile, global_config):
        gguf_pattern = os.path.expanduser(profile["gguf"])
        gguf_files = sorted(glob.glob(gguf_pattern))
        if not gguf_files:
            raise FileNotFoundError(f"No GGUF files matching: {gguf_pattern}")
        model_path = gguf_files[0]

        slots = int(profile.get("slots", 1))
        context = int(profile.get("context", 32768))
        port = int(profile["port"])
        llama_bin = os.path.expanduser(global_config["llama_server"])
        cmd = [
            llama_bin,
            "--model", model_path,
            "-ngl", str(profile.get("ngl", 99)),
            "-c", str(context),
            "-np", str(slots),
            "--host", "127.0.0.1",
            "--port", str(port),
        ]

        cache_dir = os.path.expanduser(global_config["cache_dir"])
        runtime_cache = os.path.join(cache_dir, runtime_name)
        os.makedirs(runtime_cache, exist_ok=True)
        cmd += ["--slot-save-path", runtime_cache]

        extra_args = profile.get("extra_args", "")
        if extra_args:
            cmd += shlex.split(extra_args)
        return cmd

    def resolve_health_url(self, runtime_name, profile, global_config):
        return f"http://127.0.0.1:{int(profile['port'])}/health"

    def runtime_meta(self, runtime_name, profile, global_config):
        return {
            "port": int(profile["port"]),
            "slots": int(profile.get("slots", 1)),
            "context": int(profile.get("context", 32768)),
            "gguf": os.path.expanduser(profile["gguf"]),
            "gpu": profile.get("gpu"),
        }
