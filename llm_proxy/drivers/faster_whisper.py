"""faster-whisper runtime driver."""

import os

from llm_proxy.drivers.base import RuntimeDriver
from llm_proxy.drivers.utils import resolve_python_bin


class FasterWhisperDriver(RuntimeDriver):
    name = "faster_whisper"

    def kind(self):
        return "stt"

    def validate_profile(self, profile):
        if not profile.get("port"):
            raise ValueError("faster_whisper profile requires 'port'")
        if not profile.get("model_size"):
            raise ValueError("faster_whisper profile requires 'model_size'")

    def build_command(self, runtime_name, profile, global_config):
        python_bin = resolve_python_bin(profile)
        script = os.path.join(os.path.dirname(os.path.dirname(__file__)), "runners", "faster_whisper_server.py")
        return [python_bin, script]

    def build_env(self, runtime_name, profile, global_config):
        env = {
            "LLM_PROXY_FW_PORT": str(profile["port"]),
            "LLM_PROXY_FW_MODEL_SIZE": str(profile["model_size"]),
            "LLM_PROXY_FW_DEVICE": str(profile.get("device", "cuda")),
            "LLM_PROXY_FW_COMPUTE_TYPE": str(profile.get("compute_type", "float16")),
            "LLM_PROXY_FW_SAMPLE_RATE": str(profile.get("sample_rate", 16000)),
        }
        language = profile.get("language")
        if language is not None:
            env["LLM_PROXY_FW_LANGUAGE"] = str(language)
        token = profile.get("token")
        if token:
            env["LLM_PROXY_FW_TOKEN"] = str(token)
        return env

    def resolve_health_url(self, runtime_name, profile, global_config):
        return f"http://127.0.0.1:{int(profile['port'])}/health"

    def runtime_meta(self, runtime_name, profile, global_config):
        return {
            "port": int(profile["port"]),
            "model_size": profile["model_size"],
            "language": profile.get("language"),
            "gpu": profile.get("gpu"),
            "python_bin": resolve_python_bin(profile),
        }
