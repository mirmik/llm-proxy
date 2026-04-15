"""XTTS runtime driver."""

import os
import sys

from llm_proxy.drivers.base import RuntimeDriver


class XTTSDriver(RuntimeDriver):
    name = "xtts"

    def kind(self):
        return "tts"

    def validate_profile(self, profile):
        if not profile.get("port"):
            raise ValueError("xtts profile requires 'port'")
        if not profile.get("reference_wav"):
            raise ValueError("xtts profile requires 'reference_wav'")

    def build_command(self, runtime_name, profile, global_config):
        python_bin = profile.get("python_bin") or sys.executable
        script = os.path.join(os.path.dirname(os.path.dirname(__file__)), "runners", "xtts_server.py")
        return [python_bin, script]

    def build_env(self, runtime_name, profile, global_config):
        env = {
            "LLM_PROXY_XTTS_PORT": str(profile["port"]),
            "LLM_PROXY_XTTS_DEVICE": str(profile.get("device", "cuda")),
            "LLM_PROXY_XTTS_MODEL_NAME": str(profile.get("model_name", "tts_models/multilingual/multi-dataset/xtts_v2")),
            "LLM_PROXY_XTTS_REFERENCE_WAV": os.path.expanduser(str(profile["reference_wav"])),
            "LLM_PROXY_XTTS_SAMPLE_RATE": str(profile.get("sample_rate", 24000)),
            "COQUI_TOS_AGREED": str(profile.get("coqui_tos_agreed", "1")),
        }
        default_language = profile.get("language")
        if default_language:
            env["LLM_PROXY_XTTS_LANGUAGE"] = str(default_language)
        return env

    def resolve_health_url(self, runtime_name, profile, global_config):
        return f"http://127.0.0.1:{int(profile['port'])}/health"

    def startup_timeout(self, runtime_name, profile):
        return float(profile.get("start_timeout", 240.0))

    def runtime_meta(self, runtime_name, profile, global_config):
        return {
            "port": int(profile["port"]),
            "reference_wav": os.path.expanduser(str(profile["reference_wav"])),
            "language": profile.get("language"),
            "gpu": profile.get("gpu"),
        }
