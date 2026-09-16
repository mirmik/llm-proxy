"""Managed VoxCPM2 process, using its own Python environment."""
from llm_proxy.drivers.base import RuntimeDriver
from llm_proxy.drivers.utils import resolve_python_bin


class VoxCPMDriver(RuntimeDriver):
    name = "voxcpm"

    def kind(self):
        return "tts"

    def startup_timeout(self, runtime_name, runtime):
        return float(runtime.get("startup_timeout", 240.0))

    def runtime_meta(self, runtime_name, runtime, global_config):
        meta = super().runtime_meta(runtime_name, runtime, global_config)
        meta["python_bin"] = resolve_python_bin(runtime)
        return meta
