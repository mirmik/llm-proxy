"""XTTS runtime driver — thin pass-through.

Command and env are fully controlled by runtime.command / runtime.env.
XTTS loads a large model + computes speaker conditioning on startup,
so the default startup_timeout is higher than the base class (240s).
"""

from llm_proxy.drivers.base import RuntimeDriver
from llm_proxy.drivers.utils import resolve_python_bin


class XTTSDriver(RuntimeDriver):
    name = "xtts"

    def kind(self):
        return "tts"

    def startup_timeout(self, runtime_name, runtime):
        return float(runtime.get("startup_timeout", 240.0))

    def runtime_meta(self, runtime_name, runtime, global_config):
        meta = super().runtime_meta(runtime_name, runtime, global_config)
        meta["python_bin"] = resolve_python_bin(runtime)
        return meta
