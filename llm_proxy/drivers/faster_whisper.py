"""faster-whisper runtime driver — thin pass-through.

Command and env are fully controlled by runtime.command / runtime.env.
Available placeholders: {port}, {name}, {cache_dir}, {state_dir},
{python}, {runtime_kv}, {slots}, {defaults.*}.
"""

from llm_proxy.drivers.base import RuntimeDriver
from llm_proxy.drivers.utils import resolve_python_bin


class FasterWhisperDriver(RuntimeDriver):
    name = "faster_whisper"

    def kind(self):
        return "stt"

    def runtime_meta(self, runtime_name, runtime, global_config):
        meta = super().runtime_meta(runtime_name, runtime, global_config)
        meta["python_bin"] = resolve_python_bin(runtime)
        return meta
