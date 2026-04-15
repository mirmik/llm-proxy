"""Runtime driver registry."""

from llm_proxy.drivers.faster_whisper import FasterWhisperDriver
from llm_proxy.drivers.llama_cpp import LlamaCppDriver
from llm_proxy.drivers.xtts import XTTSDriver


def build_driver_registry():
    drivers = [
        LlamaCppDriver(),
        FasterWhisperDriver(),
        XTTSDriver(),
    ]
    return {driver.name: driver for driver in drivers}
