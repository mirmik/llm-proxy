"""Runtime driver registry."""

from llm_proxy.drivers.faster_whisper import FasterWhisperDriver
from llm_proxy.drivers.llama_cpp import LlamaCppDriver
from llm_proxy.drivers.xtts import XTTSDriver
from llm_proxy.drivers.voxcpm import VoxCPMDriver


def build_driver_registry():
    drivers = [
        LlamaCppDriver(),
        FasterWhisperDriver(),
        XTTSDriver(),
        VoxCPMDriver(),
    ]
    return {driver.name: driver for driver in drivers}
