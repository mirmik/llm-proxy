"""Base interface for runtime drivers.

Drivers are thin: each one knows its `kind` (llm/stt/tts/generic) and
the default health path. The command and env come straight from the
runtime config through template expansion — no hidden flags.
"""

from llm_proxy.drivers.utils import build_substitutions, render_command, render_env


DEFAULT_HEALTH_PATH = {
    "llm": "/health",
    "stt": "/health",
    "tts": "/health",
    "generic": "/health",
}


class RuntimeDriver:
    """Base class for runtime drivers."""

    name = "base"

    def kind(self):
        return "generic"

    def validate_runtime(self, runtime):
        """Common validation. Subclasses call super() first, then add extras."""
        if not runtime.get("command"):
            raise ValueError(f"{self.name} runtime requires 'command'")
        if not runtime.get("port"):
            raise ValueError(f"{self.name} runtime requires 'port'")

    def build_command(self, runtime_name, runtime, global_config):
        subs = build_substitutions(runtime_name, runtime, global_config)
        return render_command(runtime["command"], subs)

    def build_env(self, runtime_name, runtime, global_config):
        subs = build_substitutions(runtime_name, runtime, global_config)
        return render_env(runtime.get("env") or {}, subs)

    def resolve_health_url(self, runtime_name, runtime, global_config):
        port = int(runtime["port"])
        path = runtime.get("health_path") or DEFAULT_HEALTH_PATH.get(self.kind(), "/health")
        host = runtime.get("health_host", "127.0.0.1")
        return f"http://{host}:{port}{path}"

    def stop_signal(self, runtime_name, runtime):
        return str(runtime.get("stop_signal", "SIGTERM"))

    def startup_timeout(self, runtime_name, runtime):
        return float(runtime.get("startup_timeout", 120.0))

    def runtime_meta(self, runtime_name, runtime, global_config):
        return {
            "port": int(runtime["port"]),
            "slots": int(runtime.get("slots", 1)),
            "gpu": runtime.get("gpu"),
        }
