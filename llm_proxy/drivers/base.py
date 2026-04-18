"""Base interface for runtime drivers.

Drivers are thin: each one knows its `kind` (llm/stt/tts/generic) and
the default health path. The command and env come straight from the
profile through template expansion — no hidden flags.
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

    def validate_profile(self, profile):
        """Common validation. Subclasses call super() first, then add extras."""
        if not profile.get("command"):
            raise ValueError(f"{self.name} profile requires 'command'")
        if not profile.get("port"):
            raise ValueError(f"{self.name} profile requires 'port'")

    def build_command(self, runtime_name, profile, global_config):
        subs = build_substitutions(runtime_name, profile, global_config)
        return render_command(profile["command"], subs)

    def build_env(self, runtime_name, profile, global_config):
        subs = build_substitutions(runtime_name, profile, global_config)
        return render_env(profile.get("env") or {}, subs)

    def resolve_health_url(self, runtime_name, profile, global_config):
        port = int(profile["port"])
        path = profile.get("health_path") or DEFAULT_HEALTH_PATH.get(self.kind(), "/health")
        host = profile.get("health_host", "127.0.0.1")
        return f"http://{host}:{port}{path}"

    def stop_signal(self, runtime_name, profile):
        return str(profile.get("stop_signal", "SIGTERM"))

    def startup_timeout(self, runtime_name, profile):
        return float(profile.get("startup_timeout", 120.0))

    def runtime_meta(self, runtime_name, profile, global_config):
        return {
            "port": int(profile["port"]),
            "slots": int(profile.get("slots", 1)),
            "gpu": profile.get("gpu"),
        }
