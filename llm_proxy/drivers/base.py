"""Base interface for runtime drivers."""


class RuntimeDriver:
    """Base class for runtime drivers."""

    name = "base"

    def kind(self):
        return "generic"

    def validate_profile(self, profile):
        """Raise ValueError when profile is invalid."""

    def build_command(self, runtime_name, profile, global_config):
        raise NotImplementedError

    def build_env(self, runtime_name, profile, global_config):
        return {}

    def resolve_health_url(self, runtime_name, profile, global_config):
        return None

    def stop_signal(self, runtime_name, profile):
        return "SIGTERM"

    def startup_timeout(self, runtime_name, profile):
        return 120.0

    def runtime_meta(self, runtime_name, profile, global_config):
        return {}
