"""Shared helper utilities for runtime drivers."""

import os
import sys


def resolve_python_bin(profile):
    """Resolve interpreter path from profile fields."""
    python_bin = profile.get("python_bin")
    if python_bin:
        return os.path.expanduser(str(python_bin))

    venv = profile.get("venv")
    if venv:
        return os.path.join(os.path.expanduser(str(venv)), "bin", "python")

    return sys.executable
