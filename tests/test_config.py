import os
import tempfile
import unittest

import yaml

from llm_proxy.config import load_config


class ConfigRuntimeTests(unittest.TestCase):
    def test_loads_flat_runtime_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "config.yaml")
            with open(path, "w", encoding="utf-8") as f:
                yaml.safe_dump({
                    "runtimes": {
                        "model": {
                            "driver": "llama_cpp",
                            "port": 8081,
                            "slots": 2,
                            "command": "server --port {port}",
                        },
                    },
                }, f)

            config = load_config(path)

        runtime = config["runtimes"]["model"]
        self.assertEqual(runtime["port"], 8081)
        self.assertEqual(runtime["slots"], 2)
        self.assertNotIn("profiles", runtime)
        self.assertNotIn("default_profile", runtime)

    def test_collapses_legacy_profiles_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "config.yaml")
            with open(path, "w", encoding="utf-8") as f:
                yaml.safe_dump({
                    "runtimes": {
                        "model": {
                            "driver": "llama_cpp",
                            "gpu": "local",
                            "base": {"slots": 1},
                            "default_profile": "default",
                            "profiles": {
                                "default": {
                                    "port": 8081,
                                    "slots": 2,
                                    "command": "server --port {port}",
                                },
                            },
                        },
                    },
                }, f)

            config = load_config(path)

        runtime = config["runtimes"]["model"]
        self.assertEqual(runtime["gpu"], "local")
        self.assertEqual(runtime["port"], 8081)
        self.assertEqual(runtime["slots"], 2)
        self.assertEqual(runtime["command"], "server --port {port}")
        self.assertNotIn("profiles", runtime)
        self.assertNotIn("default_profile", runtime)


if __name__ == "__main__":
    unittest.main()
