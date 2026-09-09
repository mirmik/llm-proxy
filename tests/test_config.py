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

    def test_statistics_defaults_are_privacy_preserving(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "config.yaml")
            with open(path, "w", encoding="utf-8") as f:
                f.write("{}\n")
            config = load_config(path)

        self.assertTrue(config["statistics"]["enabled"])
        self.assertEqual(config["statistics"]["client_identity"], "token")
        self.assertFalse(config["statistics"]["include_host"])
        self.assertEqual(config["statistics"]["retention_days"], 30)

    def test_kv_checkpoint_throttling_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "config.yaml")
            with open(path, "w", encoding="utf-8") as f:
                f.write("{}\n")
            config = load_config(path)

        self.assertEqual(config["kv_cache"]["checkpoint_interval_minutes"], 5)
        self.assertEqual(config["kv_cache"]["checkpoint_max_requests"], 5)

    def test_kv_checkpoint_triggers_can_be_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "config.yaml")
            with open(path, "w", encoding="utf-8") as f:
                yaml.safe_dump({
                    "kv_cache": {
                        "checkpoint_interval_minutes": None,
                        "checkpoint_max_requests": None,
                    },
                }, f)
            config = load_config(path)

        self.assertIsNone(config["kv_cache"]["checkpoint_interval_minutes"])
        self.assertIsNone(config["kv_cache"]["checkpoint_max_requests"])

    def test_invalid_kv_checkpoint_policy_is_rejected(self):
        invalid_policies = (
            {"checkpoint_interval_minutes": -1},
            {"checkpoint_max_requests": 0},
            {"checkpoint_max_requests": 1.5},
        )
        for policy in invalid_policies:
            with self.subTest(policy=policy), tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "config.yaml")
                with open(path, "w", encoding="utf-8") as f:
                    yaml.safe_dump({"kv_cache": policy}, f)
                with self.assertRaises(ValueError):
                    load_config(path)

    def test_embedding_route_is_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "config.yaml")
            with open(path, "w", encoding="utf-8") as f:
                yaml.safe_dump({
                    "routes": {
                        "bge-m3": {
                            "kind": "embedding",
                            "backends": [{"url": "https://embedding-host:8080"}],
                        },
                    },
                }, f)

            config = load_config(path)

        self.assertEqual(config["routes"]["bge-m3"]["kind"], "embedding")


if __name__ == "__main__":
    unittest.main()
