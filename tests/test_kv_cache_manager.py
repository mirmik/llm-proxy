import os
import tempfile
import time
import unittest
from pathlib import Path

from llm_proxy.kv_cache_manager import KVCacheManager
from llm_proxy.slot_manager import SlotManager


class KVCacheManagerTests(unittest.IsolatedAsyncioTestCase):
    async def test_ttl_cleanup_deletes_old_inactive_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            old_file = cache_dir / "model" / "old-session"
            fresh_file = cache_dir / "model" / "fresh-session"
            old_file.parent.mkdir(parents=True)
            old_file.write_bytes(b"old")
            fresh_file.write_bytes(b"fresh")
            old_time = time.time() - 10 * 3600
            Path(old_file).touch()
            old_file_stat_time = old_time
            os.utime(old_file, (old_file_stat_time, old_file_stat_time))

            manager = KVCacheManager(
                {
                    "cache_dir": str(cache_dir),
                    "kv_cache": {
                        "cleanup_enabled": True,
                        "ttl_hours": 1,
                        "cleanup_interval_minutes": 30,
                    },
                },
                SlotManager(),
            )

            result = await manager.cleanup_once()

            self.assertEqual(result.deleted, 1)
            self.assertFalse(old_file.exists())
            self.assertTrue(fresh_file.exists())

    async def test_ttl_cleanup_keeps_active_session_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            active_file = cache_dir / "model" / "active-session"
            active_file.parent.mkdir(parents=True)
            active_file.write_bytes(b"active")
            old_time = time.time() - 10 * 3600
            os.utime(active_file, (old_time, old_time))

            sm = SlotManager()
            sm.init_model("model", 1)
            slot = await sm.acquire("model", "active-session", 8081)
            await sm.release("model", slot)

            manager = KVCacheManager(
                {
                    "cache_dir": str(cache_dir),
                    "kv_cache": {
                        "cleanup_enabled": True,
                        "ttl_hours": 1,
                        "keep_active_sessions": True,
                    },
                },
                sm,
            )

            result = await manager.cleanup_once()

            self.assertEqual(result.deleted, 0)
            self.assertTrue(active_file.exists())

    async def test_per_runtime_size_limit_deletes_oldest_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            runtime_dir = cache_dir / "model"
            runtime_dir.mkdir(parents=True)
            old_file = runtime_dir / "old"
            new_file = runtime_dir / "new"
            old_file.write_bytes(b"1" * 700)
            new_file.write_bytes(b"2" * 700)
            old_time = time.time() - 100
            new_time = time.time()
            os.utime(old_file, (old_time, old_time))
            os.utime(new_file, (new_time, new_time))

            manager = KVCacheManager(
                {
                    "cache_dir": str(cache_dir),
                    "kv_cache": {
                        "cleanup_enabled": True,
                        "ttl_hours": None,
                        "max_per_runtime_gb": 0.000001,
                    },
                },
                SlotManager(),
            )

            result = await manager.cleanup_once()

            self.assertEqual(result.deleted, 1)
            self.assertFalse(old_file.exists())
            self.assertTrue(new_file.exists())


if __name__ == "__main__":
    unittest.main()
