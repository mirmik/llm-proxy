"""Filesystem cleanup for persisted llama.cpp slot KV cache files."""

import asyncio
import os
import time
from pathlib import Path

from llm_proxy.oplog import elapsed_ms, log_event, now


class KVCacheManager:
    """Applies TTL and size limits to saved KV cache files under cache_dir."""

    def __init__(self, config, slot_manager):
        self.config = config
        self.slot_manager = slot_manager
        self.cache_dir = Path(config.get("cache_dir", "~/.cache/llm-proxy")).expanduser()
        self.policy = config.get("kv_cache") or {}
        self.enabled = bool(self.policy.get("cleanup_enabled", True))
        self.ttl_hours = _optional_float(self.policy.get("ttl_hours", 168))
        self.max_total_bytes = _gb_to_bytes(self.policy.get("max_total_gb"))
        self.max_per_runtime_bytes = _gb_to_bytes(self.policy.get("max_per_runtime_gb"))
        self.cleanup_interval_seconds = max(
            60.0,
            float(self.policy.get("cleanup_interval_minutes", 30)) * 60.0,
        )
        self.keep_active_sessions = bool(self.policy.get("keep_active_sessions", True))
        self._task = None
        self._stop_event = asyncio.Event()

    async def start(self):
        if not self.enabled or self._task is not None:
            return
        self._stop_event.clear()
        await self.cleanup_once(reason="startup")
        self._task = asyncio.create_task(self._loop())

    async def stop(self):
        self._stop_event.set()
        task = self._task
        self._task = None
        if task is None:
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    async def _loop(self):
        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self.cleanup_interval_seconds,
                )
                return
            except asyncio.TimeoutError:
                await self.cleanup_once(reason="interval")

    async def cleanup_once(self, reason="manual"):
        if not self.enabled:
            return CleanupResult()
        started = now()
        active = self.slot_manager.active_sessions() if self.keep_active_sessions else {}
        files = self._scan_files(active)
        result = CleanupResult(scanned=len(files))

        if self.ttl_hours is not None and self.ttl_hours >= 0:
            cutoff = time.time() - self.ttl_hours * 3600.0
            expired = [item for item in files if item.mtime < cutoff]
            result += self._delete_files(expired, reason="ttl")
            expired_set = set(expired)
            files = [item for item in files if item not in expired_set]

        if self.max_per_runtime_bytes:
            files, trimmed = self._enforce_per_runtime(files)
            result += trimmed

        if self.max_total_bytes:
            files, trimmed = self._enforce_total(files)
            result += trimmed

        self._remove_empty_dirs()
        log_event(
            "kv cleanup",
            reason=reason,
            scanned=result.scanned,
            deleted=result.deleted,
            bytes=result.bytes_deleted,
            ms=elapsed_ms(started),
        )
        return result

    def _scan_files(self, active_sessions):
        root = _safe_resolve(self.cache_dir)
        if root is None or not root.is_dir():
            return []
        out = []
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            resolved = _safe_resolve(path)
            if resolved is None or not _is_within(resolved, root):
                continue
            runtime = _runtime_name(root, resolved)
            if self._is_active_file(resolved, active_sessions.get(runtime, set())):
                continue
            try:
                stat = resolved.stat()
            except OSError:
                continue
            out.append(CacheFile(runtime=runtime, path=resolved, size=stat.st_size, mtime=stat.st_mtime))
        return out

    def _is_active_file(self, path, active_for_runtime):
        if not active_for_runtime:
            return False
        return path.name in active_for_runtime or path.stem in active_for_runtime

    def _enforce_per_runtime(self, files):
        remaining = []
        total_result = CleanupResult()
        by_runtime = {}
        for item in files:
            by_runtime.setdefault(item.runtime, []).append(item)
        for runtime, items in by_runtime.items():
            items = list(items)
            size = sum(item.size for item in items)
            if size <= self.max_per_runtime_bytes:
                remaining.extend(items)
                continue
            items.sort(key=lambda item: item.mtime)
            to_delete = []
            while items and size > self.max_per_runtime_bytes:
                item = items.pop(0)
                size -= item.size
                to_delete.append(item)
            total_result += self._delete_files(to_delete, reason=f"runtime-size:{runtime}")
            remaining.extend(items)
        return remaining, total_result

    def _enforce_total(self, files):
        files = list(files)
        size = sum(item.size for item in files)
        if size <= self.max_total_bytes:
            return files, CleanupResult()
        files.sort(key=lambda item: item.mtime)
        to_delete = []
        while files and size > self.max_total_bytes:
            item = files.pop(0)
            size -= item.size
            to_delete.append(item)
        return files, self._delete_files(to_delete, reason="total-size")

    def _delete_files(self, files, reason):
        result = CleanupResult()
        for item in files:
            try:
                item.path.unlink()
            except FileNotFoundError:
                continue
            except OSError as exc:
                log_event("kv cleanup delete failed", path=item.path, reason=reason, error=exc)
                continue
            result.deleted += 1
            result.bytes_deleted += item.size
        if result.deleted:
            log_event("kv cleanup deleted", reason=reason, count=result.deleted, bytes=result.bytes_deleted)
        return result

    def _remove_empty_dirs(self):
        root = _safe_resolve(self.cache_dir)
        if root is None or not root.is_dir():
            return
        dirs = [path for path in root.rglob("*") if path.is_dir()]
        for path in sorted(dirs, key=lambda p: len(p.parts), reverse=True):
            resolved = _safe_resolve(path)
            if resolved is None or not _is_within(resolved, root):
                continue
            try:
                path.rmdir()
            except OSError:
                pass


class CacheFile:
    def __init__(self, runtime, path, size, mtime):
        self.runtime = runtime
        self.path = path
        self.size = size
        self.mtime = mtime

    def __hash__(self):
        return hash(self.path)

    def __eq__(self, other):
        return isinstance(other, CacheFile) and self.path == other.path


class CleanupResult:
    def __init__(self, scanned=0, deleted=0, bytes_deleted=0):
        self.scanned = scanned
        self.deleted = deleted
        self.bytes_deleted = bytes_deleted

    def __add__(self, other):
        return CleanupResult(
            scanned=self.scanned + other.scanned,
            deleted=self.deleted + other.deleted,
            bytes_deleted=self.bytes_deleted + other.bytes_deleted,
        )

    def __iadd__(self, other):
        self.scanned += other.scanned
        self.deleted += other.deleted
        self.bytes_deleted += other.bytes_deleted
        return self


def _runtime_name(root, path):
    try:
        relative = path.relative_to(root)
    except ValueError:
        return "_root"
    return relative.parts[0] if len(relative.parts) > 1 else "_root"


def _optional_float(value):
    if value in (None, ""):
        return None
    return float(value)


def _gb_to_bytes(value):
    if value in (None, ""):
        return None
    parsed = float(value)
    if parsed <= 0:
        return None
    return int(parsed * 1024 * 1024 * 1024)


def _safe_resolve(path):
    try:
        return Path(path).expanduser().resolve()
    except OSError:
        return None


def _is_within(path, root):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
