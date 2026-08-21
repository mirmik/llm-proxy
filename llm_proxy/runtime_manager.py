"""Unified runtime manager with pluggable drivers."""

import asyncio
import copy
import os
import re
import signal
from collections import deque

import aiohttp

from llm_proxy.oplog import elapsed_ms, log_event, now


_VALID_NAME = re.compile(r"^(?!\.)[a-zA-Z0-9_.\-]+$")

from llm_proxy.config import load_config, load_config_file, save_config_file
from llm_proxy.drivers import build_driver_registry


class RuntimeManager:
    """Manages all configured runtimes through driver modules."""

    def __init__(self, config, coordinator=None):
        self.config = config
        self.coordinator = coordinator
        self.drivers = build_driver_registry()
        self.state_dir = os.path.expanduser(config["state_dir"])
        self.runtimes = self._build_runtimes(config)
        self._processes = {}
        self._logs = {}
        self._meta = {}

    def available_runtimes(self):
        return list(self.runtimes.keys())

    def available_models(self):
        return [name for name, rt in self.runtimes.items() if self.driver_for(name).kind() == "llm"]

    def runtimes_by_kind(self, kind):
        return [name for name in self.available_runtimes() if self.driver_for(name).kind() == kind]

    def loaded_runtimes(self):
        return [name for name in self.available_runtimes() if self.is_running(name)]

    def loaded_models(self):
        return [name for name in self.available_models() if self.is_running(name)]

    def is_running(self, runtime_name):
        proc = self._processes.get(runtime_name)
        return proc is not None and proc.returncode is None

    def driver_for(self, runtime_name):
        driver_name = self.runtimes[runtime_name]["driver"]
        driver = self.drivers.get(driver_name)
        if not driver:
            raise ValueError(f"Unknown driver: {driver_name}")
        return driver

    def get_port(self, runtime_name):
        meta = self._meta.get(runtime_name) or {}
        return meta.get("port")

    def get_slots(self, runtime_name):
        runtime = self.runtime_config(runtime_name)
        return int(runtime.get("slots", 1))

    def runtime_state(self, runtime_name):
        runtime = self.runtimes[runtime_name]
        driver = self.driver_for(runtime_name)
        runtime_cfg = self.runtime_config(runtime_name)
        meta = dict(self._meta.get(runtime_name) or {})
        return {
            "name": runtime_name,
            "driver": runtime["driver"],
            "kind": driver.kind(),
            "status": "running" if self.is_running(runtime_name) else "stopped",
            "autoload": bool(runtime.get("autoload", False)),
            "on_demand": self.on_demand(runtime_name),
            "gpu": runtime_cfg.get("gpu"),
            "health_url": driver.resolve_health_url(runtime_name, runtime_cfg, self.config),
            "pid": self._processes.get(runtime_name).pid if self.is_running(runtime_name) else None,
            "settings": self._editable_settings(runtime_name),
            **meta,
        }

    def on_demand(self, runtime_name):
        runtime = self.runtimes.get(runtime_name) or {}
        return bool(runtime.get("on_demand", False))

    def list_state(self):
        return [self.runtime_state(name) for name in self.available_runtimes()]

    def get_logs(self, runtime_name, limit=200):
        return list(self._logs.get(runtime_name, deque(maxlen=200)))[-limit:]

    def list_running_on_gpu(self, gpu):
        running = []
        for name in self.available_runtimes():
            if not self.is_running(name):
                continue
            runtime_cfg = self.runtime_config(name)
            if runtime_cfg.get("gpu") == gpu:
                running.append(name)
        return running

    async def autostart(self):
        for name, runtime in self.runtimes.items():
            if runtime.get("autoload", False):
                try:
                    await self.start(name)
                except Exception as e:
                    self._append_log(name, f"[WARN] Autostart failed: {e}")
                    log_event("runtime autostart failed", runtime=name, error=e)

    async def start(self, runtime_name):
        if self.is_running(runtime_name):
            log_event("runtime start skipped", runtime=runtime_name, reason="already-running")
            return
        if runtime_name not in self.runtimes:
            raise ValueError(f"Unknown runtime: {runtime_name}")

        started = now()
        runtime = self.runtimes[runtime_name]
        driver = self.driver_for(runtime_name)
        runtime_cfg = self.runtime_config(runtime_name)
        driver.validate_runtime(runtime_cfg)

        argv = driver.build_command(runtime_name, runtime_cfg, self.config)
        env = os.environ.copy()
        env.update(driver.build_env(runtime_name, runtime_cfg, self.config))
        self._logs.setdefault(runtime_name, deque(maxlen=200))
        self._append_log(runtime_name, f"$ {' '.join(argv)}")
        log_event(
            "runtime start",
            runtime=runtime_name,
            driver=runtime["driver"],
            port=runtime_cfg.get("port"),
        )
        proc = await asyncio.create_subprocess_exec(
            *argv,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        self._processes[runtime_name] = proc
        self._meta[runtime_name] = driver.runtime_meta(runtime_name, runtime_cfg, self.config)
        asyncio.create_task(self._log_output(runtime_name, proc))

        health_url = driver.resolve_health_url(runtime_name, runtime_cfg, self.config)
        if health_url:
            timeout = float(driver.startup_timeout(runtime_name, runtime_cfg))
            if not await self._wait_healthy(health_url, timeout=timeout):
                await self.stop(runtime_name)
                raise TimeoutError(f"{runtime_name} failed to start within {timeout}s")
        self._append_log(runtime_name, "[ready]")
        log_event("runtime ready", runtime=runtime_name, ms=elapsed_ms(started), pid=proc.pid)

    async def stop(self, runtime_name):
        proc = self._processes.get(runtime_name)
        if proc is None or proc.returncode is not None:
            self._processes.pop(runtime_name, None)
            self._meta.pop(runtime_name, None)
            log_event("runtime stop skipped", runtime=runtime_name, reason="not-running")
            return

        started = now()
        driver = self.driver_for(runtime_name)
        runtime_cfg = self.runtime_config(runtime_name)
        stop_signal = str(driver.stop_signal(runtime_name, runtime_cfg)).upper()
        sig = getattr(signal, stop_signal, signal.SIGTERM)

        self._append_log(runtime_name, f"[stopping] signal={stop_signal}")
        log_event("runtime stop", runtime=runtime_name, signal=stop_signal, pid=proc.pid)
        proc.send_signal(sig)
        try:
            await asyncio.wait_for(proc.wait(), timeout=10)
        except asyncio.TimeoutError:
            self._append_log(runtime_name, "[kill]")
            proc.kill()
            await proc.wait()
        self._processes.pop(runtime_name, None)
        self._meta.pop(runtime_name, None)
        self._append_log(runtime_name, "[stopped]")
        log_event("runtime stopped", runtime=runtime_name, ms=elapsed_ms(started), code=proc.returncode)

    async def stop_all(self):
        for name in list(self._processes):
            await self.stop(name)

    async def ensure(self, runtime_name):
        if not self.is_running(runtime_name):
            await self.start(runtime_name)
        return self.get_port(runtime_name)

    async def create_runtime(self, spec):
        """Create a new runtime and persist it to the config file.

        spec is a dict with keys:
          name (required)         — runtime identifier, unique
          driver (required)       — llama_cpp / llama_cpp_embedding / faster_whisper / xtts
          gpu (optional)
          autoload (optional)
          on_demand (optional)
          port, command, slots?, venv?, python_bin?, env?, ...
        """
        config_path = self.config.get("_config_path")
        if not config_path:
            raise ValueError("No writable config path is associated with the running server")

        name = str(spec.get("name") or "").strip()
        if not name:
            raise ValueError("'name' is required")
        if not _VALID_NAME.match(name):
            raise ValueError("'name' must be [a-zA-Z0-9_.-]+ and cannot start with a dot")

        driver_name = str(spec.get("driver") or "").strip()
        if not driver_name:
            raise ValueError("'driver' is required")
        if driver_name not in self.drivers:
            raise ValueError(f"Unknown driver: {driver_name!r}. Available: {sorted(self.drivers)}")

        runtime_fields = _runtime_fields_from_spec(spec)
        if not runtime_fields.get("command"):
            raise ValueError("command is required")
        if not runtime_fields.get("port"):
            raise ValueError("port is required")
        try:
            port = int(runtime_fields["port"])
        except (TypeError, ValueError):
            raise ValueError("port must be an integer")
        runtime_fields["port"] = port

        raw = load_config_file(config_path)
        runtimes = raw.setdefault("runtimes", {}) or {}
        raw["runtimes"] = runtimes
        if name in runtimes:
            raise ValueError(f"Runtime {name!r} already exists")

        for other_name, other in runtimes.items():
            other_runtime = _collapse_legacy_runtime(other)
            if int(other_runtime.get("port", 0)) == port:
                raise ValueError(
                    f"Port {port} is already used by runtime {other_name!r}. "
                    f"Choose another port."
                )

        entry = {"driver": driver_name}
        if spec.get("gpu") not in (None, ""):
            entry["gpu"] = str(spec["gpu"])
        if "autoload" in spec:
            entry["autoload"] = bool(spec["autoload"])
        if "on_demand" in spec:
            entry["on_demand"] = bool(spec["on_demand"])
        entry.update(runtime_fields)
        runtimes[name] = entry

        save_config_file(config_path, raw)
        self._reload_from_disk()

    async def delete_runtime(self, runtime_name):
        """Stop the runtime (if running) and remove it from the config file."""
        config_path = self.config.get("_config_path")
        if not config_path:
            raise ValueError("No writable config path is associated with the running server")

        if runtime_name not in self.runtimes:
            raise ValueError(f"Unknown runtime: {runtime_name!r}")

        if self.is_running(runtime_name):
            await self.stop(runtime_name)

        raw = load_config_file(config_path)
        runtimes = raw.setdefault("runtimes", {}) or {}
        runtimes.pop(runtime_name, None)
        raw["runtimes"] = runtimes
        save_config_file(config_path, raw)

        self._logs.pop(runtime_name, None)
        self._reload_from_disk()

    async def update_settings(self, runtime_name, settings):
        runtime = self.runtimes.get(runtime_name)
        if not runtime:
            raise ValueError(f"Unknown runtime: {runtime_name}")

        config_path = self.config.get("_config_path")
        if not config_path:
            raise ValueError("No writable config path is associated with the running server")

        raw = load_config_file(config_path)
        runtimes = raw.setdefault("runtimes", {})
        raw_runtime = runtimes.get(runtime_name)
        if not raw_runtime:
            raise ValueError(f"Runtime {runtime_name} is missing from config file")
        raw_runtime = _collapse_legacy_runtime(raw_runtime)

        if "autoload" in settings:
            raw_runtime["autoload"] = bool(settings["autoload"])
        if "on_demand" in settings:
            raw_runtime["on_demand"] = bool(settings["on_demand"])

        for key, value in settings.items():
            if key in ("autoload", "on_demand"):
                continue
            if key == "env":
                env_dict = _parse_env(value)
                if env_dict:
                    raw_runtime["env"] = env_dict
                else:
                    raw_runtime.pop("env", None)
                continue
            if value in ("", None):
                raw_runtime.pop(key, None)
                continue
            if key in {"port", "slots", "sample_rate"}:
                raw_runtime[key] = int(value)
            elif key == "startup_timeout":
                raw_runtime[key] = float(value)
            else:
                raw_runtime[key] = value

        runtimes[runtime_name] = raw_runtime

        save_config_file(config_path, raw)
        running = self.is_running(runtime_name)
        if running:
            await self.stop(runtime_name)
        self._reload_from_disk()
        if running:
            await self.start(runtime_name)

    def runtime_config(self, runtime_name):
        runtime = self.runtimes[runtime_name]
        return _collapse_legacy_runtime(runtime)

    def _editable_settings(self, runtime_name):
        runtime = self.runtimes[runtime_name]
        driver = self.driver_for(runtime_name)
        runtime_cfg = self.runtime_config(runtime_name)
        return {
            "autoload": bool(runtime.get("autoload", False)),
            "on_demand": bool(runtime.get("on_demand", False)),
            "kind": driver.kind(),
            "port": runtime_cfg.get("port"),
            "slots": runtime_cfg.get("slots"),
            "venv": runtime_cfg.get("venv"),
            "python_bin": runtime_cfg.get("python_bin"),
            "command": runtime_cfg.get("command"),
            "env": runtime_cfg.get("env") or {},
            "startup_timeout": runtime_cfg.get("startup_timeout"),
        }

    def _build_runtimes(self, config):
        return copy.deepcopy(config.get("runtimes") or {})

    def _reload_from_disk(self):
        config_path = self.config.get("_config_path")
        reloaded = load_config(config_path)
        self.config = reloaded
        self.runtimes = self._build_runtimes(reloaded)

    def _append_log(self, runtime_name, line):
        self._logs.setdefault(runtime_name, deque(maxlen=200)).append(line)

    async def _wait_healthy(self, url, timeout=120):
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            try:
                async with aiohttp.ClientSession() as sess:
                    async with sess.get(url, timeout=aiohttp.ClientTimeout(total=2)) as resp:
                        if resp.status == 200:
                            return True
            except (aiohttp.ClientError, asyncio.TimeoutError):
                pass
            await asyncio.sleep(0.5)
        return False

    async def _log_output(self, runtime_name, proc):
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="replace").rstrip()
            self._append_log(runtime_name, text)
        if proc.returncode not in (None, 0):
            self._append_log(runtime_name, f"[exit] code={proc.returncode}")
            log_event("runtime exited", runtime=runtime_name, code=proc.returncode)


def _parse_env(value):
    """Accept env as dict OR 'KEY=value' lines. Returns a dict of str->str.

    Lines starting with '#' are treated as comments. Blank lines are ignored.
    """
    if value is None or value == "":
        return {}
    if isinstance(value, dict):
        return {str(k): str(v) for k, v in value.items() if v not in (None, "")}
    out = {}
    for raw in str(value).splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"Invalid env line (expected KEY=value): {raw!r}")
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip()
        if not key:
            raise ValueError(f"Empty env key in line: {raw!r}")
        out[key] = val
    return out


def _runtime_fields_from_spec(spec):
    legacy = spec.get("profile")
    if isinstance(legacy, dict):
        out = copy.deepcopy(legacy)
        out.pop("name", None)
    else:
        out = {}
    for key, value in spec.items():
        if key in {"name", "driver", "gpu", "autoload", "on_demand", "profile", "profile_name", "kind"}:
            continue
        out[key] = value
    return out


def _collapse_legacy_runtime(runtime):
    if not isinstance(runtime, dict) or "profiles" not in runtime:
        return copy.deepcopy(runtime)

    profiles = runtime.get("profiles") or {}
    profile_name = "default"
    if profile_name not in profiles:
        configured = runtime.get("default_profile")
        if configured in profiles:
            profile_name = configured
        elif len(profiles) == 1:
            profile_name = next(iter(profiles))

    base = runtime.get("base") or {}
    profile = profiles.get(profile_name) or {}
    collapsed = {
        key: copy.deepcopy(value)
        for key, value in runtime.items()
        if key not in {"profiles", "default_profile", "base"}
    }
    merged = copy.deepcopy(base)
    merged.update(copy.deepcopy(profile))
    collapsed.update(merged)
    return collapsed
