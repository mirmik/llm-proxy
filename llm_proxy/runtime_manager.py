"""Unified runtime manager with pluggable drivers."""

import asyncio
import copy
import json
import os
import signal
from collections import deque

import aiohttp

from llm_proxy.config import load_config, load_config_file, save_config_file
from llm_proxy.drivers import build_driver_registry


class RuntimeManager:
    """Manages all configured runtimes through driver modules."""

    def __init__(self, config, coordinator=None):
        self.config = config
        self.coordinator = coordinator
        self.drivers = build_driver_registry()
        self.state_dir = os.path.expanduser(config["state_dir"])
        self._state_path = os.path.join(self.state_dir, "runtime_state.json")
        self.runtimes = self._build_runtimes(config)
        self._profiles = {}
        self._processes = {}
        self._logs = {}
        self._meta = {}
        self._load_state()

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

    def current_profile(self, runtime_name):
        runtime = self.runtimes[runtime_name]
        if runtime_name in self._profiles:
            return self._profiles[runtime_name]
        if runtime.get("default_profile"):
            return runtime["default_profile"]
        profiles = runtime.get("profiles") or {}
        if profiles:
            return next(iter(profiles))
        return None

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
        runtime = self.runtimes[runtime_name]
        profile = self.resolve_profile(runtime_name)
        return int(profile.get("slots", 1))

    def runtime_state(self, runtime_name):
        runtime = self.runtimes[runtime_name]
        driver = self.driver_for(runtime_name)
        profile = self.resolve_profile(runtime_name)
        meta = dict(self._meta.get(runtime_name) or {})
        return {
            "name": runtime_name,
            "driver": runtime["driver"],
            "kind": driver.kind(),
            "status": "running" if self.is_running(runtime_name) else "stopped",
            "profile": self.current_profile(runtime_name),
            "profiles": list((runtime.get("profiles") or {}).keys()),
            "autoload": bool(runtime.get("autoload", False)),
            "gpu": profile.get("gpu"),
            "health_url": driver.resolve_health_url(runtime_name, profile, self.config),
            "pid": self._processes.get(runtime_name).pid if self.is_running(runtime_name) else None,
            "settings": self._editable_settings(runtime_name),
            **meta,
        }

    def list_state(self):
        return [self.runtime_state(name) for name in self.available_runtimes()]

    def get_logs(self, runtime_name, limit=200):
        return list(self._logs.get(runtime_name, deque(maxlen=200)))[-limit:]

    def list_running_on_gpu(self, gpu):
        running = []
        for name in self.available_runtimes():
            if not self.is_running(name):
                continue
            profile = self.resolve_profile(name)
            if profile.get("gpu") == gpu:
                running.append(name)
        return running

    async def autostart(self):
        for name, runtime in self.runtimes.items():
            if runtime.get("autoload", False):
                try:
                    await self.start(name)
                except Exception as e:
                    self._append_log(name, f"[WARN] Autostart failed: {e}")
                    print(f"  [WARN] Failed to autoload runtime {name}: {e}")

    async def start(self, runtime_name):
        if self.is_running(runtime_name):
            return
        if runtime_name not in self.runtimes:
            raise ValueError(f"Unknown runtime: {runtime_name}")

        runtime = self.runtimes[runtime_name]
        driver = self.driver_for(runtime_name)
        profile = self.resolve_profile(runtime_name)
        driver.validate_profile(profile)

        argv = driver.build_command(runtime_name, profile, self.config)
        env = os.environ.copy()
        env.update(driver.build_env(runtime_name, profile, self.config))
        self._logs.setdefault(runtime_name, deque(maxlen=200))
        self._append_log(runtime_name, f"$ {' '.join(argv)}")
        proc = await asyncio.create_subprocess_exec(
            *argv,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        self._processes[runtime_name] = proc
        self._meta[runtime_name] = driver.runtime_meta(runtime_name, profile, self.config)
        asyncio.create_task(self._log_output(runtime_name, proc))

        health_url = driver.resolve_health_url(runtime_name, profile, self.config)
        if health_url:
            timeout = float(driver.startup_timeout(runtime_name, profile))
            if not await self._wait_healthy(health_url, timeout=timeout):
                await self.stop(runtime_name)
                raise TimeoutError(f"{runtime_name} failed to start within {timeout}s")
        self._append_log(runtime_name, "[ready]")

    async def stop(self, runtime_name):
        proc = self._processes.get(runtime_name)
        if proc is None or proc.returncode is not None:
            self._processes.pop(runtime_name, None)
            self._meta.pop(runtime_name, None)
            return

        driver = self.driver_for(runtime_name)
        profile = self.resolve_profile(runtime_name)
        stop_signal = str(driver.stop_signal(runtime_name, profile)).upper()
        sig = getattr(signal, stop_signal, signal.SIGTERM)

        self._append_log(runtime_name, f"[stopping] signal={stop_signal}")
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

    async def stop_all(self):
        for name in list(self._processes):
            await self.stop(name)

    async def ensure(self, runtime_name):
        if not self.is_running(runtime_name):
            await self.start(runtime_name)
        return self.get_port(runtime_name)

    async def select_profile(self, runtime_name, profile_name):
        runtime = self.runtimes.get(runtime_name)
        if not runtime:
            raise ValueError(f"Unknown runtime: {runtime_name}")
        profiles = runtime.get("profiles") or {}
        if profile_name not in profiles:
            raise ValueError(f"Unknown profile for {runtime_name}: {profile_name}")
        self._profiles[runtime_name] = profile_name
        self._save_state()
        if self.is_running(runtime_name):
            await self.stop(runtime_name)
            await self.start(runtime_name)

    async def update_settings(self, runtime_name, settings):
        runtime = self.runtimes.get(runtime_name)
        if not runtime:
            raise ValueError(f"Unknown runtime: {runtime_name}")

        config_path = self.config.get("_config_path")
        if not config_path:
            raise ValueError("No writable config path is associated with the running server")

        current_profile = self.current_profile(runtime_name)
        if not current_profile:
            raise ValueError(f"Runtime {runtime_name} has no active profile")

        raw = load_config_file(config_path)
        runtimes = raw.setdefault("runtimes", {})
        raw_runtime = runtimes.get(runtime_name)
        if not raw_runtime:
            raise ValueError(f"Runtime {runtime_name} is missing from config file")
        raw_profiles = raw_runtime.setdefault("profiles", {})
        raw_profile = raw_profiles.setdefault(current_profile, {})

        if "autoload" in settings:
            raw_runtime["autoload"] = bool(settings["autoload"])

        for key, value in settings.items():
            if key == "autoload":
                continue
            if value in ("", None):
                raw_profile.pop(key, None)
                continue
            if key in {"port", "context", "slots", "sample_rate"}:
                raw_profile[key] = int(value)
            else:
                raw_profile[key] = value

        save_config_file(config_path, raw)
        running = self.is_running(runtime_name)
        if running:
            await self.stop(runtime_name)
        self._reload_from_disk()
        if running:
            await self.start(runtime_name)

    def resolve_profile(self, runtime_name):
        runtime = self.runtimes[runtime_name]
        profile_name = self.current_profile(runtime_name)
        base = copy.deepcopy(runtime.get("base", {}))
        profile = copy.deepcopy((runtime.get("profiles") or {}).get(profile_name, {}))
        merged = base
        merged.update(profile)
        if "gpu" not in merged and runtime.get("gpu"):
            merged["gpu"] = runtime["gpu"]
        return merged

    def _editable_settings(self, runtime_name):
        runtime = self.runtimes[runtime_name]
        profile = self.resolve_profile(runtime_name)
        current_profile = self.current_profile(runtime_name)
        return {
            "autoload": bool(runtime.get("autoload", False)),
            "profile_name": current_profile,
            "port": profile.get("port"),
            "context": profile.get("context"),
            "slots": profile.get("slots"),
            "model_size": profile.get("model_size"),
            "language": profile.get("language"),
            "venv": profile.get("venv"),
            "python_bin": profile.get("python_bin"),
            "reference_wav": profile.get("reference_wav"),
        }

    def _build_runtimes(self, config):
        runtimes = copy.deepcopy(config.get("runtimes") or {})
        if runtimes:
            return runtimes

        # Legacy compatibility: map old config["models"] to runtimes.
        for name, model in (config.get("models") or {}).items():
            runtimes[name] = {
                "driver": "llama_cpp",
                "gpu": model.get("gpu"),
                "autoload": model.get("autoload", False),
                "default_profile": "default",
                "profiles": {
                    "default": dict(model),
                },
            }
        return runtimes

    def _reload_from_disk(self):
        config_path = self.config.get("_config_path")
        reloaded = load_config(config_path)
        self.config = reloaded
        self.runtimes = self._build_runtimes(reloaded)

    def _load_state(self):
        if not os.path.isfile(self._state_path):
            return
        try:
            with open(self._state_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            return
        self._profiles = payload.get("profiles") or {}

    def _save_state(self):
        os.makedirs(self.state_dir, exist_ok=True)
        tmp = self._state_path + ".tmp"
        payload = {"profiles": self._profiles}
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2, sort_keys=True)
        os.replace(tmp, self._state_path)

    def _append_log(self, runtime_name, line):
        self._logs.setdefault(runtime_name, deque(maxlen=200)).append(line)
        print(f"  [rt:{runtime_name}] {line}")

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
