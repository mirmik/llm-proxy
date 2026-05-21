"""Supervisor for non-LLM local services such as STT/TTS."""

import asyncio
import json
import os
import shlex
import signal
from collections import deque

import aiohttp

from llm_proxy.oplog import elapsed_ms, log_event, now


class ServiceManager:
    """Manages command-based local services with profile switching."""

    def __init__(self, config, coordinator=None):
        self.config = config
        self.services = config.get("services", {})
        self.state_dir = os.path.expanduser(config["state_dir"])
        self.coordinator = coordinator
        self._processes = {}
        self._profiles = {}
        self._logs = {}
        self._state_path = os.path.join(self.state_dir, "control_state.json")
        self._load_state()

    def available_services(self):
        return list(self.services.keys())

    def is_running(self, service_name):
        proc = self._processes.get(service_name)
        return proc is not None and proc.returncode is None

    def loaded_services(self):
        return [name for name in self.available_services() if self.is_running(name)]

    def current_profile(self, service_name):
        cfg = self.services.get(service_name) or {}
        default_profile = cfg.get("default_profile")
        profiles = cfg.get("profiles") or {}
        if service_name in self._profiles:
            return self._profiles[service_name]
        if default_profile:
            return default_profile
        if profiles:
            return next(iter(profiles))
        return None

    def service_state(self, service_name):
        cfg = self.services[service_name]
        effective = self._resolve(service_name)
        return {
            "name": service_name,
            "kind": cfg.get("kind", "service"),
            "status": "running" if self.is_running(service_name) else "stopped",
            "profile": self.current_profile(service_name),
            "profiles": list((cfg.get("profiles") or {}).keys()),
            "health_url": effective.get("health_url"),
            "workdir": effective.get("workdir"),
            "gpu": effective.get("gpu"),
            "autostart": bool(cfg.get("autostart", False)),
            "pid": self._processes.get(service_name).pid if self.is_running(service_name) else None,
        }

    def list_state(self):
        return [self.service_state(name) for name in self.available_services()]

    def get_logs(self, service_name, limit=200):
        return list(self._logs.get(service_name, deque(maxlen=200)))[-limit:]

    def list_running_on_gpu(self, gpu):
        running = []
        for name in self.available_services():
            if not self.is_running(name):
                continue
            effective = self._resolve(name)
            if effective.get("gpu") == gpu:
                running.append(name)
        return running

    async def autostart(self):
        for name, cfg in self.services.items():
            if cfg.get("autostart", False):
                try:
                    await self.start(name)
                except Exception as e:
                    self._append_log(name, f"[WARN] Autostart failed: {e}")
                    log_event("service autostart failed", service=name, error=e)

    async def start(self, service_name):
        if self.is_running(service_name):
            log_event("service start skipped", service=service_name, reason="already-running")
            return
        if service_name not in self.services:
            raise ValueError(f"Unknown service: {service_name}")

        started = now()
        effective = self._resolve(service_name)
        command = effective.get("command")
        if not command:
            raise ValueError(f"Service {service_name} has no command configured")

        if self.coordinator:
            await self.coordinator.prepare("service", service_name, effective.get("gpu"))

        argv = self._command_argv(command)
        workdir = os.path.expanduser(effective.get("workdir") or os.getcwd())
        env = os.environ.copy()
        for key, value in (effective.get("env") or {}).items():
            env[str(key)] = os.path.expanduser(str(value))

        os.makedirs(workdir, exist_ok=True)
        self._logs.setdefault(service_name, deque(maxlen=200))
        self._append_log(service_name, f"$ {' '.join(argv)}")
        log_event(
            "service start",
            service=service_name,
            port=effective.get("port"),
            workdir=workdir,
        )
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=workdir,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        self._processes[service_name] = proc
        asyncio.create_task(self._log_output(service_name, proc))

        health_url = effective.get("health_url")
        timeout = float(effective.get("start_timeout", 120))
        if health_url and not await self._wait_healthy(health_url, timeout=timeout):
            await self.stop(service_name)
            raise TimeoutError(f"{service_name} failed to start within {timeout}s")

        self._append_log(service_name, "[ready]")
        log_event("service ready", service=service_name, ms=elapsed_ms(started), pid=proc.pid)

    async def stop(self, service_name):
        proc = self._processes.get(service_name)
        if proc is None or proc.returncode is not None:
            self._processes.pop(service_name, None)
            log_event("service stop skipped", service=service_name, reason="not-running")
            return

        started = now()
        effective = self._resolve(service_name)
        stop_signal = str(effective.get("stop_signal", "SIGTERM")).upper()
        sig = getattr(signal, stop_signal, signal.SIGTERM)
        timeout = float(effective.get("stop_timeout", 10))

        self._append_log(service_name, f"[stopping] signal={stop_signal}")
        log_event("service stop", service=service_name, signal=stop_signal, pid=proc.pid)
        proc.send_signal(sig)
        try:
            await asyncio.wait_for(proc.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            self._append_log(service_name, "[kill]")
            proc.kill()
            await proc.wait()
        self._processes.pop(service_name, None)
        self._append_log(service_name, "[stopped]")
        log_event("service stopped", service=service_name, ms=elapsed_ms(started), code=proc.returncode)

    async def restart(self, service_name):
        await self.stop(service_name)
        await self.start(service_name)

    async def select_profile(self, service_name, profile_name):
        cfg = self.services.get(service_name)
        if not cfg:
            raise ValueError(f"Unknown service: {service_name}")
        profiles = cfg.get("profiles") or {}
        if profile_name not in profiles:
            raise ValueError(f"Unknown profile for {service_name}: {profile_name}")
        self._profiles[service_name] = profile_name
        self._save_state()
        if self.is_running(service_name):
            await self.restart(service_name)

    async def stop_all(self):
        for name in list(self._processes):
            await self.stop(name)

    def _resolve(self, service_name):
        cfg = json.loads(json.dumps(self.services[service_name]))
        profile_name = self.current_profile(service_name)
        profiles = cfg.pop("profiles", {}) or {}
        profile_cfg = profiles.get(profile_name, {}) if profile_name else {}
        return _deep_merge(cfg, profile_cfg)

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

    def _append_log(self, service_name, line):
        self._logs.setdefault(service_name, deque(maxlen=200)).append(line)

    def _command_argv(self, command):
        if isinstance(command, str):
            return shlex.split(command)
        return [str(part) for part in command]

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

    async def _log_output(self, service_name, proc):
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="replace").rstrip()
            self._append_log(service_name, text)


def _deep_merge(base, override):
    if not isinstance(base, dict) or not isinstance(override, dict):
        return override
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
