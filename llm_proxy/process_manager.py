"""Process Manager — start/stop llama-server instances per model."""

import asyncio
import glob
import os
import signal

import aiohttp


class ProcessManager:
    """Manages llama-server processes, one per model."""

    def __init__(self, config, coordinator=None):
        self.config = config
        self.llama_bin = config["llama_server"]
        self.models = config["models"]
        self.gpu_groups = config.get("gpu_groups", {})
        self.coordinator = coordinator
        self._processes = {}  # model_name → asyncio.subprocess.Process
        self._ports = {}      # model_name → port

    def is_running(self, model_name):
        proc = self._processes.get(model_name)
        return proc is not None and proc.returncode is None

    def get_port(self, model_name):
        return self._ports.get(model_name)

    def loaded_models(self):
        return [name for name in self._processes if self.is_running(name)]

    def available_models(self):
        return list(self.models.keys())

    def list_running_on_gpu(self, gpu):
        running = []
        for name, cfg in self.models.items():
            if self.is_running(name) and cfg.get("gpu") == gpu:
                running.append(name)
        return running

    async def start(self, model_name):
        """Start llama-server for the given model. Returns when server is healthy."""
        if self.is_running(model_name):
            return

        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")

        mcfg = self.models[model_name]

        # Handle GPU conflicts — stop models sharing the same GPU group
        gpu = mcfg.get("gpu")
        if self.coordinator:
            await self.coordinator.prepare("model", model_name, gpu)
        elif gpu:
            for other_name, other_cfg in self.models.items():
                if other_name != model_name and other_cfg.get("gpu") == gpu and self.is_running(other_name):
                    print(f"  [PM] Stopping {other_name} (GPU conflict: {gpu})")
                    await self.stop(other_name)

        port = mcfg.get("port", 8081)
        slots = mcfg.get("slots", 1)
        context = mcfg.get("context", 32768)
        gguf = mcfg["gguf"]

        # Resolve glob pattern in gguf path
        gguf_files = sorted(glob.glob(gguf))
        if not gguf_files:
            raise FileNotFoundError(f"No GGUF files matching: {gguf}")
        model_path = gguf_files[0]

        cmd = [
            self.llama_bin,
            "--model", model_path,
            "-ngl", "99",
            "-c", str(context),
            "-np", str(slots),
            "--host", "127.0.0.1",
            "--port", str(port),
        ]

        # Slot save/restore support
        cache_dir = self.config["cache_dir"]
        model_cache = os.path.join(cache_dir, model_name)
        os.makedirs(model_cache, exist_ok=True)
        cmd += ["--slot-save-path", model_cache]

        # Extra args from config
        extra = mcfg.get("extra_args", "")
        if extra:
            cmd += extra.split()

        print(f"  [PM] Starting {model_name} on :{port} (slots={slots}, ctx={context})")
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        self._processes[model_name] = proc
        self._ports[model_name] = port

        # Stream server output in background
        asyncio.create_task(self._log_output(model_name, proc))

        # Wait for health
        if not await self._wait_healthy(port, timeout=120):
            raise TimeoutError(f"{model_name} failed to start within 120s")

        print(f"  [PM] {model_name} is ready on :{port}")

    async def stop(self, model_name):
        """Stop a running model."""
        proc = self._processes.get(model_name)
        if proc is None or proc.returncode is not None:
            self._processes.pop(model_name, None)
            self._ports.pop(model_name, None)
            return

        print(f"  [PM] Stopping {model_name}...")
        proc.send_signal(signal.SIGTERM)
        try:
            await asyncio.wait_for(proc.wait(), timeout=10)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()

        self._processes.pop(model_name, None)
        self._ports.pop(model_name, None)
        print(f"  [PM] {model_name} stopped")

    async def stop_all(self):
        """Stop all running models."""
        for name in list(self._processes):
            await self.stop(name)

    async def ensure(self, model_name):
        """Make sure model is running, start if needed."""
        if not self.is_running(model_name):
            await self.start(model_name)
        return self._ports[model_name]

    async def _wait_healthy(self, port, timeout=120):
        """Poll /health until ok or timeout."""
        url = f"http://127.0.0.1:{port}/health"
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

    async def _log_output(self, model_name, proc):
        """Read and print server stdout/stderr."""
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="replace").rstrip()
            print(f"  [{model_name}] {text}")
