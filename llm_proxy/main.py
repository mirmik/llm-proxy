"""Entry point for llm-proxy."""

import argparse
import asyncio
import signal

from aiohttp import web

from llm_proxy.config import load_config
from llm_proxy.process_manager import ProcessManager
from llm_proxy.slot_manager import SlotManager
from llm_proxy.router import create_app


async def run(config):
    pm = ProcessManager(config)
    sm = SlotManager()

    # Autoload models marked with autoload: true
    for name, mcfg in config["models"].items():
        if mcfg.get("autoload", False):
            try:
                await pm.start(name)
            except Exception as e:
                print(f"  [WARN] Failed to autoload {name}: {e}")

    app = create_app(pm, sm)

    listen = config["listen"]
    host = listen["host"] if isinstance(listen, dict) else "0.0.0.0"
    port = listen["port"] if isinstance(listen, dict) else listen

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    print(f"  LLM Proxy listening on {host}:{port}")
    print(f"  Models: {pm.available_models()}")
    print(f"  Loaded: {pm.loaded_models()}")

    # Wait for shutdown signal
    stop = asyncio.Event()
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)

    await stop.wait()
    print("\n  Shutting down...")

    # Save all slot caches before stopping
    for name in pm.loaded_models():
        port = pm.get_port(name)
        if port:
            await sm.save_all(name, port)

    await pm.stop_all()
    await runner.cleanup()


def main():
    parser = argparse.ArgumentParser(description="LLM Proxy")
    parser.add_argument("-c", "--config", help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    asyncio.run(run(config))


if __name__ == "__main__":
    main()
