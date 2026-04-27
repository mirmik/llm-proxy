"""Entry point for llm-proxy."""

import argparse
import asyncio
import os
import signal

from aiohttp import web

from llm_proxy.auth import AuthManager
from llm_proxy.certs import certificate_fingerprint, generate_server_cert, make_ssl_context
from llm_proxy.config import load_config
from llm_proxy.resource_coordinator import ResourceCoordinator
from llm_proxy.route_manager import RouteManager
from llm_proxy.runtime_manager import RuntimeManager
from llm_proxy.slot_manager import SlotManager
from llm_proxy.router import create_api_app, create_control_app


async def run(config):
    coordinator = ResourceCoordinator()
    rm = RuntimeManager(config, coordinator=coordinator)
    sm = SlotManager()
    auth = AuthManager(config)
    routes = RouteManager(config)
    coordinator.register("runtime", rm.list_running_on_gpu, rm.stop)
    api_app = create_api_app(rm, sm, auth, route_manager=routes)
    control_app = create_control_app(rm, sm, auth, route_manager=routes)

    listen = config["listen"]
    api_host = listen["host"] if isinstance(listen, dict) else "0.0.0.0"
    api_port = listen["port"] if isinstance(listen, dict) else listen
    control_cfg = config.get("control") or {}
    control_host = control_cfg.get("host") or api_host
    control_port = control_cfg.get("port") or (api_port + 1)
    tls_cfg = config.get("tls") or {}
    ssl_context = None
    scheme = "http"
    if tls_cfg.get("enabled", False):
        ssl_context = make_ssl_context(tls_cfg["cert_file"], tls_cfg["key_file"])
        scheme = "https"

    api_runner = web.AppRunner(api_app)
    control_runner = web.AppRunner(control_app)
    await api_runner.setup()
    await control_runner.setup()
    api_site = web.TCPSite(api_runner, api_host, api_port, ssl_context=ssl_context)
    control_site = web.TCPSite(control_runner, control_host, control_port, ssl_context=ssl_context)
    await api_site.start()
    await control_site.start()
    print(f"  API:     {scheme}://{api_host}:{api_port}")
    print(f"  Control: {scheme}://{control_host}:{control_port}")
    print(f"  Runtimes: {rm.available_runtimes()}")
    print(f"  Loaded models: {rm.loaded_models()}")
    print(f"  API auth.enabled: {auth.enabled}; web password set: {auth.web_enabled}")
    if tls_cfg.get("enabled", False):
        print(f"  TLS fingerprint: {certificate_fingerprint(tls_cfg['cert_file'])}")
    asyncio.create_task(rm.autostart())
    await routes.start_monitoring()

    stop = asyncio.Event()
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)

    await stop.wait()
    print("\n  Shutting down...")

    for name in rm.loaded_models():
        port = rm.get_port(name)
        if port:
            await sm.save_all(name, port)

    await rm.stop_all()
    await routes.stop_monitoring()
    await api_runner.cleanup()
    await control_runner.cleanup()


def main():
    parser = argparse.ArgumentParser(description="LLM Proxy")
    parser.add_argument("-c", "--config", help="Path to config.yaml")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("serve", help="Run the proxy server")

    auth = subparsers.add_parser("auth", help="Manage API tokens")
    auth_subparsers = auth.add_subparsers(dest="auth_command", required=True)

    create = auth_subparsers.add_parser("create", help="Create a new token (plain value shown once)")
    create.add_argument("name", help="Human-readable token name (e.g. host or client name)")

    auth_subparsers.add_parser("list", help="List existing tokens (no secrets)")

    revoke = auth_subparsers.add_parser("revoke", help="Revoke a token by id or name")
    revoke.add_argument("identifier", help="Token id or name")

    cert = subparsers.add_parser("cert", help="Manage HTTPS certificates")
    cert_subparsers = cert.add_subparsers(dest="cert_command", required=True)

    cert_init = cert_subparsers.add_parser("init", help="Generate a self-signed HTTPS certificate")
    cert_init.add_argument("--host", default="llm-proxy", help="Certificate host/CN")
    cert_init.add_argument("--ip", default=None, help="Optional IP SAN")

    args = parser.parse_args()

    if args.command in (None, "serve"):
        config = load_config(args.config)
        asyncio.run(run(config))
        return

    config = load_config(args.config)
    auth_manager = AuthManager(config)
    os.makedirs(auth_manager.state_dir, exist_ok=True)

    if args.command == "cert":
        tls_cfg = config.get("tls") or {}
        generated = generate_server_cert(config["state_dir"], host=args.host, ip=args.ip)
        print("TLS certificate created:")
        print(f"  Cert:        {generated['cert']}")
        print(f"  Key:         {generated['key']}")
        print(f"  Fingerprint: {generated['fingerprint']}")
        if tls_cfg.get("cert_file") != generated["cert"] or tls_cfg.get("key_file") != generated["key"]:
            print("  Note: update config tls.cert_file/tls.key_file if you want to use these paths.")
        return

    if args.auth_command == "create":
        record = auth_manager.create_token(args.name)
        print("Token created. Copy it now — it will not be shown again.")
        print(f"  id:    {record['id']}")
        print(f"  name:  {record['name']}")
        print(f"  token: {record['token']}")
        print()
        print("Use as: Authorization: Bearer " + record["token"])
        return

    if args.auth_command == "list":
        records = auth_manager.list_tokens()
        if not records:
            print("(no tokens)")
            return
        print(f"{'id':<14} {'name':<20} created                  last used")
        for r in records:
            print(
                f"{r.get('id',''):<14} "
                f"{r.get('name',''):<20} "
                f"{r.get('created_at') or '-':<24} "
                f"{r.get('last_used_at') or '-'}"
            )
        return

    if args.auth_command == "revoke":
        ok = auth_manager.revoke_token(args.identifier)
        if not ok:
            raise SystemExit(f"no such token: {args.identifier}")
        print(f"Revoked: {args.identifier}")
        return


if __name__ == "__main__":
    main()
