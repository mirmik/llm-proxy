"""Entry point for llm-proxy."""

import argparse
import asyncio
import os
import signal

from aiohttp import web

from llm_proxy.auth import AuthManager
from llm_proxy.certs import certificate_fingerprint, generate_server_cert, make_ssl_context
from llm_proxy.config import load_config
from llm_proxy.https_client import post_json
from llm_proxy.resource_coordinator import ResourceCoordinator
from llm_proxy.runtime_manager import RuntimeManager
from llm_proxy.slot_manager import SlotManager
from llm_proxy.router import create_app


async def run(config):
    coordinator = ResourceCoordinator()
    rm = RuntimeManager(config, coordinator=coordinator)
    sm = SlotManager()
    auth = AuthManager(config)
    coordinator.register("runtime", rm.list_running_on_gpu, rm.stop)
    app = create_app(rm, sm, auth)

    listen = config["listen"]
    host = listen["host"] if isinstance(listen, dict) else "0.0.0.0"
    port = listen["port"] if isinstance(listen, dict) else listen
    tls_cfg = config.get("tls") or {}
    ssl_context = None
    scheme = "http"
    if tls_cfg.get("enabled", False):
        ssl_context = make_ssl_context(tls_cfg["cert_file"], tls_cfg["key_file"])
        scheme = "https"

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port, ssl_context=ssl_context)
    await site.start()
    print(f"  LLM Proxy listening on {scheme}://{host}:{port}")
    print(f"  Runtimes: {rm.available_runtimes()}")
    print(f"  Loaded models: {rm.loaded_models()}")
    if auth.enabled:
        print(f"  Control auth: enabled ({auth.state_dir})")
    if tls_cfg.get("enabled", False):
        print(f"  TLS fingerprint: {certificate_fingerprint(tls_cfg['cert_file'])}")
    asyncio.create_task(rm.autostart())

    # Wait for shutdown signal
    stop = asyncio.Event()
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)

    await stop.wait()
    print("\n  Shutting down...")

    # Save all slot caches before stopping
    for name in rm.loaded_models():
        port = rm.get_port(name)
        if port:
            await sm.save_all(name, port)

    await rm.stop_all()
    await runner.cleanup()


def main():
    parser = argparse.ArgumentParser(description="LLM Proxy")
    parser.add_argument("-c", "--config", help="Path to config.yaml")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("serve", help="Run the proxy server")

    auth = subparsers.add_parser("auth", help="Manage control-plane authentication")
    auth_subparsers = auth.add_subparsers(dest="auth_command", required=True)

    invite = auth_subparsers.add_parser("invite", help="Create a one-time bootstrap invite")
    invite.add_argument("host_id", help="Stable host identifier")
    invite.add_argument("--ttl-hours", type=float, default=24.0, help="Invite lifetime in hours")

    login = auth_subparsers.add_parser("login", help="Bootstrap and save local control credentials")
    login.add_argument("--server", required=True, help="Control API URL, preferably https://...")
    login.add_argument("--token", default=None, help="One-time invite token")
    login.add_argument("--password", default=None, help="Bootstrap password")
    login.add_argument("--host-id", default=None, help="Stable host identifier, required for password login")

    set_password = auth_subparsers.add_parser("set-password", help="Store a bootstrap password hash")
    set_password.add_argument("password", help="Bootstrap password")

    clear_password = auth_subparsers.add_parser("clear-password", help="Remove stored bootstrap password")

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

    if args.auth_command == "invite":
        token, invite_meta = auth_manager.create_invite(args.host_id, ttl_hours=args.ttl_hours)
        print("Invite created:")
        print(f"  Host ID:    {invite_meta['host_id']}")
        print(f"  Expires at: {invite_meta['expires_at']}")
        print(f"  Token:      {token}")
        return

    if args.auth_command == "login":
        if bool(args.token) == bool(args.password):
            raise SystemExit("pass exactly one of --token or --password")
        payload = {}
        if args.token:
            payload["token"] = args.token
        if args.password:
            if not args.host_id:
                raise SystemExit("--host-id is required with --password")
            payload["password"] = args.password
            payload["host_id"] = args.host_id
        body = post_json(args.server.rstrip("/") + "/v1/auth/bootstrap", payload, timeout=30)
        host_id = body["host_id"]
        token = body["token"]
        server_fingerprint = body.get("_server_fingerprint")
        if args.server.startswith("https://") and not server_fingerprint:
            raise SystemExit("login over HTTPS did not return a server fingerprint")
        auth_manager.save_local_auth(
            server=args.server,
            host_id=host_id,
            token=token,
            server_fingerprint=server_fingerprint,
        )
        print("Login completed:")
        print(f"  Server:      {args.server}")
        print(f"  Host ID:     {host_id}")
        if server_fingerprint:
            print(f"  Fingerprint: {server_fingerprint}")
        print(f"  Auth:        {auth_manager.local_auth_path()}")
        return

    if args.auth_command == "set-password":
        auth_manager.set_bootstrap_password(args.password)
        print("Bootstrap password stored.")
        print(f"  File: {os.path.join(auth_manager.state_dir, 'auth_bootstrap.json')}")
        return

    if args.auth_command == "clear-password":
        auth_manager.clear_bootstrap_password()
        print("Bootstrap password cleared.")
        return


if __name__ == "__main__":
    main()
