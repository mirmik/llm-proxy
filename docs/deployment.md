# Deployment

## Linux user service

From a persistent checkout, run:

```bash
./install.sh
```

The installer creates `~/.local/share/llm-proxy/venv`, installs this checkout
in editable mode, copies the example config only when no user config exists,
and installs and enables the user systemd unit. It does not start or restart
the service. Python's `venv` module, pip, and a user systemd session must be
available. No root installation or `--break-system-packages` is used.

Edit `~/.config/llm-proxy/config.yaml` with your binary and model paths, then:

```bash
systemctl --user start llm-proxy
systemctl --user status llm-proxy
journalctl --user -u llm-proxy -f
```

The service uses the user config explicitly. Its executable lives in the
dedicated environment, so it does not depend on an interactive shell's PATH.
Keep the checkout at its installed path while using the editable installation.
Model backends need their own installations; the script only installs the proxy.

`KillMode=mixed` lets the proxy save clean llama.cpp slot state before stopping
its child processes. Remaining processes are killed after the 15-second
shutdown timeout. Large checkpoints may require a longer `TimeoutStopSec` in
a systemd override.

## Updates

After updating the checkout, rerun `./install.sh` to refresh package metadata,
dependencies, and the service definition. Existing configuration is preserved.
Restart when ready:

```bash
systemctl --user restart llm-proxy
```

To stop and disable the service:

```bash
systemctl --user disable --now llm-proxy
```

This retains models, configuration, tokens, statistics, and cached sessions.

## Network access

The example binds inference and control to loopback. Change each listener
explicitly if remote access is required. Configure [authentication and
TLS](authentication.md) before exposing either service. The control panel can
launch processes and edit runtime configuration, so give access only to trusted
operators. Built-in TLS protects inference only; use a reverse proxy or tunnel
for remote control access.
