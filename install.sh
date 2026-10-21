#!/usr/bin/env bash
# Install from this checkout; does not start or restart the service.
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
venv_dir="$HOME/.local/share/llm-proxy/venv"
config_dir="$HOME/.config/llm-proxy"
unit_dir="$HOME/.config/systemd/user"

python3 -m venv "$venv_dir"
"$venv_dir/bin/python" -m pip install -e "$repo_dir"
install -d -m 700 "$config_dir"
if [[ ! -e "$config_dir/config.yaml" ]]; then
    install -m 600 "$repo_dir/config.example.yaml" "$config_dir/config.yaml"
fi
install -d "$unit_dir"
install -m 644 "$repo_dir/deploy/systemd/llm-proxy.service" "$unit_dir/llm-proxy.service"
systemctl --user daemon-reload
systemctl --user enable llm-proxy

echo "Installed. Edit $config_dir/config.yaml, then run: systemctl --user start llm-proxy"
