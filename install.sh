#!/bin/bash
set -e
cd "$(dirname "$0")"

pip3 install --user --break-system-packages -e . -q
mkdir -p ~/.config/systemd/user
cp llm-proxy.service ~/.config/systemd/user/llm-proxy.service
systemctl --user daemon-reload
systemctl --user enable llm-proxy

echo "Installed. Run: systemctl --user start llm-proxy"
