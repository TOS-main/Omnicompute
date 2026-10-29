#!/bin/sh
# Install (first time) and start Omnicompute on macOS or Linux: sh start-mac-linux.sh
cd "$(dirname "$0")" || exit 1
python3 -m pip install --quiet --upgrade . && exec python3 -m omnicompute "$@"
