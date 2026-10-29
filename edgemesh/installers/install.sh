#!/bin/sh
# EdgeMesh one-line installer for macOS and Linux, served by your main computer.
#   curl -fsSL "@ORIGIN@/install.sh?c=@KEY@" | sh
# Installs EdgeMesh into ~/.edgemesh/venv and starts contributing in the background.
# It asks before installing, and starts again at every login. Remove it with
# ~/.edgemesh/venv/bin/edgemesh contribute --uninstall (or Devices tab > Remove on the main computer).
set -e
HUB="@ORIGIN@"
KEY="@KEY@"
DIR="$HOME/.edgemesh"

echo "EdgeMesh will set up this computer to lend its spare power to the AI mesh at $HUB."
echo "  - installs into $DIR (no admin rights needed)"
echo "  - runs in the background with no window (it shows in Activity Monitor / ps as python, edgemesh join)"
echo "  - starts when you log in, uses at most 80% by default, and pauses when the main computer says so"
echo "  - stop and remove it any time:  $DIR/venv/bin/edgemesh contribute --uninstall"
echo "Only do this on a computer you own or have permission to use."
if [ "${EDGEMESH_YES:-}" != "1" ]; then
  printf "Continue? [y/N] "
  { read -r answer < /dev/tty; } 2>/dev/null || answer=""
  case "$answer" in y*|Y*) ;; *) echo "Nothing was installed."; exit 0 ;; esac
fi
mkdir -p "$DIR"
echo "EdgeMesh installer" > "$DIR/installed-by-edgemesh-installer"  # lets "Remove device" delete it again

PY=""
for c in python3 python; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c "import sys; assert sys.version_info >= (3, 10)" 2>/dev/null; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then
  echo "EdgeMesh needs Python 3.10 or newer."
  if [ "$(uname)" = "Darwin" ]; then echo "Install it from https://www.python.org/downloads/ (or: brew install python), then run this again."
  else echo "Install it with your package manager (for example: sudo apt install python3 python3-venv), then run this again."; fi
  exit 1
fi

echo "EdgeMesh: setting up this computer to contribute to $HUB"
echo "Downloading EdgeMesh from your main computer..."
curl -fsSL "$HUB/download/edgemesh.zip?c=$KEY" -o "$DIR/edgemesh.zip"
rm -rf "$DIR/src"
"$PY" -m zipfile -e "$DIR/edgemesh.zip" "$DIR/src"

if [ ! -x "$DIR/venv/bin/python" ]; then
  "$PY" -m venv "$DIR/venv" || { echo "Could not create a virtual environment (on Debian/Ubuntu: sudo apt install python3-venv)"; exit 1; }
fi
echo "Installing (this takes a minute the first time)..."
"$DIR/venv/bin/python" -m pip install --quiet --disable-pip-version-check --upgrade "$DIR/src/edgemesh-src"

"$DIR/venv/bin/python" -m edgemesh contribute "$HUB/contribute?c=$KEY"
echo
echo "Done. This computer now shows up in the main computer's Devices tab."
echo "Usage limit is 80% by default; change it there. On this computer:"
echo "  pause:      $DIR/venv/bin/edgemesh contribute --stop"
echo "  uninstall:  $DIR/venv/bin/edgemesh contribute --uninstall"
