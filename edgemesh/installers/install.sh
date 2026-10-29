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

# NVIDIA card (Linux): offer CUDA (PyTorch), several times faster than Vulkan for this
if command -v nvidia-smi >/dev/null 2>&1 && ! "$DIR/venv/bin/python" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  cuda=y
  if [ "${EDGEMESH_YES:-}" != "1" ]; then
    printf "Found an NVIDIA card. Install CUDA support for it (about 3 GB download, much faster)? [Y/n] "
    { read -r cuda < /dev/tty; } 2>/dev/null || cuda=y
  fi
  case "$cuda" in n*|N*) ;; *)
    echo "Installing CUDA support (PyTorch); this can take a few minutes..."
    "$DIR/venv/bin/python" -m pip install --quiet --disable-pip-version-check torch --index-url https://download.pytorch.org/whl/cu126 \
      || echo "CUDA support could not be installed; EdgeMesh will use Vulkan for the GPU instead." ;;
  esac
fi

"$DIR/venv/bin/python" -m edgemesh contribute "$HUB/contribute?c=$KEY"
echo
echo "Done. This computer now shows up in the main computer's Devices tab."
echo "Usage limit is 80% by default; change it there. On this computer:"
echo "  pause:      $DIR/venv/bin/edgemesh contribute --stop"
echo "  uninstall:  $DIR/venv/bin/edgemesh contribute --uninstall"
