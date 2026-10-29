#!/bin/sh
# Omnicompute one-line installer for macOS and Linux. Works from any network:
#   curl -fsSL https://raw.githubusercontent.com/TOS-main/omnicompute/main/omnicompute/installers/install.sh | OMNICOMPUTE_CODE=CODE sh
# (CODE = the contribute code or permanent link the main device shows), or served by the main device:
#   curl -fsSL "@ORIGIN@/install.sh?c=@KEY@" | sh
# Without a code it asks: join my mesh, start my own mesh here (OMNICOMPUTE_HOST=1), or help the
# creator's public mesh (OMNICOMPUTE_CODE=creator).
# OMNICOMPUTE_CORES=N picks how many CPU cores to use without asking; OMNICOMPUTE_YES=1 answers yes.
# Installs Omnicompute into ~/.omnicompute/venv and starts contributing in the background.
# It asks before installing, and starts again at every login. Remove it with
# ~/.omnicompute/venv/bin/omnicompute contribute --uninstall (or Devices tab > Remove on the main computer).
set -e
HUB="@ORIGIN@"
KEY="@KEY@"
DIR="$HOME/.omnicompute"
REPO="${OMNICOMPUTE_REPO:-TOS-main/omnicompute}"
case "$HUB" in
  @*)  # run straight from GitHub: the code finds the main device wherever it is
    HUB=""
    CODE="${OMNICOMPUTE_CODE:-}"
    if [ -z "$CODE" ] && [ "${OMNICOMPUTE_HOST:-}" != "1" ]; then
      echo "What should this computer do?"
      echo "  1) Help my own mesh (I have a code or link from my main device)"
      echo "  2) Start my own mesh here (this computer becomes the main device)"
      echo "  3) Help the Omnicompute creator's public mesh with spare power"
      printf "Choose 1, 2 or 3: "
      { read -r choice < /dev/tty; } 2>/dev/null || choice=""
      case "$choice" in
        1) printf "Paste the code or link: "; { read -r CODE < /dev/tty; } 2>/dev/null || CODE="" ;;
        2) OMNICOMPUTE_HOST=1 ;;
        3) CODE="creator" ;;
      esac
      [ -n "$CODE" ] || [ "${OMNICOMPUTE_HOST:-}" = "1" ] || { echo "Nothing was installed."; exit 1; }
    fi
    FROM="GitHub ($REPO)"
    if [ "${OMNICOMPUTE_HOST:-}" = "1" ]; then JOIN=""; WHERE="a new mesh of your own"
    elif [ "$CODE" = "creator" ]; then JOIN="$CODE"; WHERE="the Omnicompute creator's public mesh"
    else JOIN="$CODE"; WHERE="your mesh"; fi ;;
  *) JOIN="$HUB/contribute?c=$KEY"; FROM="your main device"; WHERE="$HUB" ;;
esac
CORES_ALL=$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1)

if [ -z "$JOIN" ]; then
echo "Omnicompute will make this computer the main device of $WHERE."
echo "  - installs into $DIR (no admin rights needed)"
echo "  - then starts it in this window; close the window to stop, run $DIR/omnicompute-host to start again"
else
echo "Omnicompute will set up this computer to lend its spare power to $WHERE."
echo "  - installs into $DIR (no admin rights needed)"
echo "  - runs in the background with no window (it shows in Activity Monitor / ps as python, omnicompute join)"
echo "  - starts when you log in, uses at most 80% by default, and pauses when the main device says so"
echo "  - stop and remove it any time:  $DIR/venv/bin/omnicompute contribute --uninstall"
fi
echo "Only do this on a computer you own or have permission to use."
if [ "${OMNICOMPUTE_YES:-}" != "1" ]; then
  printf "Continue? [y/N] "
  { read -r answer < /dev/tty; } 2>/dev/null || answer=""
  case "$answer" in y*|Y*) ;; *) echo "Nothing was installed."; exit 0 ;; esac
fi
mkdir -p "$DIR"
echo "Omnicompute installer" > "$DIR/installed-by-edgemesh-installer"  # lets "Remove device" delete it again

PY=""
for c in python3 python; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c "import sys; assert sys.version_info >= (3, 10)" 2>/dev/null; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then
  echo "Omnicompute needs Python 3.10 or newer."
  if [ "$(uname)" = "Darwin" ]; then echo "Install it from https://www.python.org/downloads/ (or: brew install python), then run this again."
  else echo "Install it with your package manager (for example: sudo apt install python3 python3-venv), then run this again."; fi
  exit 1
fi

CORES="${OMNICOMPUTE_CORES:-}"
if [ -z "$CORES" ] && [ "${OMNICOMPUTE_YES:-}" != "1" ]; then
  printf "This computer has %s CPU cores. How many should Omnicompute use? [all] " "$CORES_ALL"
  { read -r CORES < /dev/tty; } 2>/dev/null || CORES=""
fi
AWAKE="${OMNICOMPUTE_AWAKE:-}"
if [ -z "$AWAKE" ] && [ "${OMNICOMPUTE_YES:-}" != "1" ]; then
  printf "Keep this computer from going to sleep while it helps (the screen can still turn off)? [y/N] "
  { read -r AWAKE < /dev/tty; } 2>/dev/null || AWAKE=""
fi
case "$AWAKE" in y*|Y*|1) AWAKE="--awake" ;; *) AWAKE="" ;; esac
case "$CORES" in ''|*[!0-9]*) CORES="" ;; *) [ "$CORES" -ge "$CORES_ALL" ] && CORES="" ;; esac

echo "Downloading Omnicompute from $FROM..."
if [ -n "$HUB" ]; then curl -fsSL "$HUB/download/omnicompute.zip?c=$KEY" -o "$DIR/omnicompute.zip"
else curl -fsSL "https://github.com/$REPO/archive/refs/heads/main.zip" -o "$DIR/omnicompute.zip"; fi
rm -rf "$DIR/src"
"$PY" -m zipfile -e "$DIR/omnicompute.zip" "$DIR/src"
PKG=$(dirname "$(ls "$DIR"/src/*/pyproject.toml | head -n 1)")

if [ ! -x "$DIR/venv/bin/python" ]; then
  "$PY" -m venv "$DIR/venv" || { echo "Could not create a virtual environment (on Debian/Ubuntu: sudo apt install python3-venv)"; exit 1; }
fi
echo "Installing (this takes a minute the first time)..."
"$DIR/venv/bin/python" -m pip install --quiet --disable-pip-version-check --upgrade "$PKG"

# NVIDIA card (Linux): offer CUDA (PyTorch), several times faster than Vulkan for this
if command -v nvidia-smi >/dev/null 2>&1 && ! "$DIR/venv/bin/python" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  cuda=y
  if [ "${OMNICOMPUTE_YES:-}" != "1" ]; then
    printf "Found an NVIDIA card. Install CUDA support for it (about 3 GB download, much faster)? [Y/n] "
    { read -r cuda < /dev/tty; } 2>/dev/null || cuda=y
  fi
  case "$cuda" in n*|N*) ;; *)
    echo "Installing CUDA support (PyTorch); this can take a few minutes..."
    "$DIR/venv/bin/python" -m pip install --quiet --disable-pip-version-check torch --index-url https://download.pytorch.org/whl/cu126 \
      || echo "CUDA support could not be installed; Omnicompute will use Vulkan for the GPU instead." ;;
  esac
fi

if [ -z "$JOIN" ]; then
  printf '#!/bin/sh\nexec "%s/venv/bin/python" -m omnicompute host --public %s "$@"\n' "$DIR" "${CORES:+--cores $CORES}${OMNICOMPUTE_KEY:+ --key $OMNICOMPUTE_KEY}" > "$DIR/omnicompute-host"
  chmod +x "$DIR/omnicompute-host"
  echo
  echo "Starting your mesh. Share the permanent contribute link it prints; it keeps working when"
  echo "this computer's address changes. Start it again later with: $DIR/omnicompute-host"
  exec "$DIR/omnicompute-host"
fi
"$DIR/venv/bin/python" -m omnicompute contribute "$JOIN" ${CORES:+--cores "$CORES"} $AWAKE
echo
echo "Done. This computer now shows up in the main device's Devices tab, from any network."
echo "Usage limit is 80% by default; change it and the cores there. On this computer:"
echo "  change:     $DIR/venv/bin/omnicompute limits --use both --cpu 50 --gpu 80 --cores 4 --awake on"
echo "  pause:      $DIR/venv/bin/omnicompute contribute --stop"
echo "  uninstall:  $DIR/venv/bin/omnicompute contribute --uninstall"
