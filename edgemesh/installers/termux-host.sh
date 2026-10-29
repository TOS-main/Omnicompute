#!/data/data/com.termux/files/usr/bin/sh
# EdgeMesh main computer on an Android phone, in Termux (install Termux from F-Droid or GitHub,
# not the old Play Store version). Paste into Termux:
#   curl -fsSL "@ORIGIN@/termux.sh?k=@KEY@" | sh
# or run it from an unzipped EdgeMesh folder:  sh edgemesh/installers/termux-host.sh
# It keeps the same mesh key, so your links only change address. The phone only coordinates
# (light on battery); the computers and other phones do the model's work.
set -e
HUB="@ORIGIN@"
KEY="@KEY@"
case "$HUB" in @*) HUB="" ;; esac
case "$KEY" in @*) KEY="" ;; esac
DIR="$HOME/.edgemesh"

echo "EdgeMesh will make this phone the main computer of your AI mesh."
echo "  - installs Python and EdgeMesh inside Termux (about 300 MB, no root needed)"
echo "  - keeps running with the screen off (Termux shows a notification while it runs)"
echo "  - stop it with: edgemesh-host-stop   (or close Termux from its notification)"
if [ "${EDGEMESH_YES:-}" != "1" ]; then
  printf "Continue? [y/N] "
  { read -r answer < /dev/tty; } 2>/dev/null || answer=""
  case "$answer" in y*|Y*) ;; *) echo "Nothing was installed."; exit 0 ;; esac
fi

# Termux has no prebuilt wheels: use the pure-Python versions of aiohttp's helpers (no compiler)
export PIP_BREAK_SYSTEM_PACKAGES=1 AIOHTTP_NO_EXTENSIONS=1 MULTIDICT_NO_EXTENSIONS=1 YARL_NO_EXTENSIONS=1 \
  FROZENLIST_NO_EXTENSIONS=1 PROPCACHE_NO_EXTENSIONS=1
echo "Installing Python and libraries (a few minutes)..."
pkg update -y >/dev/null 2>&1 || true
pkg install -y python python-numpy python-cryptography cloudflared unzip >/dev/null
# tokenizers is compiled code: use Termux's package when there is one, otherwise build it once
pkg install -y python-tokenizers >/dev/null 2>&1 || {
  echo "Building the tokenizer library (one time, can take 10-20 minutes)..."
  pkg install -y rust binutils >/dev/null
  pip install --quiet tokenizers
}
pip install --quiet --disable-pip-version-check aiohttp jinja2 segno threadpoolctl

mkdir -p "$DIR"
if [ -n "$HUB" ]; then
  echo "Downloading EdgeMesh from $HUB..."
  curl -fsSL "$HUB/download/edgemesh.zip?k=$KEY" -o "$DIR/edgemesh.zip"
  rm -rf "$DIR/src" && unzip -q "$DIR/edgemesh.zip" -d "$DIR/src"
  SRC="$DIR/src/edgemesh-src"
else
  SRC="$(cd "$(dirname "$0")/../.." && pwd)"
fi
pip install --quiet --disable-pip-version-check --no-deps --upgrade "$SRC"
[ -n "$KEY" ] && printf "%s" "$KEY" > "$DIR/mesh.key"  # same key as before: links keep working

BIN="$PREFIX/bin"
cat > "$BIN/edgemesh-host" <<'RUN'
#!/data/data/com.termux/files/usr/bin/sh
# Start the EdgeMesh main computer in the background and keep the phone awake for it.
termux-wake-lock 2>/dev/null || true
pkill -f "edgemesh host" 2>/dev/null || true
nohup python -m edgemesh host --public --no-worker > "$HOME/.edgemesh/host.log" 2>&1 &
sleep 12
grep -E "Open|Contribute|Windows|Mac" "$HOME/.edgemesh/host.log" || echo "Starting... see ~/.edgemesh/host.log"
RUN
cat > "$BIN/edgemesh-host-stop" <<'RUN'
#!/data/data/com.termux/files/usr/bin/sh
pkill -f "edgemesh host" 2>/dev/null && echo "EdgeMesh stopped." || echo "EdgeMesh wasn't running."
termux-wake-unlock 2>/dev/null || true
RUN
chmod +x "$BIN/edgemesh-host" "$BIN/edgemesh-host-stop"
# start again after the phone restarts, if the Termux:Boot app is installed
mkdir -p "$HOME/.termux/boot" && cp "$BIN/edgemesh-host" "$HOME/.termux/boot/edgemesh-host"

edgemesh-host
echo
echo "Done. Open the 'ANYWHERE' link above on your devices (it changes when the phone restarts EdgeMesh)."
echo "Later: edgemesh-host to start, edgemesh-host-stop to stop, cat ~/.edgemesh/host.log for the links."
echo "Tip: Android Settings > Apps > Termux > Battery > Unrestricted keeps it from being paused."
