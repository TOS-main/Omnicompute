#!/data/data/com.termux/files/usr/bin/sh
# EdgeMesh main computer on an Android phone, in Termux (install Termux from F-Droid or GitHub,
# not the old Play Store version). Paste into Termux:
#   curl -fsSL "@ORIGIN@/termux.sh?k=@KEY@" | sh      (host: this phone becomes the main computer)
#   curl -fsSL "@ORIGIN@/termux.sh?c=CONTRIBUTE" | sh  (contribute: this phone computes for the mesh)
# or run it from an unzipped EdgeMesh folder:  sh edgemesh/installers/termux-host.sh
# Hosting keeps the same mesh key, so your links only change address. A host can also compute
# (asked below); computing natively uses all CPU cores with NumPy and is usually faster than
# the browser page.
set -e
HUB="@ORIGIN@"
KEY="@KEY@"
MODE="@MODE@"
case "$HUB" in @*) HUB="" ;; esac
case "$KEY" in @*) KEY="" ;; esac
case "$MODE" in @*) MODE="host" ;; esac
DIR="$HOME/.edgemesh"

if [ "$MODE" = "contribute" ]; then
  echo "EdgeMesh will let this phone compute for the AI mesh at $HUB."
else
  echo "EdgeMesh will make this phone the main computer of your AI mesh."
fi
echo "  - installs Python and EdgeMesh inside Termux (about 300 MB, no root needed)"
echo "  - keeps running with the screen off (Termux shows a notification while it runs)"
echo "  - stop it with: edgemesh-host-stop   (or close Termux from its notification)"
if [ "${EDGEMESH_YES:-}" != "1" ]; then
  printf "Continue? [y/N] "
  { read -r answer < /dev/tty; } 2>/dev/null || answer=""
  case "$answer" in y*|Y*) ;; *) echo "Nothing was installed."; exit 0 ;; esac
fi
COMPUTE="${EDGEMESH_COMPUTE:-}"
if [ "$MODE" = "host" ] && [ -z "$COMPUTE" ]; then
  printf "Also let this phone compute on the side (at most 50%%, off by default; change later with edgemesh-host --compute / --no-compute)? [y/N] "
  { read -r COMPUTE < /dev/tty; } 2>/dev/null || COMPUTE=""
fi
case "$COMPUTE" in y*|Y*|1) COMPUTE=1 ;; *) COMPUTE=0 ;; esac

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
[ "$MODE" = "host" ] && [ -n "$KEY" ] && printf "%s" "$KEY" > "$DIR/mesh.key"  # same key: links keep working
echo "$COMPUTE" > "$DIR/termux-compute"
[ "$MODE" = "contribute" ] && printf "%s/contribute?c=%s" "$HUB" "$KEY" > "$DIR/termux-join"

BIN="$PREFIX/bin"
cat > "$BIN/edgemesh-host" <<'RUN'
#!/data/data/com.termux/files/usr/bin/sh
# Start EdgeMesh in the background (host, or contribute) and keep the phone awake for it.
D="$HOME/.edgemesh"
case "$1" in --compute) echo 1 > "$D/termux-compute" ;; --no-compute) echo 0 > "$D/termux-compute" ;; esac
termux-wake-lock 2>/dev/null || true
pkill -f "python -m edgemesh (host|join)" 2>/dev/null || true
if [ -f "$D/termux-join" ]; then
  nohup python -m edgemesh join "$(cat "$D/termux-join")" --name "phone-$(getprop ro.product.model 2>/dev/null | tr -c 'a-zA-Z0-9\n' - | tr A-Z a-z)" > "$D/host.log" 2>&1 &
  sleep 15
  grep -E "joined|could not" "$D/host.log" || echo "Starting... see ~/.edgemesh/host.log"
  exit 0
fi
# hosting comes first: computing is a side job, capped at 50% so coordinating stays quick
if [ "$(cat "$D/termux-compute" 2>/dev/null)" = "1" ]; then EXTRA="--cpu 50"; else EXTRA="--no-worker"; fi
nohup python -m edgemesh host --public $EXTRA > "$D/host.log" 2>&1 &
sleep 12
grep -E "Open|Contribute|Windows|Mac|joined" "$D/host.log" || echo "Starting... see ~/.edgemesh/host.log"
RUN
cat > "$BIN/edgemesh-host-stop" <<'RUN'
#!/data/data/com.termux/files/usr/bin/sh
pkill -f "python -m edgemesh (host|join)" 2>/dev/null && echo "EdgeMesh stopped." || echo "EdgeMesh wasn't running."
termux-wake-unlock 2>/dev/null || true
RUN
chmod +x "$BIN/edgemesh-host" "$BIN/edgemesh-host-stop"
# start again after the phone restarts, if the Termux:Boot app is installed
mkdir -p "$HOME/.termux/boot" && cp "$BIN/edgemesh-host" "$HOME/.termux/boot/edgemesh-host"

edgemesh-host
echo
if [ "$MODE" = "contribute" ]; then echo "Done. This phone now shows up in the main computer's Devices tab."
else echo "Done. Open the 'ANYWHERE' link above on your devices (it changes when the phone restarts EdgeMesh)."; fi
echo "Later: edgemesh-host to start, edgemesh-host-stop to stop, cat ~/.edgemesh/host.log for the links."
echo "Tip: Android Settings > Apps > Termux > Battery > Unrestricted keeps it from being paused."
