#!/data/data/com.termux/files/usr/bin/sh
# Omnicompute on an Android phone, in Termux (install Termux from F-Droid or GitHub, not the old
# Play Store version). No computer needed; paste into Termux:
#   curl -fsSL https://raw.githubusercontent.com/TOS-main/Omnicompute/main/omnicompute/installers/termux-host.sh | sh
#       host: this phone becomes the main device of a new mesh (OMNICOMPUTE_KEY=KEY keeps an old mesh key)
#   curl -fsSL https://raw.githubusercontent.com/TOS-main/Omnicompute/main/omnicompute/installers/termux-host.sh | OMNICOMPUTE_CODE=CODE sh
#       contribute: this phone computes for the mesh whose contribute code or link this is
# A main device serves the same script with its key filled in: curl -fsSL "@ORIGIN@/termux.sh?k=@KEY@" | sh
# Hosting keeps the same mesh key, so your links only change address. A host can also compute
# (asked below); computing natively uses all CPU cores with NumPy and is usually faster than
# the browser page.
set -e
HUB="@ORIGIN@"
KEY="@KEY@"
MODE="@MODE@"
case "$HUB" in @*) HUB="" ;; esac
case "$KEY" in @*) KEY="${OMNICOMPUTE_KEY:-}" ;; esac
case "$MODE" in @*) MODE="host" ;; esac
REPO="${OMNICOMPUTE_REPO:-TOS-main/Omnicompute}"
JOIN=""
if [ -n "${OMNICOMPUTE_CODE:-}" ]; then MODE="contribute"; JOIN="$OMNICOMPUTE_CODE"
elif [ "$MODE" = "contribute" ]; then JOIN="$HUB/contribute?c=$KEY"; fi
DIR="$HOME/.omnicompute"

if [ "$MODE" = "contribute" ]; then
  echo "Omnicompute will let this phone compute for your AI mesh."
else
  echo "Omnicompute will make this phone the main computer of your AI mesh."
fi
echo "  - installs Python and Omnicompute inside Termux (about 300 MB, no root needed)"
echo "  - keeps running with the screen off (Termux shows a notification while it runs)"
echo "  - stop it with: omnicompute-host-stop   (or close Termux from its notification)"
if [ "${OMNICOMPUTE_YES:-}" != "1" ]; then
  printf "Continue? [y/N] "
  { read -r answer < /dev/tty; } 2>/dev/null || answer=""
  case "$answer" in y*|Y*) ;; *) echo "Nothing was installed."; exit 0 ;; esac
fi
COMPUTE="${OMNICOMPUTE_COMPUTE:-}"
if [ "$MODE" = "host" ] && [ -z "$COMPUTE" ]; then
  printf "Also let this phone compute on the side (at most 50%%, off by default; change later with omnicompute-host --compute / --no-compute)? [y/N] "
  { read -r COMPUTE < /dev/tty; } 2>/dev/null || COMPUTE=""
fi
case "$COMPUTE" in y*|Y*|1) COMPUTE=1 ;; *) COMPUTE=0 ;; esac

# Termux has no prebuilt wheels: use the pure-Python versions of aiohttp's helpers (no compiler)
export PIP_BREAK_SYSTEM_PACKAGES=1 AIOHTTP_NO_EXTENSIONS=1 MULTIDICT_NO_EXTENSIONS=1 YARL_NO_EXTENSIONS=1 \
  FROZENLIST_NO_EXTENSIONS=1 PROPCACHE_NO_EXTENSIONS=1
echo "Installing Python and libraries (a few minutes)..."
pkg update -y >/dev/null 2>&1 || true
pkg install -y python python-numpy python-cryptography cloudflared unzip >/dev/null
# the fast tokenizer library when Termux has it; otherwise Omnicompute's own (no compiling needed)
python -c "import tokenizers" 2>/dev/null || pkg install -y python-tokenizers >/dev/null 2>&1 || true
pip install --quiet --disable-pip-version-check aiohttp jinja2 segno threadpoolctl gguf

mkdir -p "$DIR"
if [ -n "$HUB" ]; then
  echo "Downloading Omnicompute from $HUB..."
  curl -fsSL "$HUB/download/omnicompute.zip?k=$KEY" -o "$DIR/omnicompute.zip"
  rm -rf "$DIR/src" && unzip -q "$DIR/omnicompute.zip" -d "$DIR/src"
  SRC="$DIR/src/omnicompute-src"
elif [ -f "$(dirname "$0")/../../pyproject.toml" ]; then
  SRC="$(cd "$(dirname "$0")/../.." && pwd)"  # run from an unzipped Omnicompute folder
else
  echo "Downloading Omnicompute from GitHub ($REPO)..."
  curl -fsSL "https://github.com/$REPO/archive/refs/heads/main.zip" -o "$DIR/omnicompute.zip"
  rm -rf "$DIR/src" && unzip -q "$DIR/omnicompute.zip" -d "$DIR/src"
  SRC="$(dirname "$(ls "$DIR"/src/*/pyproject.toml | head -n 1)")"
fi
pip install --quiet --disable-pip-version-check --no-deps --upgrade "$SRC"
[ "$MODE" = "host" ] && [ -n "$KEY" ] && printf "%s" "$KEY" > "$DIR/mesh.key"  # same key: links keep working
echo "$COMPUTE" > "$DIR/termux-compute"
if [ "$MODE" = "contribute" ]; then printf "%s" "$JOIN" > "$DIR/termux-join"
else rm -f "$DIR/termux-join"; fi  # hosting replaces an earlier contribute setup

BIN="$PREFIX/bin"
cat > "$BIN/omnicompute-host" <<'RUN'
#!/data/data/com.termux/files/usr/bin/sh
# Start Omnicompute in the background (host, or contribute) and keep the phone awake for it.
D="$HOME/.omnicompute"
case "$1" in --compute) echo 1 > "$D/termux-compute" ;; --no-compute) echo 0 > "$D/termux-compute" ;; esac
termux-wake-lock 2>/dev/null || true
pkill -f "python -m omnicompute (host|join)" 2>/dev/null || true
if [ -f "$D/termux-join" ]; then
  nohup python -m omnicompute join "$(cat "$D/termux-join")" --name "phone-$(getprop ro.product.model 2>/dev/null | tr -c 'a-zA-Z0-9\n' - | tr A-Z a-z)" > "$D/host.log" 2>&1 &
  sleep 15
  grep -E "joined|could not" "$D/host.log" || echo "Starting... see ~/.omnicompute/host.log"
  exit 0
fi
# hosting comes first: computing is a side job, capped at 50% so coordinating stays quick
if [ "$(cat "$D/termux-compute" 2>/dev/null)" = "1" ]; then EXTRA="--cpu 50"; else EXTRA="--no-worker"; fi
nohup python -m omnicompute host --public $EXTRA > "$D/host.log" 2>&1 &
sleep 12
grep -E -A1 "Open|Permanent|full:|contribute:|Windows|Mac|joined" "$D/host.log" || echo "Starting... see ~/.omnicompute/host.log"
RUN
cat > "$BIN/omnicompute-host-stop" <<'RUN'
#!/data/data/com.termux/files/usr/bin/sh
pkill -f "python -m omnicompute (host|join)" 2>/dev/null && echo "Omnicompute stopped." || echo "Omnicompute wasn't running."
termux-wake-unlock 2>/dev/null || true
RUN
chmod +x "$BIN/omnicompute-host" "$BIN/omnicompute-host-stop"
# start again after the phone restarts, if the Termux:Boot app is installed
mkdir -p "$HOME/.termux/boot" && cp "$BIN/omnicompute-host" "$HOME/.termux/boot/omnicompute-host"

omnicompute-host
echo
if [ "$MODE" = "contribute" ]; then echo "Done. This phone now shows up in the main computer's Devices tab."
else echo "Done. Save the permanent links above: they keep working when the phone's address changes."; fi
echo "Later: omnicompute-host to start, omnicompute-host-stop to stop, cat ~/.omnicompute/host.log for the links."
echo "Tip: Android Settings > Apps > Termux > Battery > Unrestricted keeps it from being paused."
