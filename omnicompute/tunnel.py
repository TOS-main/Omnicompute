"""`omnicompute host --public`: reach the hub from anywhere through a Cloudflare quick tunnel.

Cloudflare's free `cloudflared` tool opens an outbound connection from the hub machine
and hands back a public https://<random>.trycloudflare.com address that forwards to the
hub. No router setup or port forwarding, and it looks like ordinary web traffic, so it
usually works from school, office and cellular networks. Devices that join through it
use the relay (UDP hole punching needs a reachable UDP port).
"""

from __future__ import annotations

import asyncio
import platform
import re
import shutil
import stat
import tarfile
import urllib.request
from pathlib import Path

BASE = "https://github.com/cloudflare/cloudflared/releases/latest/download/"
URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")


def _asset() -> str:
    system, machine = platform.system(), platform.machine().lower()
    arch = "arm64" if machine in ("arm64", "aarch64") else "386" if machine in ("i386", "i686", "x86") else "amd64"
    if system == "Windows":
        return f"cloudflared-windows-{arch}.exe"
    if system == "Darwin":
        return f"cloudflared-darwin-{'arm64' if arch == 'arm64' else 'amd64'}.tgz"
    return f"cloudflared-linux-{arch}"


def find_or_download(home: Path) -> str:
    import os

    given = os.environ.get("OMNICOMPUTE_CLOUDFLARED")  # the Android app ships its own build
    if given and Path(given).exists():
        return given
    found = shutil.which("cloudflared")
    if found:
        return found
    bin_dir = home / "bin"
    exe = bin_dir / ("cloudflared.exe" if platform.system() == "Windows" else "cloudflared")
    if exe.exists():
        return str(exe)
    bin_dir.mkdir(parents=True, exist_ok=True)
    asset = _asset()
    print(f"  downloading Cloudflare's tunnel tool ({asset}), one time only…", flush=True)
    tmp = bin_dir / (asset + ".part")
    urllib.request.urlretrieve(BASE + asset, tmp)
    if asset.endswith(".tgz"):
        with tarfile.open(tmp) as t:
            member = next(m for m in t.getmembers() if m.name.endswith("cloudflared"))
            member.name = exe.name
            t.extract(member, bin_dir)
        tmp.unlink()
    else:
        tmp.replace(exe)
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    return str(exe)


async def start(port: int, home: Path, timeout: float = 45) -> tuple[asyncio.subprocess.Process, str]:
    """Start a quick tunnel to the local hub. Returns (process, public https origin)."""
    exe = await asyncio.to_thread(find_or_download, home)
    proc = await asyncio.create_subprocess_exec(
        exe, "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{port}",
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)

    async def read_url():
        while True:
            line = await proc.stderr.readline()
            if not line:
                raise RuntimeError("cloudflared exited before giving a public address")
            m = URL_RE.search(line.decode(errors="replace"))
            if m:
                return m.group(0)

    try:
        url = await asyncio.wait_for(read_url(), timeout)
    except Exception:
        proc.kill()
        raise

    async def drain():  # keep reading so cloudflared never blocks on a full pipe
        while await proc.stderr.readline():
            pass

    asyncio.ensure_future(drain())
    return proc, url
