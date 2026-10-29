"""Runs the main device inside the Android app (android/), through Chaquopy.

The app's background service calls start() once; the hub then runs on its own thread, exactly
like `omnicompute host --public`, with the tunnel tool the app ships. The app's screen is a web
view of the hub's own page, so everything else (models, devices, links) works as on a computer.
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
from pathlib import Path

PORT = 8000
_state: dict = {"thread": None, "loop": None, "task": None, "error": None}


def _env(home: str, cloudflared: str = "") -> None:
    os.environ["OMNICOMPUTE_HOME"] = str(Path(home) / ".omnicompute")
    os.environ.setdefault("HOME", home)
    if cloudflared:
        os.environ["OMNICOMPUTE_CLOUDFLARED"] = cloudflared
    os.environ.setdefault("OMNICOMPUTE_GPU", "0")  # no Vulkan/WebGPU inside the app
    os.environ.setdefault("GODEBUG", "netdns=cgo")  # cloudflared: Android has no /etc/resolv.conf
    Path(os.environ["OMNICOMPUTE_HOME"]).mkdir(parents=True, exist_ok=True)
    log = open(Path(os.environ["OMNICOMPUTE_HOME"]) / "host.log", "a", buffering=1, encoding="utf-8")
    sys.stdout = sys.stderr = log  # the host prints its links; the app shows them from this file


def _run(make) -> None:
    """Run the coroutine make() returns on its own thread, like the command line would."""
    def run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        _state["loop"], _state["error"] = loop, None
        _state["task"] = loop.create_task(make())
        try:
            loop.run_until_complete(_state["task"])
        except asyncio.CancelledError:
            pass
        except BaseException as e:  # noqa: BLE001 - shown in the app instead of crashing it
            _state["error"] = str(e)
            print(f"Omnicompute stopped: {e}", flush=True)
        finally:
            loop.close()

    _state["thread"] = threading.Thread(target=run, name="omnicompute", daemon=True)
    _state["thread"].start()


def start(home: str, cloudflared: str = "", compute: bool = False, cores: int = 0, port: int = PORT,
          public: bool = True, public_url: str = "") -> None:
    """Start hosting (no-op while it already runs). compute=True also runs layers on the phone,
    capped at 50% so coordinating stays quick; cores=0 means all of them. public=False hosts on
    the phone's own address instead of a Cloudflare tunnel: same Wi-Fi only, or from anywhere
    through public_url (e.g. http://your-ip:8000 with the port forwarded on the router)."""
    if running():
        return
    _env(home, cloudflared)
    from . import cli

    argv = ["host", "--port", str(port)] + (["--public-url", public_url] if public_url.strip() and not public
                                             else ["--public"] if public else [])
    if compute:
        argv += ["--cpu", "50"] + (["--cores", str(cores)] if cores else [])
    else:
        argv += ["--no-worker"]
    args = cli.parser().parse_args(argv)
    _run(lambda: cli.cmd_host(args))


def join(home: str, link: str, name: str = "", cores: int = 0) -> None:
    """Help a mesh another device hosts (no-op while running): this device runs layers for the
    computer the link points to, exactly like `omnicompute join LINK` there. A permanent link or
    a bare code keeps finding the computer when its address changes."""
    if running():
        return
    _env(home)
    from . import cli

    argv = ["join", link.strip()] + (["--name", name] if name else []) + (["--cores", str(cores)] if cores else [])
    args = cli.parser().parse_args(argv)
    _run(lambda: cli.cmd_join(args))


def check_link(link: str) -> str:
    """'' when the link (or code) can be used to help a mesh, else why not."""
    from .node import parse_join_url

    try:
        url, key = parse_join_url(link)
    except Exception:  # noqa: BLE001
        return "That doesn't look like an Omnicompute link."
    if not key:
        return "That link has no key. Copy the contribute link from your main computer's Connect tab."
    return ""


def status(home: str) -> str:
    """The last lines the worker printed, for the app's screen while it helps a mesh."""
    p = Path(home) / ".omnicompute" / "host.log"
    try:
        lines = [x for x in p.read_text(encoding="utf-8", errors="replace").splitlines() if x.strip()]
    except OSError:
        return ""
    return "\n".join(lines[-8:])


def stop() -> None:
    loop, task = _state["loop"], _state["task"]
    if loop and task and not loop.is_closed():
        loop.call_soon_threadsafe(task.cancel)
    if _state["thread"]:
        _state["thread"].join(timeout=10)


def running() -> bool:
    return bool(_state["thread"] and _state["thread"].is_alive())


def error() -> str:
    return _state["error"] or ""


def mesh_key(home: str) -> str:
    p = Path(home) / ".omnicompute" / "mesh.key"
    return p.read_text(encoding="utf-8").strip() if p.exists() else ""


def permanent_link(home: str) -> str:
    """The link to share: it finds this phone wherever it is (see beacon.py)."""
    from .crypto import contribute_key
    from .project import permanent_link as link

    key = mesh_key(home)
    return link(contribute_key(key)) if key else ""


def pages(home: str = "") -> str:
    from .project import PAGES

    return PAGES


def set_key(home: str, key: str) -> str:
    """Keep an existing mesh (same links, same devices) by using its key here. '' when saved,
    else why not. The app restarts hosting afterwards."""
    from .crypto import is_contribute_key
    from .node import parse_join_url

    key = key.strip()
    if "/" in key or "=" in key:  # a pasted link
        key = parse_join_url(key)[1]
    if is_contribute_key(key):
        return "That is a contribute-only code; the main device needs the full mesh key."
    if len(key) < 8 or not all(ch.isalnum() or ch in "-_" for ch in key):
        return "A mesh key has at least 8 letters, digits, - or _."
    p = Path(home) / ".omnicompute" / "mesh.key"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(key)
    return ""
