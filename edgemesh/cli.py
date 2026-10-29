"""Command line entry point: `edgemesh host`, `edgemesh join`, `edgemesh run`, ...

Running `edgemesh` with no arguments (e.g. double-clicking the Windows .exe)
asks whether to host or join, so nobody has to learn the flags.
"""

from __future__ import annotations

import argparse
import asyncio
import errno
import json
import logging
import os
import secrets
import socket
import sys
from pathlib import Path

log = logging.getLogger("edgemesh")

HOME = Path(os.environ.get("EDGEMESH_HOME", Path.home() / ".edgemesh"))


def _mesh_key(given: str | None, save: bool = False) -> str:
    """This computer's mesh key: made once and kept in ~/.edgemesh/mesh.key, so links and
    installed devices keep working across restarts. `host --key NEW` replaces it for good."""
    HOME.mkdir(parents=True, exist_ok=True)
    f = HOME / "mesh.key"
    if given:
        if save and (not f.exists() or f.read_text().strip() != given):
            f.write_text(given)
            print(f"  Saved {given!r} as this computer's mesh key; it stays until you change it again.")
        return given
    if not f.exists():
        f.write_text(secrets.token_urlsafe(18))
    return f.read_text().strip()


def _lan_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def _demo_model() -> str:
    from .model import random_model, save_model

    path = HOME / "demo-model.emm"
    if not path.exists():
        HOME.mkdir(parents=True, exist_ok=True)
        save_model(str(path), *random_model())
    return str(path)


async def _run_worker(url: str, key: str, name: str, relay_only: bool, mem_fraction: float, forever: bool = True,
                      cpu: int | None = None, schedule: str | None = None, always: bool | None = None,
                      hub_machine: bool = False):
    from . import limits as L
    from .node import MeshNode
    from .gpu import open_gpu
    from .worker import Worker, worker_info

    gpu = open_gpu()
    info = {**worker_info(mem_fraction, gpu), "hub_machine": hub_machine}
    overrides = {}
    if cpu is not None:
        overrides["cpu"] = cpu
    if schedule:
        overrides["schedule"] = L.parse_schedule(schedule)
    if always is not None:
        overrides["always"] = always
    store = L.LimitStore(HOME / f"limits-{name}.json", overrides)
    while True:
        node = MeshNode(name, key, url, info, allow_direct=not relay_only)
        worker = Worker(store, gpu).attach(node)
        worker.on_uninstall = lambda: _uninstall(name)
        watcher = None
        try:
            await node.start()
            cpu_now, why = L.effective(store.limits)
            gpu_note = f", GPU {gpu.name} {info['gpu']['score']} GFLOP/s" if gpu else ", no usable GPU (using the CPU)"
            print(f"[{name}] joined {url}  (CPU {info['cpu_score']} GFLOP/s{gpu_note}, offering {info['mem_mb']} MB, "
                  f"usage limit {cpu_now}% ({why}), public UDP {node.public_addr or 'unknown'})", flush=True)
            watcher = asyncio.ensure_future(worker.watch_schedule())
            await node.closed.wait()
        except (OSError, PermissionError) as e:
            print(f"[{name}] could not join {url}: {e}", flush=True)
            if isinstance(e, PermissionError):
                if "key" in str(e):
                    print("  The key in this link doesn't match the main computer's. Copy the link again from the\n"
                          "  main computer's window (or run  edgemesh links  there), and check the port number too:\n"
                          "  a different EdgeMesh may be running on another port.", flush=True)
                return
        finally:
            if watcher:
                watcher.cancel()
            if hasattr(node, "ws"):
                await node.close()
        if not forever:
            return
        await asyncio.sleep(3)
        print(f"[{name}] reconnecting...", flush=True)


def _uninstall(name: str) -> list[str]:
    """The main computer removed this device: stop for good (no autostart) and delete what the
    one-line installer put here. Exits the process shortly after replying."""
    from . import background

    removed = background.uninstall(HOME)
    print(f"[{name}] removed from the mesh by the main computer; "
          + (f"deleting {', '.join(removed)}" if removed else "autostart turned off"), flush=True)
    asyncio.get_running_loop().call_later(0.5, os._exit, 0)
    return removed


def _open_model(spec: str | None, token: str | None):
    """None -> demo model; *.emm -> EdgeMesh file; folder or hf:org/name or org/name -> Hugging Face."""
    from .model import DictSource, load_model

    if not spec:
        return DictSource(*load_model(_demo_model()))
    if spec.endswith(".emm") and os.path.isfile(spec):
        return DictSource(*load_model(spec))
    from .hf import Download, resolve

    if not os.path.isdir(spec) and not spec.startswith("hf:"):
        spec = "hf:" + spec
    progress = Download(spec)
    done = asyncio.Event()

    async def report():
        while not done.is_set():
            if progress.state == "downloading" and progress.total:
                print(f"\r  downloading {spec[3:]}: {progress.done / 2**30:.2f} / {progress.total / 2**30:.2f} GB",
                      end="", flush=True)
            await asyncio.sleep(1)

    return progress, done, report, lambda: resolve(spec, HOME / "models", token, progress)


async def _load_source(spec: str | None, token: str | None):
    opened = _open_model(spec, token)
    if not isinstance(opened, tuple):
        return opened
    progress, done, report, load = opened
    task = asyncio.ensure_future(report())
    try:
        return await asyncio.to_thread(load)
    finally:
        done.set()
        await task
        if progress.total:
            print()


PORT_TAKEN = {errno.EADDRINUSE, errno.EACCES, 10048, 10013}  # 10013: Windows reserves some port ranges


async def _edgemesh_on(port: int) -> dict | None:
    import aiohttp

    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=2)) as s:
            async with s.get(f"http://127.0.0.1:{port}/api/hello") as r:
                return (await r.json()) if r.status == 200 else None
    except Exception:
        return None


async def _start_on_free_port(hub, key: str, port: int) -> bool:
    """Start the hub on `port`, or the next free one. If this mesh's hub is already running
    here, open it in the browser instead of failing. Returns False when nothing was started."""
    import hashlib
    import webbrowser

    mine = hashlib.sha256(key.encode()).hexdigest()[:12]
    for p in range(port, port + 30):
        hub.port = p
        try:
            await hub.start()
            if p != port:
                print(f"  (port {port} is busy, so EdgeMesh is using port {p}. Use the links below: links that"
                      f" say :{port} reach whatever else is running there)")
            return True
        except OSError as e:
            if e.errno not in PORT_TAKEN and getattr(e, "winerror", None) not in PORT_TAKEN:
                raise
        other = await _edgemesh_on(p)
        if other and other.get("mesh") == mine:
            link = f"http://127.0.0.1:{p}/?k={key}"
            print(f"\nEdgeMesh is already running on this computer (port {p}). Opening it: {link}")
            print("To restart it, close the other EdgeMesh window first.")
            try:
                webbrowser.open(link)
            except Exception:
                pass
            return False
    sys.exit(f"Ports {port}-{port + 29} are all in use. Pick another with: edgemesh host --port 9000")


async def cmd_host(args) -> None:
    from .crypto import is_contribute_key
    from .hub import Hub

    if args.key and is_contribute_key(args.key):
        sys.exit("That is a contribute-only key; the main computer needs the full mesh key.")
    if args.key and (len(args.key) < 8 or not all(ch.isalnum() or ch in "-_" for ch in args.key)):
        sys.exit("Pick a key of at least 8 letters, digits, - or _ (anyone with it can use your mesh).")
    key = _mesh_key(args.key, save=True)
    token = args.hf_token or os.environ.get("HF_TOKEN")
    try:
        source = await _load_source(args.model, token)
    except (ValueError, PermissionError, FileNotFoundError) as e:
        sys.exit(f"could not load model: {e}")
    hub = Hub(key, args.bind, args.port, source, str(HOME / "models"), token)
    if not await _start_on_free_port(hub, key, args.port):
        return
    hub.lan_origin = f"http://{_lan_ip()}:{hub.port}"
    lan = f"{hub.lan_origin}/?k={key}"
    tunnel = None
    public = None
    if args.public:
        from . import tunnel as T

        try:
            tunnel, origin = await T.start(hub.port, HOME)
            hub.public_origin = origin
            public = f"{origin}/?k={key}"
        except Exception as e:
            print(f"  (could not open the public tunnel: {e})")
    if args.public_url:  # e.g. a Tailscale, ngrok or VPS address that forwards to this port
        hub.public_origin = args.public_url.rstrip("/")
        public = f"{hub.public_origin}/?k={key}"
    print("\nEdgeMesh is running.\n")
    print(f"  Open on this network (Wi-Fi/LAN):   {lan}")
    if public:
        print(f"  Open from ANYWHERE (school, cellular): {public}")
        print("  (this address changes each time you restart; scan the QR code below or copy it)")
        _print_qr(public)
    else:
        print("  From other networks (school, cellular): restart with  --public  for an https link that")
        print(f"  works anywhere, or forward TCP+UDP port {hub.port} on this machine's router.")
    origin = hub.public_origin or hub.lan_origin
    ck = hub.contribute_key
    print(f"  Contribute-only link (no chat or settings): {origin}/contribute?c={ck}")
    print("  Add a computer in one line (it then contributes in the background):")
    print(f"    Windows PowerShell:  irm \"{origin}/install.ps1?c={ck}\" | iex")
    print(f"    Mac / Linux:         curl -fsSL \"{origin}/install.sh?c={ck}\" | sh")
    print(f"  Model: {hub.orchestrator.cfg.get('name')} ({hub.orchestrator.cfg['n_layer']} layers). "
          "Load another from Hugging Face on the web page.\n", flush=True)
    tasks = []
    if not args.no_worker:
        tasks.append(asyncio.ensure_future(_run_worker(f"http://127.0.0.1:{hub.port}", key,
                                                       f"{socket.gethostname()[:20]}-host", False, args.mem,
                                                       cpu=args.cpu, schedule=args.schedule, hub_machine=True)))
    try:
        await asyncio.Event().wait()
    finally:
        for t in tasks:
            t.cancel()
        if tunnel:
            tunnel.kill()
        await hub.close()


async def cmd_links(args) -> None:
    """Print this computer's join links, for when the host window is closed or scrolled away."""
    import hashlib

    from .crypto import contribute_key

    key = _mesh_key(args.key)
    mine = hashlib.sha256(key.encode()).hexdigest()[:12]
    port = None
    for p in range(8000, 8030):
        hello = await _edgemesh_on(p)
        if hello and hello.get("mesh") == mine:
            port = p
            break
    origin = f"http://{_lan_ip()}:{port or 8000}"
    print(f"Mesh key: {key}   (stored in {HOME / 'mesh.key'})")
    print(f"  Full link (chat, settings, join):  {origin}/?k={key}")
    print(f"  Contribute-only link:              {origin}/contribute?c={contribute_key(key)}")
    if port is None:
        print("  (EdgeMesh isn't hosting on this computer right now; the port is 8000 unless it was busy.)")


def _print_qr(text: str) -> None:
    try:
        import segno

        segno.make(text, error="l").terminal(compact=True)
    except Exception:
        pass


def _log_to(path: str) -> None:
    """Background mode (pythonw has no console): send all output to a log file."""
    f = open(path, "a", buffering=1, encoding="utf-8")
    sys.stdout = sys.stderr = f
    logging.getLogger().handlers.clear()
    logging.basicConfig(stream=f, level=logging.INFO, format="%(asctime)s %(name)s %(message)s", force=True)


async def cmd_join(args) -> None:
    from .node import parse_join_url

    if args.log:
        _log_to(args.log)
    url, key = parse_join_url(args.url)
    key = args.key or key
    if not key:
        sys.exit("The join link needs the mesh key (…/?k=KEY), or pass --key.")
    name = args.name or f"{socket.gethostname()[:20]}-{secrets.token_hex(2)}"
    await _run_worker(url, key, name, args.relay_only, args.mem, cpu=args.cpu, schedule=args.schedule,
                      always=True if args.always else None)


def cmd_contribute(args) -> None:
    from . import background

    if args.uninstall:
        background.stop(HOME, quiet=True)
        removed = background.uninstall(HOME)
        print("EdgeMesh stopped and won't start at login anymore."
              + (f" Deleting {', '.join(removed)}." if removed else ""))
        return
    if args.stop:
        background.stop(HOME)
        return
    if not args.url:
        sys.exit("usage: edgemesh contribute LINK   (or --stop)")
    from .node import parse_join_url

    url, key = parse_join_url(args.url)
    if not key:
        sys.exit("The link needs a key (…/contribute?c=KEY or …/?k=KEY).")
    name = args.name or f"{socket.gethostname()[:20]}-{secrets.token_hex(2)}"
    extra = []
    if args.cpu is not None:
        extra += ["--cpu", str(args.cpu)]
    if args.schedule:
        extra += ["--schedule", args.schedule]
    from .crypto import is_contribute_key

    link = f"{url}/contribute?c={key}" if is_contribute_key(key) else f"{url}/?k={key}"
    pid = background.start(link, name, HOME, extra, autostart=not args.no_autostart)
    print(f"EdgeMesh is contributing in the background as '{name}' (process {pid}).")
    print("It starts again when you log in." if not args.no_autostart else "It stops when you log out.")
    print(f"Log: {HOME / 'contribute.log'}   Stop: edgemesh contribute --stop")


async def cmd_run(args) -> None:
    import aiohttp

    from .node import parse_join_url

    url, key = parse_join_url(args.url)
    key = args.key or key
    body = {"prompt": args.prompt, "max_new": args.max_new, "temperature": args.temperature, "chat": args.chat}
    printed = 0
    async with aiohttp.ClientSession() as s:
        async with s.post(url + "/api/generate", json=body, headers={"X-Mesh-Key": key}) as r:
            if r.status != 200:
                sys.exit(f"hub said {r.status}: {await r.text()}")
            sys.stdout.write(args.prompt)
            async for line in r.content:
                ev = json.loads(line)
                if ev["type"] == "token":
                    sys.stdout.write(ev["text"][printed:])
                    printed = len(ev["text"])
                    sys.stdout.flush()
                elif ev["type"] == "plan" and args.verbose:
                    print(f"[plan] {ev['plan']}", file=sys.stderr)
                elif ev["type"] == "replan":
                    print(f"\n[a device dropped out, re-planned: {ev['plan']}]", file=sys.stderr)
                elif ev["type"] == "done":
                    print(f"\n\n[{ev['stats']['tokens']} tokens, {ev['stats']['tok_per_s']} tok/s]", file=sys.stderr)
                elif ev["type"] == "error":
                    sys.exit(f"\nerror: {ev['error']}")


async def cmd_status(args) -> None:
    import aiohttp

    from .node import parse_join_url

    url, key = parse_join_url(args.url)
    async with aiohttp.ClientSession() as s:
        async with s.get(url + "/api/status", headers={"X-Mesh-Key": args.key or key}) as r:
            print(json.dumps(await r.json(), indent=2))


def cmd_make_model(args) -> None:
    from .model import random_model, save_model

    save_model(args.out, *random_model(args.layers, args.dim, args.heads, args.ctx, seed=args.seed))
    print(f"wrote {args.out}")


async def cmd_pull(args) -> None:
    src = await _load_source(args.repo, args.hf_token or os.environ.get("HF_TOKEN"))
    print(f"{src.cfg['name']}: {src.cfg['n_layer']} layers, saved in {src.folder}")


def _interactive() -> list[str]:
    from . import __version__

    print(f"EdgeMesh {__version__}: run AI models across your devices.\n")
    print("  1) Host a mesh on this computer (others join it)")
    print("  2) Join an existing mesh as a worker")
    print("  3) Show this computer's mesh key and join links")
    choice = input("\nChoose 1, 2 or 3: ").strip()
    if choice == "3":
        return ["links"]
    if choice == "2":
        return ["join", input("Paste the join link: ").strip()]
    public = input("Let devices on other networks join too (school, cellular)? [y/N]: ").strip().lower()
    return ["host", "--public"] if public.startswith("y") else ["host"]


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if not argv and sys.stdin.isatty():
        argv = _interactive()
    p = argparse.ArgumentParser(prog="edgemesh", description=__doc__.split("\n")[0])
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    h = sub.add_parser("host", help="run the hub (+ a local worker) and serve the web page")
    h.add_argument("--model", help="Hugging Face repo (e.g. Qwen/Qwen2.5-0.5B-Instruct), a local HF folder, "
                                   "or a .emm file (default: a small random demo model; you can switch "
                                   "models from the web page)")
    h.add_argument("--hf-token", help="Hugging Face token for gated/private models (or set HF_TOKEN)")
    h.add_argument("--port", type=int, default=8000)
    h.add_argument("--bind", default="0.0.0.0")
    h.add_argument("--key", help="set your own mesh key; it is saved and used from then on "
                   "(default: one generated once and kept in ~/.edgemesh/mesh.key)")
    h.add_argument("--no-worker", action="store_true", help="don't run layers on the hub machine itself")
    h.add_argument("--public", action="store_true",
                   help="also open a free Cloudflare https tunnel so devices on any network can join")
    h.add_argument("--public-url", help="an address you set up yourself that reaches this port (Tailscale, "
                                        "ngrok, a VPS, port forwarding); shown on the Connect page")
    h.add_argument("--mem", type=float, default=0.5, help="fraction of RAM to offer (default 0.5)")

    j = sub.add_parser("join", help="contribute this device's compute to a mesh")
    j.add_argument("url", help="join link printed by `edgemesh host`")
    j.add_argument("--key")
    j.add_argument("--name")
    j.add_argument("--relay-only", action="store_true", help="skip hole punching, always use the relay")
    j.add_argument("--mem", type=float, default=0.5)
    j.add_argument("--always", action="store_true", help="always give this device layers, even if it is slow")
    j.add_argument("--log", help="write output to this file instead of the console (background mode)")

    b = sub.add_parser("contribute", help="contribute quietly in the background, and again after every login")
    b.add_argument("url", nargs="?", help="contribute link (…/contribute?c=KEY) or join link")
    b.add_argument("--name")
    b.add_argument("--stop", action="store_true", help="stop contributing and remove the autostart entry")
    b.add_argument("--uninstall", action="store_true", help="stop, remove autostart, and delete what the installer added")
    b.add_argument("--no-autostart", action="store_true", help="don't start again at login")
    for sp in (h, j, b):
        sp.add_argument("--cpu", type=int, help="usage limit in percent (default 80; 0 = paused)")
        sp.add_argument("--schedule", help='time-of-day limits, e.g. "22:00-07:00=100,09:00-17:00=0"')

    r = sub.add_parser("run", help="run a prompt on the mesh")
    r.add_argument("url")
    r.add_argument("prompt")
    r.add_argument("--key")
    r.add_argument("--max-new", type=int, default=40)
    r.add_argument("--temperature", type=float, default=0.0)
    r.add_argument("--chat", action="store_true", help="wrap the prompt in the model's chat template")

    s = sub.add_parser("status", help="show devices and the current layer split")
    s.add_argument("url")
    s.add_argument("--key")

    lk = sub.add_parser("links", help="show this computer's mesh key and join links")
    lk.add_argument("--key", help=argparse.SUPPRESS)

    m = sub.add_parser("make-model", help="write a random test model")
    m.add_argument("out")
    m.add_argument("--layers", type=int, default=8)
    m.add_argument("--dim", type=int, default=128)
    m.add_argument("--heads", type=int, default=4)
    m.add_argument("--ctx", type=int, default=256)
    m.add_argument("--seed", type=int, default=0)

    c = sub.add_parser("pull", help="download a Hugging Face model ahead of time")
    c.add_argument("repo", help="e.g. Qwen/Qwen2.5-0.5B-Instruct")
    c.add_argument("--hf-token")

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(name)s %(message)s")
    fn = {"host": cmd_host, "join": cmd_join, "contribute": cmd_contribute, "run": cmd_run, "status": cmd_status,
          "make-model": cmd_make_model, "pull": cmd_pull,
          "links": cmd_links}[args.cmd]
    try:
        if asyncio.iscoroutinefunction(fn):
            asyncio.run(fn(args))
        else:
            fn(args)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
