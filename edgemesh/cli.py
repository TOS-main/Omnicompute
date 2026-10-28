"""Command line entry point: `edgemesh host`, `edgemesh join`, `edgemesh run`, ...

Running `edgemesh` with no arguments (e.g. double-clicking the Windows .exe)
asks whether to host or join, so nobody has to learn the flags.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import secrets
import socket
import sys
from pathlib import Path

log = logging.getLogger("edgemesh")

HOME = Path(os.environ.get("EDGEMESH_HOME", Path.home() / ".edgemesh"))


def _mesh_key(given: str | None) -> str:
    if given:
        return given
    HOME.mkdir(parents=True, exist_ok=True)
    f = HOME / "mesh.key"
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


async def _run_worker(url: str, key: str, name: str, relay_only: bool, mem_fraction: float, forever: bool = True):
    from .node import MeshNode
    from .worker import Worker, worker_info

    info = worker_info(mem_fraction)
    while True:
        node = MeshNode(name, key, url, info, allow_direct=not relay_only)
        Worker().attach(node)
        try:
            await node.start()
            print(f"[{name}] joined {url}  (speed {info['score']} GFLOP/s, offering {info['mem_mb']} MB, "
                  f"public UDP {node.public_addr or 'unknown'})", flush=True)
            await node.closed.wait()
        except (OSError, PermissionError) as e:
            print(f"[{name}] could not join {url}: {e}", flush=True)
            if isinstance(e, PermissionError):
                return
        finally:
            if hasattr(node, "ws"):
                await node.close()
        if not forever:
            return
        await asyncio.sleep(3)
        print(f"[{name}] reconnecting...", flush=True)


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


async def cmd_host(args) -> None:
    from .hub import Hub

    key = _mesh_key(args.key)
    token = args.hf_token or os.environ.get("HF_TOKEN")
    try:
        source = await _load_source(args.model, token)
    except (ValueError, PermissionError, FileNotFoundError) as e:
        sys.exit(f"could not load model: {e}")
    hub = Hub(key, args.bind, args.port, source, str(HOME / "models"), token)
    await hub.start()
    hub.lan_origin = f"http://{_lan_ip()}:{hub.port}"
    lan = f"{hub.lan_origin}/?k={key}"
    print("\nEdgeMesh hub is running.\n")
    print(f"  Open on any phone/PC on this network:  {lan}")
    print(f"  From other networks (Wi-Fi elsewhere, cellular): forward TCP+UDP port {hub.port} on this")
    print(f"  machine's router, or run the hub on a cloud VM, and use http://<public-ip>:{hub.port}/?k={key}")
    print(f"  Add a PC/Mac as a worker:              edgemesh join \"{lan}\"")
    print(f"  Model: {hub.orchestrator.cfg.get('name')} ({hub.orchestrator.cfg['n_layer']} layers). "
          "Load another from Hugging Face on the web page.\n", flush=True)
    tasks = []
    if not args.no_worker:
        tasks.append(asyncio.ensure_future(_run_worker(f"http://127.0.0.1:{hub.port}", key,
                                                       f"{socket.gethostname()[:20]}-host", False, args.mem)))
    try:
        await asyncio.Event().wait()
    finally:
        for t in tasks:
            t.cancel()
        await hub.close()


async def cmd_join(args) -> None:
    from .node import parse_join_url

    url, key = parse_join_url(args.url)
    key = args.key or key
    if not key:
        sys.exit("The join link needs the mesh key (…/?k=KEY), or pass --key.")
    name = args.name or f"{socket.gethostname()[:20]}-{secrets.token_hex(2)}"
    await _run_worker(url, key, name, args.relay_only, args.mem)


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
    print("EdgeMesh: run AI models across your devices.\n")
    print("  1) Host a mesh on this computer (others join it)")
    print("  2) Join an existing mesh as a worker")
    choice = input("\nChoose 1 or 2: ").strip()
    if choice == "2":
        return ["join", input("Paste the join link: ").strip()]
    return ["host"]


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
    h.add_argument("--key", help="mesh key (default: generated once, stored in ~/.edgemesh)")
    h.add_argument("--no-worker", action="store_true", help="don't run layers on the hub machine itself")
    h.add_argument("--mem", type=float, default=0.5, help="fraction of RAM to offer (default 0.5)")

    j = sub.add_parser("join", help="contribute this device's compute to a mesh")
    j.add_argument("url", help="join link printed by `edgemesh host`")
    j.add_argument("--key")
    j.add_argument("--name")
    j.add_argument("--relay-only", action="store_true", help="skip hole punching, always use the relay")
    j.add_argument("--mem", type=float, default=0.5)

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
    fn = {"host": cmd_host, "join": cmd_join, "run": cmd_run, "status": cmd_status,
          "make-model": cmd_make_model, "pull": cmd_pull}[args.cmd]
    try:
        if asyncio.iscoroutinefunction(fn):
            asyncio.run(fn(args))
        else:
            fn(args)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
