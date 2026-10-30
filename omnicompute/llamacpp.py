"""The fast engine for GGUF models on one computer: llama.cpp's llama-server, the engine LM Studio
runs, with the same speed settings (all layers on the GPU when there is one, flash attention,
memory-mapped weights, one thread per physical core).

When the main computer can hold a GGUF model by itself, this runs it far faster than splitting it
across devices: the weights stay quantized (a 4-bit model takes about a quarter of the memory) and
llama.cpp's GPU kernels (Vulkan on Windows and Linux, Metal on Macs) do the work. The mesh is still
used for models too big for one computer, or when the engine is set to "mesh".

llama-server comes from llama.cpp's own GitHub releases, downloaded once into ~/.omnicompute/llama.cpp
(about 30-60 MB), or from OMNICOMPUTE_LLAMA_SERVER / the PATH when it is already installed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import platform
import re
import shutil
import socket
import subprocess
import time
import urllib.request
import zipfile
from pathlib import Path

log = logging.getLogger("omnicompute.llamacpp")

RELEASES = "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=8"
EXE = "llama-server.exe" if os.name == "nt" else "llama-server"
DEFAULTS = {"engine": "auto", "gpu_layers": "auto", "threads": 0, "flash_attn": True, "context": 8192}


def physical_cores() -> int:
    try:
        import psutil  # optional

        return psutil.cpu_count(logical=False) or 1
    except Exception:  # noqa: BLE001
        return max(1, (os.cpu_count() or 2) // 2)


def asset_patterns() -> list[str]:
    """Release file name parts to look for on this computer, best first."""
    import sys

    if hasattr(sys, "getandroidapilevel"):  # Linux builds need glibc, which Android doesn't have
        return []
    system, machine = platform.system(), platform.machine().lower()
    arm = machine in ("arm64", "aarch64")
    if system == "Windows":
        return ["bin-win-cpu-arm64"] if arm else ["bin-win-vulkan-x64", "bin-win-cpu-x64", "bin-win-avx2-x64"]
    if system == "Darwin":
        return ["bin-macos-arm64"] if arm else ["bin-macos-x64"]
    return (["bin-ubuntu-arm64", "bin-ubuntu-vulkan-arm64"] if arm else ["bin-ubuntu-vulkan-x64", "bin-ubuntu-x64"])


def find_server(home: Path) -> str | None:
    env = os.environ.get("OMNICOMPUTE_LLAMA_SERVER")
    if env:
        return env if Path(env).exists() or shutil.which(env) else None
    mine = sorted((home / "llama.cpp").rglob(EXE)) if (home / "llama.cpp").exists() else []
    if mine:
        return str(mine[0])
    return shutil.which("llama-server")


def install(home: Path, progress=lambda text: None, cpu_only: bool = False) -> str:
    """Download llama-server from llama.cpp's latest release. Returns its path. cpu_only: the
    plain CPU build, for computers whose GPU driver has no Vulkan."""
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "omnicompute"}
    if os.environ.get("GITHUB_TOKEN"):  # CI machines share an address, and its rate limit
        headers["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"
    req = urllib.request.Request(RELEASES, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as r:
        releases = json.load(r)
    # the newest release whose builds are all uploaded (a release appears before its files do)
    rel, pick, assets, seen = None, None, {}, []
    for rel in releases:
        assets = {a["name"]: a["browser_download_url"] for a in rel.get("assets", [])
                  if a["name"].endswith((".zip", ".tar.gz"))}
        seen = seen or sorted(assets)
        patterns = [p for p in asset_patterns() if not (cpu_only and "vulkan" in p)]
        pick = next((n for p in patterns for n in assets if p in n and "cudart" not in n), None)
        if pick:
            break
    if pick is None:
        raise RuntimeError(f"llama.cpp has no ready-made build for this computer ({platform.system()} "
                           f"{platform.machine()}); its releases have {', '.join(seen)[:600]}")
    dest = home / "llama.cpp"
    tmp = home / ("llama.cpp.zip" if pick.endswith(".zip") else "llama.cpp.tar.gz")
    home.mkdir(parents=True, exist_ok=True)
    progress(f"Downloading llama.cpp {rel.get('tag_name', '')} ({pick})…")
    with urllib.request.urlopen(urllib.request.Request(assets[pick], headers={"User-Agent": "omnicompute"}), timeout=120) as r, \
            open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    if pick.endswith(".zip"):
        with zipfile.ZipFile(tmp) as z:
            z.extractall(dest)
    else:
        import tarfile

        with tarfile.open(tmp) as t:
            t.extractall(dest, filter="data") if hasattr(tarfile, "data_filter") else t.extractall(dest)
    tmp.unlink(missing_ok=True)
    (dest / "version.txt").write_text(f"{rel.get('tag_name', '')} {pick}\n", encoding="utf-8")
    exe = find_server(home)
    if not exe:
        raise RuntimeError("the llama.cpp download has no llama-server in it")
    if os.name != "nt":
        for f in Path(exe).parent.iterdir():
            if f.is_file() and not f.suffix:
                f.chmod(f.stat().st_mode | 0o755)
    return exe


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def offload(settings: dict, n_layer: int, limits: dict | None, gpu_share: float | None = None) -> int:
    """GPU layers, like LM Studio's "GPU offload" slider: all of them when this computer may use
    its GPU, none when it is set to CPU only, and in proportion when it may use both."""
    g = settings.get("gpu_layers", "auto")
    if g not in ("auto", None, ""):
        return max(0, int(g))
    limits = limits or {}
    if limits.get("gpu") is False:
        return 0
    if limits.get("use_cpu") is not False and gpu_share is not None and gpu_share < 1:
        return round(n_layer * gpu_share)
    return 999  # everything on the GPU (llama.cpp falls back to the CPU when there is none)


class Engine:
    """One llama-server process serving one GGUF file."""

    def __init__(self, home: Path, path: Path, cfg: dict):
        self.home, self.path, self.cfg = home, Path(path), cfg
        self.proc: subprocess.Popen | None = None
        self.port = 0
        self.state, self.error, self.note = "starting", None, ""
        self.args: list[str] = []
        self.started_at = 0.0
        self._ready: asyncio.Task | None = None
        self.log_file = home / "llama-server.log"

    # ---------- lifecycle

    def settings_args(self, settings: dict, limits: dict | None, gpu_share: float | None) -> list[str]:
        ctx = min(int(settings.get("context") or DEFAULTS["context"]), int(self.cfg.get("n_ctx") or 1 << 20))
        threads = int(settings.get("threads") or (limits or {}).get("cores") or physical_cores())
        args = ["-c", str(ctx), "-ngl", str(offload(settings, self.cfg.get("n_layer", 0), limits, gpu_share)),
                "-t", str(threads), "-np", "1", "--jinja"]
        if settings.get("flash_attn", True):
            args += ["-fa", "auto"]
        return args

    def start(self, settings: dict, limits: dict | None = None, gpu_share: float | None = None) -> None:
        """(Re)start in the background; ready() waits for it."""
        args = self.settings_args(settings, limits, gpu_share)
        if self.proc and self.proc.poll() is None and args == self.args and self.state != "error":
            return
        self.stop()
        self.args, self.state, self.error = args, "starting", None
        self._ready = asyncio.ensure_future(self._start())

    async def _start(self) -> None:
        try:
            exe = find_server(self.home)
            if not exe:
                self.note = "Downloading llama.cpp (the engine LM Studio uses), once…"
                exe = await asyncio.to_thread(install, self.home, lambda t: setattr(self, "note", t))
            self.port = _free_port()
            cmd = [exe, "-m", str(self.path), "--host", "127.0.0.1", "--port", str(self.port), "--no-webui", *self.args]
            self.note, self.state, self.started_at = "Loading the model into memory…", "loading", time.monotonic()
            log.info("starting %s", " ".join(cmd))
            self.proc = self._popen(cmd)
            await self._wait_healthy()
            if self.proc.poll() is not None:  # an older llama-server: only the basic options
                basic = cmd[:7] + [a for k in ("-c", "-ngl", "-t") for a in (k, self.args[self.args.index(k) + 1])]
                self.proc = self._popen(basic)
                await self._wait_healthy()
            if self.proc.poll() is not None and "vulkan" in self._build() and not os.environ.get("OMNICOMPUTE_LLAMA_SERVER"):
                # the GPU build needs a Vulkan driver; without one, the CPU build still beats the mesh
                self.note = "This GPU has no Vulkan driver; downloading llama.cpp's CPU build…"
                exe = await asyncio.to_thread(install, self.home, lambda t: setattr(self, "note", t), True)
                self.proc = self._popen([exe, *cmd[1:]])
                await self._wait_healthy()
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server stopped: {self._tail()}")
            self.state, self.note = "ready", ""
        except Exception as e:  # noqa: BLE001 - shown on the Models tab; the mesh takes over
            self.state, self.error = "error", str(e)
            log.warning("fast engine unavailable: %s", e)

    def _build(self) -> str:
        try:
            return (self.home / "llama.cpp" / "version.txt").read_text(encoding="utf-8")
        except OSError:
            return ""

    def _popen(self, cmd: list[str]) -> subprocess.Popen:
        self.home.mkdir(parents=True, exist_ok=True)
        out = open(self.log_file, "ab")
        flags = 0x08000000 if os.name == "nt" else 0  # no console window
        return subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, creationflags=flags)

    async def _wait_healthy(self, timeout: float = 900) -> None:
        import aiohttp

        t0 = time.monotonic()
        async with aiohttp.ClientSession() as s:
            while time.monotonic() - t0 < timeout:
                if self.proc.poll() is not None:
                    return
                try:
                    async with s.get(f"http://127.0.0.1:{self.port}/health", timeout=aiohttp.ClientTimeout(total=2)) as r:
                        if r.status == 200:
                            return
                except (aiohttp.ClientError, asyncio.TimeoutError):
                    pass
                self.note = f"Loading the model into memory… {time.monotonic() - t0:.0f}s"
                await asyncio.sleep(0.5)
        raise RuntimeError("llama-server took too long to load the model")

    async def ready(self) -> bool:
        if self._ready:
            await asyncio.shield(self._ready)
        return self.state == "ready" and self.proc is not None and self.proc.poll() is None

    def _tail(self) -> str:
        try:
            lines = self.log_file.read_text(encoding="utf-8", errors="replace").strip().splitlines()
            return " / ".join(lines[-3:])[-400:]
        except OSError:
            return "no log"

    def stop(self) -> None:
        if self._ready and not self._ready.done():
            self._ready.cancel()
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None
        self.state = "stopped"

    def info(self) -> dict:
        ngl = self.args[self.args.index("-ngl") + 1] if "-ngl" in self.args else None
        return {"state": self.state, "error": self.error, "note": self.note, "gpu_layers": ngl,
                "threads": self.args[self.args.index("-t") + 1] if "-t" in self.args else None,
                "context": self.args[self.args.index("-c") + 1] if "-c" in self.args else None}

    # ---------- generation

    async def generate(self, prompt, max_new: int, temperature: float, chat: bool):
        """Yields ("thinking" | "text", piece) and finally ("timings", {...})."""
        import aiohttp

        url = f"http://127.0.0.1:{self.port}"
        if isinstance(prompt, list) or chat:
            msgs = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
            path, body = "/v1/chat/completions", {"messages": msgs, "stream": True, "max_tokens": max_new,
                                                  "temperature": temperature, "cache_prompt": True}
        else:
            path, body = "/completion", {"prompt": prompt, "n_predict": max_new, "temperature": temperature,
                                         "stream": True, "cache_prompt": True}
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, sock_read=900)) as s:
            async with s.post(url + path, json=body) as r:
                if r.status != 200:
                    raise RuntimeError(f"llama-server said {r.status}: {(await r.text())[:300]}")
                async for raw in r.content:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    ev = json.loads(data)
                    if "choices" in ev:
                        delta = (ev["choices"] or [{}])[0].get("delta") or {}
                        if delta.get("reasoning_content"):
                            yield "thinking", delta["reasoning_content"]
                        if delta.get("content"):
                            yield "text", delta["content"]
                    elif ev.get("content"):
                        yield "text", ev["content"]
                    if ev.get("timings"):
                        yield "timings", ev["timings"]


def split_thinking(text: str) -> tuple[str, str]:
    """('thinking', 'answer') from text with <think>…</think> or Gemma's thought channel markers,
    for builds that don't separate reasoning themselves."""
    m = re.match(r"\s*<think>(.*?)(?:</think>|$)(.*)", text, re.S)
    if m:
        return m.group(1).strip(), m.group(2).lstrip()
    return "", text


class RemoteSource:
    """A model that lives on another of the owner's devices (it downloaded the file and runs it
    on llama.cpp there). The main computer only has its settings."""

    def __init__(self, cfg: dict, node: str, repo: str):
        self.cfg, self.remote_node, self.repo = cfg, node, repo


class RoughTokenizer:
    """Stands in for the tokenizer of a model on another device: only counts, roughly."""

    eos_ids: set = set()
    has_chat_template = True

    def encode(self, prompt, chat: bool = False) -> list[int]:
        text = prompt if isinstance(prompt, str) else " ".join(str(m.get("content", "")) for m in prompt)
        return [0] * max(1, len(text) // 4)

    def decode(self, ids) -> str:
        return ""


class RemoteEngine:
    """Engine's interface for llama.cpp running on another device, through the mesh."""

    def __init__(self, endpoint, node: str, repo: str, cfg: dict):
        self.endpoint, self.node, self.repo, self.cfg = endpoint, node, repo, cfg
        self.state, self.error, self.note = "loading", None, f"Loading on {node}…"
        self.queues: dict[str, asyncio.Queue] = {}
        self._info: dict = {}
        self._ready: asyncio.Task | None = None

    def start(self, settings: dict, limits: dict | None = None, gpu_share: float | None = None) -> None:
        if self._ready is None or (self._ready.done() and self.state == "error"):
            self._ready = asyncio.ensure_future(self._poll())

    async def _poll(self) -> None:
        t0 = time.monotonic()
        while time.monotonic() - t0 < 1800:
            try:
                h, _ = await self.endpoint.request(self.node, {"op": "engine_status"}, timeout=15)
            except Exception as e:  # noqa: BLE001
                self.state, self.error = "error", f"{self.node} doesn't answer: {e}"
                return
            g = h.get("engine") or {}
            self._info = g
            if g.get("state") in ("ready", "error"):
                self.state, self.error = g["state"], g.get("error")
                self.note = ""
                return
            self.note = f"{self.node}: {g.get('note') or 'loading the model…'}"
            await asyncio.sleep(1)
        self.state, self.error = "error", f"{self.node} took too long to load the model"

    async def ready(self) -> bool:
        if self._ready:
            await asyncio.shield(self._ready)
        return self.state == "ready"

    def stop(self) -> None:
        asyncio.ensure_future(self.endpoint.notify(self.node, {"op": "engine_unload"}))
        self.state = "stopped"

    def info(self) -> dict:
        return {"state": self.state, "error": self.error, "note": self.note, "device": self.node,
                "gpu_layers": self._info.get("gpu_layers"), "threads": self._info.get("threads"),
                "context": self._info.get("context")}

    def event(self, header: dict) -> None:
        q = self.queues.get(header.get("req"))
        if q is not None:
            q.put_nowait(header)

    async def generate(self, prompt, max_new: int, temperature: float, chat: bool):
        req = os.urandom(8).hex()
        q = self.queues[req] = asyncio.Queue()
        try:
            await self.endpoint.notify(self.node, {"op": "engine_generate", "req": req, "prompt": prompt,
                                                   "max_new": max_new, "temperature": temperature, "chat": chat})
            while True:
                try:
                    ev = await asyncio.wait_for(q.get(), 900)
                except asyncio.TimeoutError:
                    raise RuntimeError(f"{self.node} stopped answering")
                if ev["what"] == "end":
                    if ev.get("error"):
                        raise RuntimeError(f"{self.node}: {ev['error']}")
                    return
                yield ev["what"], ev["piece"]
        finally:
            self.queues.pop(req, None)
            asyncio.ensure_future(self.endpoint.notify(self.node, {"op": "engine_stop", "req": req}))
