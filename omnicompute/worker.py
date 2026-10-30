"""Worker role: hosts layer shards and runs them as part of a device-to-device chain.

Ops (the browser worker in web/worker.js implements the same ones):
  load     {model, cfg, start, end} + tensors   -> keep a shard in memory
  forward  {model, job, pos, route, hop, req, reply_to} + {"x"}
           run our layers, then hand the activations straight to the next
           device in `route` (or the logits back to `reply_to` if we are last)
  drop     {model, job}                          -> free that job's KV cache
  preconnect {peer}                              -> start hole punching to the next device
  set_limits {limits}                            -> change this device's usage limits (see limits.py)
  credit   {tokens, layers, compute_s, ...}      -> how much this device did for the last answer

Only the hub may send the control ops (everything except forward and ping), so a device that
joined with a contribute-only link cannot reconfigure the others.
"""

from __future__ import annotations

import asyncio
import logging
import os
import platform
import time

import numpy as np

from . import limits as L
from .project import home_dir
from .gpu import GPUMismatch
from .model import Shard
from .protocol import decode_tensors, encode_tensors

log = logging.getLogger("omnicompute.worker")


def benchmark() -> float:
    """Effective GFLOP/s for token-by-token generation, used to size each device's share of
    layers and to estimate speed. Generation multiplies one vector by every weight matrix, which
    is limited by memory bandwidth, so this times a vector x matrix product on a 4096x4096
    float32 matrix (64 MB, bigger than CPU caches, the size of a 7B model's attention weights)."""
    w = np.random.default_rng(0).standard_normal((4096, 4096), dtype=np.float32)
    x = np.ones((1, 4096), np.float32)
    x @ w
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < 0.4:
        x @ w
        n += 1
    return round(n * 2 * 4096 * 4096 / (time.perf_counter() - t0) / 1e9, 2)


def total_memory_mb() -> int:
    try:
        if hasattr(os, "sysconf") and "SC_PHYS_PAGES" in os.sysconf_names:
            return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") // (1024 * 1024)
    except (OSError, ValueError):
        pass
    if platform.system() == "Windows":
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [
                (n, ctypes.c_ulonglong) for n in ("ullTotalPhys", "ullAvailPhys", "ullTotalPageFile",
                                                  "ullAvailPageFile", "ullTotalVirtual", "ullAvailVirtual",
                                                  "ullAvailExtendedVirtual")]
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(stat)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        return stat.ullTotalPhys // (1024 * 1024)
    return 4096


def worker_info(mem_fraction: float = 0.5, gpu=None) -> dict:
    cpu = float(os.environ.get("OMNICOMPUTE_SCORE") or benchmark())
    info = {"role": "worker", "kind": "native", "platform": f"{platform.system()} {platform.machine()}",
            "score": cpu, "cpu_score": cpu, "mem_mb": int(total_memory_mb() * mem_fraction),
            "cores_total": os.cpu_count() or 1}
    if gpu is not None:
        info["gpu"] = {"name": gpu.name, "score": gpu.benchmark(), "via": getattr(gpu, "backend", "webgpu")}
    return info


CONTROL_OPS = {"load", "unload", "engine_load", "engine_status", "engine_generate", "engine_stop",
               "engine_unload", "set_limits", "drop", "preconnect", "info", "credit", "live", "uninstall"}
HYBRID_MIN = 0.15  # below this share, splitting a matrix between CPU and GPU costs more than it saves


class Worker:
    def __init__(self, limits: L.LimitStore | None = None, gpu=None):
        self.node = None  # set by attach()
        self.shards: dict[str, Shard] = {}
        self.limits = limits
        self.gpu = gpu  # gpu.GPU or None
        self.steps = 0
        self.helped = {"answers": 0, "tokens": 0, "compute_s": 0.0}
        self.power = L.power_state()
        self.gpu_broken = False
        self._busy = asyncio.Lock()  # load and forward never overlap
        self.on_uninstall = None  # set by the CLI: what "Remove device" on the main computer does
        self._reported = None
        self.engine = None  # llamacpp.Engine running a whole model here (see _engine_*)
        self.engine_cfg: dict | None = None
        self.loader = None
        self._engine_stop: set[str] = set()

    def attach(self, node) -> "Worker":
        self.node = node
        node.handler = self.handle
        node.info.update(self.limit_info())
        return self

    def use_gpu(self) -> bool:
        """The GPU computes at least part of each step (see engines)."""
        return self.engines()[0] != "cpu"

    def engines(self) -> tuple[str, float]:
        """("cpu" | "gpu" | "both", share of each matrix's columns the CPU computes). With both
        allowed, the split follows each engine's speed times its usage limit; a tiny share isn't
        worth the hand-off, so then the faster one does it all."""
        limits = self.limits.limits if self.limits else L.DEFAULT
        info = self.node.info if self.node else {}
        if self.gpu is None or self.gpu_broken or not limits.get("gpu", True):
            return "cpu", 1.0
        if not limits.get("use_cpu", True) or os.environ.get("OMNICOMPUTE_GPU") == "force":
            return "gpu", 0.0
        total = info.get("cores_total") or 1
        cores = L.effective_cores(limits, self.power)
        cs = info.get("cpu_score", 0) * (min(cores, total) / total if cores else 1) * L.effective(limits, power=self.power)[0]
        gs = info.get("gpu", {}).get("score", 0) * L.effective_for(limits, True, power=self.power)[0]
        if cs + gs <= 0:
            return ("gpu", 0.0) if gs >= cs else ("cpu", 1.0)
        share = cs / (cs + gs)
        return ("gpu", 0.0) if share < HYBRID_MIN else ("cpu", 1.0) if share > 1 - HYBRID_MIN else ("both", round(share, 3))

    def limit_info(self) -> dict:
        limits = self.limits.limits if self.limits else dict(L.DEFAULT)
        ready = self.node is not None and "cpu_score" in self.node.info
        mode, _ = self.engines() if ready else ("cpu", 1.0)
        cpu, why = L.effective_for(limits, mode == "gpu", power=self.power)
        if mode == "both":  # each engine rests by its own share; the step as a whole by the larger limit
            g, gwhy = L.effective_for(limits, True, power=self.power)
            cpu, why = max((cpu, why), (g, gwhy), key=lambda t: t[0])
        out = {"limit": cpu, "limit_reason": why, "limits": limits, "power": self.power,
               "cores": L.effective_cores(limits, self.power)}
        if ready:
            out["uses"] = {"gpu": "gpu", "cpu": "cpu", "both": "gpu+cpu"}[mode]
            total = self.node.info.get("cores_total") or 1
            share = min(out["cores"], total) / total if out["cores"] else 1  # fewer cores, less speed
            cs, gs = self.node.info["cpu_score"] * share, self.node.info.get("gpu", {}).get("score", 0)
            if mode == "both":  # speed at this limit, as the planner scales score by it
                c, g = L.effective(limits, power=self.power)[0], L.effective_for(limits, True, power=self.power)[0]
                out["score"] = round((cs * c + gs * g) / max(1, cpu), 2)
            else:
                out["score"] = gs if mode == "gpu" else round(cs, 2)
        return out

    async def report_limits(self, force: bool = False) -> None:
        self.power = L.power_state()
        info = self.limit_info()
        key = (info["limit"], info["limit_reason"], str(info["limits"]), info.get("uses"), info["power"])
        if force or key != self._reported:
            self._reported = key
            await self.node.update_info(info)

    async def watch_schedule(self) -> None:
        """Re-check the schedule (and the limits file, which `omnicompute limits` changes) every few
        seconds, and tell the hub when the limit in force changes."""
        while True:
            try:
                if self.limits is not None:
                    self.limits.reload_if_changed()
                    if self.node is not None and self.node.info.get("kind") == "native":
                        from .awake import set_awake

                        set_awake(bool(self.limits.limits.get("awake")))
                await self.report_limits()
            except Exception as e:  # hub connection gone: the reconnect loop takes over
                log.debug("could not report limits: %s", e)
            await asyncio.sleep(5)

    # ---------- the whole model on this device, on llama.cpp: for a main computer without the
    # disk space, this device downloads the GGUF file and runs it; the main computer only relays
    # the chat. Only the mesh owner's own devices get this (the hub never asks helpers).

    def _engine_close(self) -> None:
        if self.engine is not None:
            self.engine.stop()
        self.engine, self.engine_cfg = None, None

    async def _engine_load(self, h: dict):
        from .hf import Loader
        from .llamacpp import Engine

        self._engine_close()
        if self.loader is None:
            self.loader = Loader(home_dir() / "models", None)
        loop = asyncio.get_running_loop()

        async def ready(source):
            path = getattr(source, "path", None)
            if path is None or str(path).lower()[-5:] != ".gguf":
                raise RuntimeError("only GGUF models can run on another device (use a GGUF repo, e.g. org/name:Q4_K_M)")
            self.engine = Engine(home_dir(), path, source.cfg)
            self.engine.start(h.get("settings") or {}, self.limits.limits if self.limits else None)
            self.engine_cfg = source.cfg

        def on_ready(source):
            asyncio.run_coroutine_threadsafe(ready(source), loop).result(60)

        try:
            self.loader.start(str(h["repo"]), h.get("token"), on_ready)
        except RuntimeError as e:
            return {"ok": False, "error": str(e)}, b""
        return {"ok": True}, b""

    async def _engine_generate(self, peer: str, h: dict) -> None:
        req = h["req"]
        try:
            if self.engine is None or not await self.engine.ready():
                raise RuntimeError(self.engine.error if self.engine else "no model loaded on this device")
            async for kind, piece in self.engine.generate(h["prompt"], int(h["max_new"]), float(h["temperature"]),
                                                          bool(h["chat"])):
                if req in self._engine_stop:
                    break
                await self.node.notify(peer, {"op": "engine_event", "req": req, "what": kind, "piece": piece})
            await self.node.notify(peer, {"op": "engine_event", "req": req, "what": "end"})
        except Exception as e:  # noqa: BLE001 - shown in the chat
            await self.node.notify(peer, {"op": "engine_event", "req": req, "what": "end", "error": str(e)})
        finally:
            self._engine_stop.discard(req)

    async def handle(self, peer_id: str, header: dict, payload: bytes):
        op = header.get("op")
        if op in CONTROL_OPS and peer_id != "hub":
            raise PermissionError(f"{op} is only accepted from the hub")
        if op == "load":
            async with self._busy:
                return await self._load(header, payload)
        if op == "engine_load":
            return await self._engine_load(header)
        if op == "engine_status":
            d = self.loader.current if self.loader else None
            return {"download": d.as_dict() if d else None, "engine": self.engine.info() if self.engine else None,
                    "cfg": self.engine_cfg}, b""
        if op == "engine_generate":
            asyncio.ensure_future(self._engine_generate(peer_id, header))
            return None
        if op == "engine_stop":
            self._engine_stop.add(header["req"])
            return None
        if op == "engine_unload":
            self._engine_close()
            return None
        if op == "unload":  # the main computer unloaded or switched models: free the memory now
            async with self._busy:
                self.shards.clear()
                if self.gpu is not None:
                    self.gpu.clear()
                import gc

                gc.collect()
            return None
        if op == "uninstall":  # the main computer removed this device from the mesh
            if self.on_uninstall is None:
                raise RuntimeError("this device can't uninstall itself")
            removed = self.on_uninstall()
            return {"ok": True, "removed": removed}, b""
        if op == "live":  # the answer this device is helping to write (shown by browser workers)
            return None
        if op == "credit":
            self.helped["answers"] += 1
            self.helped["tokens"] += header.get("tokens", 0)
            self.helped["compute_s"] += header.get("compute_s", 0.0)
            log.info("helped with an answer: %s%% of the work, %.1fs of compute", header.get("share"),
                     header.get("compute_s", 0.0))
            return None
        if op == "forward":
            await self._forward(header, payload)
            return None
        if op == "preconnect":  # the hub tells us who is next in the chain; open a direct path early
            if hasattr(self.node, "connect_direct"):
                asyncio.ensure_future(self.node.connect_direct(header["peer"]))
            return None
        if op == "drop":
            shard = self.shards.get(header.get("model"))
            if shard:
                shard.drop(header["job"])
            return None
        if op == "set_limits":
            if self.limits is None:
                self.limits = L.LimitStore(home_dir() / f"limits-{self.node.node_id}.json")
            self.limits.set(header["limits"])  # a GPU on/off change applies the next time layers load
            await self.report_limits(force=True)
            return {"limits": self.limits.limits}, b""
        if op == "ping":
            return {"pong": True}, b""
        if op == "info":
            return {"shards": {m: [s.start, s.end] for m, s in self.shards.items()}}, b""
        raise ValueError(f"unknown op {op}")

    async def _load(self, header: dict, payload: bytes):
        if self.gpu is not None:
            self.gpu.clear()
        t = decode_tensors(payload)
        mode, cpu_share = self.engines()
        gpu = mode != "cpu"

        def on_cpu(t):  # weights travel as float16; expand once so every step runs on fast float32 BLAS
            t = {k: v.astype(np.float32) if v.dtype == np.float16 else v for k, v in t.items()}
            return Shard(header["cfg"], t, header["start"], header["end"])

        if gpu:
            shard = Shard(header["cfg"], dict(t), header["start"], header["end"])
            try:
                await asyncio.to_thread(shard.use_gpu, self.gpu, cpu_share if mode == "both" else 0.0)
            except GPUMismatch as e:  # a GPU driver that computes wrong answers: never use it again
                log.warning("the GPU gave wrong results (%s); using the CPU instead", e)
                self.gpu_broken = True
                self.gpu.clear()
                gpu = False
                shard = on_cpu(t)
                await self.report_limits(force=True)
        else:
            shard = on_cpu(t)
        self.shards = {header["model"]: shard}
        log.info("loaded %s layers %d-%d on the %s", header["model"], header["start"], header["end"],
                 (f"GPU ({self.gpu.name})" + (f" and CPU ({cpu_share:.0%} of the work)" if shard.cpu_share else "")) if gpu else "CPU")
        return {"ok": True}, b""

    async def _forward(self, h: dict, payload: bytes) -> None:
        try:
            hop = h["route"][h["hop"]]
            shard = self.shards.get(h["model"])
            if shard is None or [shard.start, shard.end] != [hop["start"], hop["end"]]:
                raise RuntimeError("shard not loaded here (device restarted?)")
            info = self.limit_info()
            cpu = info["limit"]
            if cpu <= 0:
                raise RuntimeError("paused by its usage limit")
            t0 = time.perf_counter()
            async with self._busy:
                out = await asyncio.to_thread(_limited, info["cores"], shard.forward_ex, h["job"], decode_tensors(payload), h["pos"])
            self.steps += 1
            busy = time.perf_counter() - t0
            h = {**h, "timings": h.get("timings", []) + [[self.node.node_id, round(busy, 5)]]}
            rest = L.rest_after(busy, cpu)
            if rest:
                await asyncio.sleep(rest)
        except Exception as e:
            await self.node.notify(h["reply_to"], {"op": "error", "req": h["req"], "error": str(e),
                                                   "node": self.node.node_id})
            return
        if h["hop"] + 1 < len(h["route"]):
            nxt = h["route"][h["hop"] + 1]["node"]
            await self.node.notify(nxt, {**h, "hop": h["hop"] + 1}, encode_tensors(out))
        else:
            await self.node.notify(h["reply_to"], {"op": "result", "req": h["req"], "timings": h["timings"]},
                                   encode_tensors({"logits": out["x"]}))


def _limited(cores, fn, *args):
    """Run fn with numpy's thread pool capped to `cores` (needs threadpoolctl)."""
    if cores:
        try:
            from threadpoolctl import threadpool_limits

            with threadpool_limits(cores):
                return fn(*args)
        except ImportError:
            pass
    return fn(*args)
