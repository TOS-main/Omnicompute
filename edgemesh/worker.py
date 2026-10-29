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
from .gpu import GPUMismatch
from .model import Shard
from .protocol import decode_tensors, encode_tensors

log = logging.getLogger("edgemesh.worker")


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
    cpu = float(os.environ.get("EDGEMESH_SCORE") or benchmark())
    info = {"role": "worker", "kind": "native", "platform": f"{platform.system()} {platform.machine()}",
            "score": cpu, "cpu_score": cpu, "mem_mb": int(total_memory_mb() * mem_fraction)}
    if gpu is not None:
        info["gpu"] = {"name": gpu.name, "score": gpu.benchmark(), "via": getattr(gpu, "backend", "webgpu")}
    return info


CONTROL_OPS = {"load", "set_limits", "drop", "preconnect", "info", "credit", "live", "uninstall"}


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

    def attach(self, node) -> "Worker":
        self.node = node
        node.handler = self.handle
        node.info.update(self.limit_info())
        return self

    def use_gpu(self) -> bool:
        """GPU on, allowed by the limits, and faster than this machine's CPU."""
        limits = self.limits.limits if self.limits else L.DEFAULT
        info = self.node.info if self.node else {}
        faster = info.get("gpu", {}).get("score", 0) > info.get("cpu_score", 0)
        return (self.gpu is not None and not self.gpu_broken and limits.get("gpu", True)
                and (faster or os.environ.get("EDGEMESH_GPU") == "force"))

    def limit_info(self) -> dict:
        limits = self.limits.limits if self.limits else dict(L.DEFAULT)
        cpu, why = L.effective(limits, power=self.power)
        out = {"limit": cpu, "limit_reason": why, "limits": limits, "power": self.power,
               "cores": L.effective_cores(limits, self.power)}
        if self.node is not None and "cpu_score" in self.node.info:
            gpu = self.use_gpu()
            out["uses"] = "gpu" if gpu else "cpu"
            out["score"] = self.node.info["gpu"]["score"] if gpu else self.node.info["cpu_score"]
        return out

    async def report_limits(self, force: bool = False) -> None:
        self.power = L.power_state()
        info = self.limit_info()
        key = (info["limit"], info["limit_reason"], str(info["limits"]), info.get("uses"), info["power"])
        if force or key != self._reported:
            self._reported = key
            await self.node.update_info(info)

    async def watch_schedule(self) -> None:
        """Re-check the schedule every 20 s and tell the hub when the limit in force changes."""
        while True:
            try:
                await self.report_limits()
            except Exception as e:  # hub connection gone: the reconnect loop takes over
                log.debug("could not report limits: %s", e)
            await asyncio.sleep(20)

    async def handle(self, peer_id: str, header: dict, payload: bytes):
        op = header.get("op")
        if op in CONTROL_OPS and peer_id != "hub":
            raise PermissionError(f"{op} is only accepted from the hub")
        if op == "load":
            async with self._busy:
                return await self._load(header, payload)
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
                self.limits = L.LimitStore(L.Path.home() / ".edgemesh" / f"limits-{self.node.node_id}.json")
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
        gpu = self.use_gpu()

        def on_cpu(t):  # weights travel as float16; expand once so every step runs on fast float32 BLAS
            t = {k: v.astype(np.float32) if v.dtype == np.float16 else v for k, v in t.items()}
            return Shard(header["cfg"], t, header["start"], header["end"])

        if gpu:
            shard = Shard(header["cfg"], dict(t), header["start"], header["end"])
            try:
                await asyncio.to_thread(shard.use_gpu, self.gpu)
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
                 f"GPU ({self.gpu.name})" if gpu else "CPU")
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
                y = await asyncio.to_thread(_limited, info["cores"], shard.forward, h["job"], decode_tensors(payload)["x"], h["pos"])
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
            await self.node.notify(nxt, {**h, "hop": h["hop"] + 1}, encode_tensors({"x": y}))
        else:
            await self.node.notify(h["reply_to"], {"op": "result", "req": h["req"], "timings": h["timings"]},
                                   encode_tensors({"logits": y}))


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
