"""Worker role: hosts layer shards and runs them as part of a device-to-device chain.

Ops (the browser worker in web/worker.js implements the same ones):
  load     {model, cfg, start, end} + tensors   -> keep a shard in memory
  forward  {model, job, pos, route, hop, req, reply_to} + {"x"}
           run our layers, then hand the activations straight to the next
           device in `route` (or the logits back to `reply_to` if we are last)
  drop     {model, job}                          -> free that job's KV cache
  preconnect {peer}                              -> start hole punching to the next device
"""

from __future__ import annotations

import asyncio
import logging
import os
import platform
import time

import numpy as np

from .model import Shard
from .protocol import decode_tensors, encode_tensors

log = logging.getLogger("edgemesh.worker")


def benchmark() -> float:
    """Rough matmul throughput in GFLOP/s, used to size each device's share of layers."""
    a = np.random.default_rng(0).standard_normal((256, 256), dtype=np.float32)
    a @ a
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < 0.3:
        a @ a
        n += 1
    return round(n * 2 * 256 ** 3 / (time.perf_counter() - t0) / 1e9, 2)


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


def worker_info(mem_fraction: float = 0.5) -> dict:
    return {"role": "worker", "kind": "native", "platform": f"{platform.system()} {platform.machine()}",
            "score": float(os.environ.get("EDGEMESH_SCORE") or benchmark()), "mem_mb": int(total_memory_mb() * mem_fraction)}


class Worker:
    def __init__(self):
        self.node = None  # set by attach()
        self.shards: dict[str, Shard] = {}

    def attach(self, node) -> "Worker":
        self.node = node
        node.handler = self.handle
        return self

    async def handle(self, peer_id: str, header: dict, payload: bytes):
        op = header.get("op")
        if op == "load":
            t = decode_tensors(payload)
            self.shards = {header["model"]: Shard(header["cfg"], t, header["start"], header["end"])}
            log.info("loaded %s layers %d-%d", header["model"], header["start"], header["end"])
            return {"ok": True}, b""
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
        if op == "info":
            return {"shards": {m: [s.start, s.end] for m, s in self.shards.items()}}, b""
        raise ValueError(f"unknown op {op}")

    async def _forward(self, h: dict, payload: bytes) -> None:
        try:
            hop = h["route"][h["hop"]]
            shard = self.shards.get(h["model"])
            if shard is None or [shard.start, shard.end] != [hop["start"], hop["end"]]:
                raise RuntimeError("shard not loaded here (device restarted?)")
            y = await asyncio.to_thread(shard.forward, h["job"], decode_tensors(payload)["x"], h["pos"])
        except Exception as e:
            await self.node.notify(h["reply_to"], {"op": "error", "req": h["req"], "error": str(e),
                                                   "node": self.node.node_id})
            return
        if h["hop"] + 1 < len(h["route"]):
            nxt = h["route"][h["hop"] + 1]["node"]
            await self.node.notify(nxt, {**h, "hop": h["hop"] + 1}, encode_tensors({"x": y}))
        else:
            await self.node.notify(h["reply_to"], {"op": "result", "req": h["req"]}, encode_tensors({"logits": y}))
