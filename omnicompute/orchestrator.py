"""Runs a generation across the mesh. Lives on the hub, next to the model.

For each token it sends one message into a chain of devices
(hub -> A -> B -> C -> hub); devices pass activations to each other directly
when hole punching succeeded, otherwise through the relay. If a device drops
out mid-generation, the orchestrator re-plans over the devices still online
and replays the sequence so far.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time

import numpy as np

from .estimate import best_plan
from .model import shard_tensors
from .protocol import decode_tensors, encode_tensors
from .tokenizer import load_tokenizer

log = logging.getLogger("omnicompute.orchestrator")

HOP_TIMEOUT = 60.0
MAX_REPLANS = 3


class Orchestrator:
    def __init__(self, endpoint, source, list_workers):
        """endpoint: a Messenger whose handler we own. source: DictSource or HFSource.
        list_workers: () -> [{"node_id", "info"}]."""
        self.endpoint = endpoint
        endpoint.handler = self.handle
        self.list_workers = list_workers
        self.set_model(source)
        self._waiters: dict[str, asyncio.Future] = {}
        self._routes: dict[str, set[str]] = {}  # req -> devices the in-flight step depends on
        self._lock = asyncio.Lock()  # one generation at a time in this MVP
        self.last_run: dict | None = None
        self.estimate: dict | None = None
        self.busy: dict[str, float] = {}  # node -> seconds spent computing in the current run
        self.helped: dict[str, dict] = {}  # node -> {"answers", "tokens", "compute_s"} since the hub started
        self.share_output = True  # show the answer on the devices computing it
        # ...including helpers on the contribute-only link (people outside the mesh); off, so a
        # public contribute link never shows anyone your chats
        self.share_with_helpers = False

    def set_model(self, source) -> None:
        """Switch models; devices receive the new layers on the next generation."""
        self.source, self.cfg = source, source.cfg
        self.model_id = f"{self.cfg.get('name', 'model')}-{os.urandom(3).hex()}"
        self.tokenizer = load_tokenizer(source)
        self.half = getattr(source, "dtype", np.float32) == np.float16
        self.plan: list[tuple[str, int, int]] = []
        self.loaded: dict[str, tuple[int, int]] = {}

    async def handle(self, peer_id: str, header: dict, payload: bytes):
        fut = self._waiters.get(header.get("req"))
        if fut is not None and not fut.done():
            if header.get("op") == "result":
                fut.set_result((decode_tensors(payload)["logits"], header.get("timings", [])))
            elif header.get("op") == "error":
                fut.set_exception(RuntimeError(f"{header.get('node')}: {header.get('error')}"))
        return None

    def node_left(self, node_id: str) -> None:
        """Called by the hub when a device disconnects: fail its in-flight steps right away."""
        self.loaded.pop(node_id, None)
        for req, nodes in list(self._routes.items()):
            fut = self._waiters.get(req)
            if node_id in nodes and fut is not None and not fut.done():
                fut.set_exception(ConnectionError(f"{node_id}: disconnected"))

    async def ensure_plan(self, exclude: set[str] = frozenset()) -> list[tuple[str, int, int]]:
        workers = [{"node_id": w["node_id"], **w["info"]} for w in self.list_workers()
                   if w["info"].get("role") == "worker" and w["node_id"] not in exclude
                   and w["info"].get("limit", 100) > 0]  # paused by its usage limit
        # weights travel as float16 but devices expand them to float32 in memory, for speed
        plan, self.estimate = best_plan(self.cfg, workers)
        loads = []
        for node, s, e in plan:
            if self.loaded.get(node) != (s, e):
                loads.append(self._load(node, s, e))
        if loads:
            await asyncio.gather(*loads)
        self.plan = plan
        for (a, _, _), (b, _, _) in zip(plan, plan[1:]):
            asyncio.ensure_future(self.endpoint.notify(a, {"op": "preconnect", "peer": b}))
        return plan

    async def _load(self, node: str, s: int, e: int) -> None:
        t0 = time.monotonic()
        source, model_id = self.source, self.model_id
        blob = await asyncio.to_thread(lambda: encode_tensors(shard_tensors(source, s, e)))
        if model_id != self.model_id:
            raise RuntimeError("model changed while loading")
        await self.endpoint.request(node, {"op": "load", "model": self.model_id, "cfg": self.cfg,
                                           "start": s, "end": e}, blob, timeout=600)
        self.loaded[node] = (s, e)
        log.info("sent layers %d-%d (%.1f MB) to %s in %.1fs", s, e, len(blob) / 1e6, node, time.monotonic() - t0)

    async def _step(self, job: str, x: np.ndarray, pos: int) -> np.ndarray:
        online = {w["node_id"] for w in self.list_workers()}
        for n, _, _ in self.plan:
            if n not in online:
                raise ConnectionError(f"{n}: disconnected")
        req = os.urandom(8).hex()
        fut = asyncio.get_running_loop().create_future()
        self._waiters[req] = fut
        self._routes[req] = {n for n, _, _ in self.plan}
        route = [{"node": n, "start": s, "end": e} for n, s, e in self.plan]
        header = {"op": "forward", "model": self.model_id, "job": job, "pos": pos, "route": route,
                  "hop": 0, "req": req, "reply_to": self.endpoint.node_id}
        try:
            await self.endpoint.notify(route[0]["node"], header, encode_tensors({"x": x}))
            logits, timings = await asyncio.wait_for(fut, HOP_TIMEOUT)
            for node, busy in timings:
                self.busy[node] = self.busy.get(node, 0.0) + busy
            return logits
        finally:
            self._waiters.pop(req, None)
            self._routes.pop(req, None)

    async def generate(self, prompt, max_new: int = 32, temperature: float = 0.0, seed: int | None = None,
                       chat: bool = False, context: int | None = None):
        """Async generator yielding events: {"plan"}, {"token", "text"}, {"replan"}, {"done", "stats"}.
        `prompt` is a string, or a list of chat messages [{"role", "content"}] for multi-turn chat."""
        async with self._lock:
            rng = np.random.default_rng(seed)
            ctx = max(16, min(int(context or self.cfg.get("ctx_default", self.cfg["n_ctx"])), self.cfg["n_ctx"]))
            ids = self.tokenizer.encode(prompt, chat=chat)[-(ctx - 1):] or [0]
            exclude: set[str] = set()
            await self.ensure_plan()
            yield {"type": "plan", "plan": [list(p) for p in self.plan], "prompt_tokens": len(ids), "context": ctx,
                   "estimate": self.estimate.get("tok_per_s") if self.estimate else None}
            job, pos, pending = os.urandom(6).hex(), 0, list(ids)
            out_ids: list[int] = []
            replans, t0, first_token, last_push = 0, time.monotonic(), None, 0.0
            self.busy = {}
            try:
                while len(out_ids) < max_new and pos + len(pending) < ctx:
                    try:
                        logits = await self._step(job, np.array(pending, np.int32), pos)
                    except (RuntimeError, asyncio.TimeoutError, ConnectionError) as e:
                        replans += 1
                        if replans > MAX_REPLANS:
                            raise RuntimeError(f"generation failed after {MAX_REPLANS} re-plans: {e}")
                        failed = self._guess_failed(e)
                        exclude |= failed
                        for n in failed:
                            self.loaded.pop(n, None)
                        log.warning("step failed (%s); re-planning without %s", e, failed or "nobody")
                        await self.ensure_plan(exclude)
                        yield {"type": "replan", "reason": str(e), "plan": [list(p) for p in self.plan]}
                        job, pos, pending = os.urandom(6).hex(), 0, ids + out_ids  # replay into fresh caches
                        continue
                    pos += len(pending)
                    if self.cfg.get("tokenizer") == "bytes":  # random demo model: keep output readable ASCII
                        logits = np.where(_PRINTABLE[: len(logits)], logits, -1e9)
                    nxt = _sample(logits, temperature, rng)
                    if nxt in self.tokenizer.eos_ids:
                        break
                    first_token = first_token or time.monotonic()
                    out_ids.append(nxt)
                    pending = [nxt]
                    elapsed = max(time.monotonic() - t0, 1e-6)
                    if time.monotonic() - last_push > 0.5:
                        last_push = time.monotonic()
                        self._push_live(self.tokenizer.decode(out_ids), False)
                    yield {"type": "token", "id": nxt, "text": self.tokenizer.decode(out_ids),
                           "load": {n: round(min(100.0, 100 * b / elapsed), 1) for n, b in self.busy.items()}}
            finally:  # also runs when the client presses Stop
                for node, _, _ in self.plan:
                    asyncio.ensure_future(self.endpoint.notify(node, {"op": "drop", "model": self.model_id, "job": job}))
                dt = time.monotonic() - t0
                gen_dt = time.monotonic() - first_token if first_token else 0
                self._push_live(self.tokenizer.decode(out_ids), True)
                contribution = self._credit(len(out_ids))
                self.last_run = {"tokens": len(out_ids), "seconds": round(dt, 2),
                                 "first_token_s": round(first_token - t0, 2) if first_token else None,
                                 "tok_per_s": round((len(out_ids) - 1) / gen_dt, 2) if gen_dt > 0 else None,
                                 "devices": len(self.plan), "finished": time.time(),
                                 "prompt_tokens": len(ids), "context": ctx,
                                 "stopped": "context" if pos + len(pending) >= ctx else
                                            "length" if len(out_ids) >= max_new else "eos",
                                 "load": {n: round(min(100.0, 100 * b / max(dt, 1e-6)), 1) for n, b in self.busy.items()},
                                 "contribution": contribution}
            yield {"type": "done", "text": self.tokenizer.decode(out_ids), "stats": self.last_run}

    def _push_live(self, text: str, done: bool) -> None:
        """Show the answer being written on the devices computing it (not the hub's own machine)."""
        if not self.share_output:
            return
        info = {w["node_id"]: w["info"] for w in self.list_workers()}
        for node, _, _ in self.plan:
            if node in info and not info[node].get("hub_machine") and (self.share_with_helpers or not info[node].get("contributor")):
                asyncio.ensure_future(self.endpoint.notify(node, {"op": "live", "text": text[-4000:], "done": done}))

    def _credit(self, tokens: int) -> list[dict]:
        """Each device's part in the answer just made: its layers and its share of the compute
        time. Also tells each device, so its own screen can show what it did."""
        total = sum(self.busy.values()) or 1e-9
        out = []
        for node, s, e in self.plan:
            busy = self.busy.get(node, 0.0)
            c = {"node_id": node, "layers": [s, e], "share": round(100 * busy / total, 1),
                 "layer_share": round(100 * (e - s) / self.cfg["n_layer"], 1), "compute_s": round(busy, 3),
                 "ms_per_token": round(1000 * busy / max(tokens, 1), 1)}
            out.append(c)
            h = self.helped.setdefault(node, {"answers": 0, "tokens": 0, "compute_s": 0.0})
            if tokens:
                h["answers"] += 1
                h["tokens"] += tokens
                h["compute_s"] = round(h["compute_s"] + busy, 3)
                asyncio.ensure_future(self.endpoint.notify(node, {"op": "credit", "tokens": tokens, **c,
                                                                  "total": h}))
        return out

    def _guess_failed(self, err: Exception) -> set[str]:
        online = {w["node_id"] for w in self.list_workers()}
        gone = {n for n, _, _ in self.plan if n not in online}
        msg = str(err)
        named = {n for n, _, _ in self.plan if msg.startswith(n + ":")}
        return gone | named


_PRINTABLE = np.zeros(256, bool)
_PRINTABLE[32:127] = True


def _sample(logits: np.ndarray, temperature: float, rng) -> int:
    if temperature <= 0:
        return int(np.argmax(logits))
    z = logits.astype(np.float64) / temperature
    p = np.exp(z - z.max())
    p /= p.sum()
    return int(rng.choice(len(p), p=p))
