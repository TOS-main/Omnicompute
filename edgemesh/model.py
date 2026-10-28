"""A GPT-2-architecture decoder in plain numpy, split into layer shards.

The weight names and layouts follow Hugging Face's GPT-2 checkpoint (Conv1D
weights stored [in, out]), so real GPT-2 weights can be dropped in with
`edgemesh convert-gpt2`. The browser worker (web/model.js) implements the
same math so phones can host shards too.

A model file is a tensor pack (see protocol.py) with a "__config__" entry
holding the JSON config as uint8 bytes.
"""

from __future__ import annotations

import json
from collections import OrderedDict

import numpy as np

from .protocol import decode_tensors, encode_tensors


def layer_keys(i: int) -> list[str]:
    p = f"h.{i}."
    return [p + k for k in ("ln_1.w", "ln_1.b", "attn.c_attn.w", "attn.c_attn.b", "attn.c_proj.w",
                            "attn.c_proj.b", "ln_2.w", "ln_2.b", "mlp.c_fc.w", "mlp.c_fc.b",
                            "mlp.c_proj.w", "mlp.c_proj.b")]


def random_model(n_layer=8, n_embd=128, n_head=4, n_ctx=256, vocab=256, seed=0) -> tuple[dict, dict]:
    """A randomly initialised model; useful for testing the mesh without downloads."""
    rng = np.random.default_rng(seed)
    cfg = {"n_layer": n_layer, "n_embd": n_embd, "n_head": n_head, "n_ctx": n_ctx, "vocab": vocab,
           "tokenizer": "bytes", "name": f"random-{n_layer}x{n_embd}"}
    D = n_embd
    w = {"wte": rng.normal(0, 0.1, (vocab, D)), "wpe": rng.normal(0, 0.02, (n_ctx, D)),
         "ln_f.w": np.ones(D), "ln_f.b": np.zeros(D)}
    for i in range(n_layer):
        k = layer_keys(i)
        shapes = [(D,), (D,), (D, 3 * D), (3 * D,), (D, D), (D,), (D,), (D,), (D, 4 * D), (4 * D,), (4 * D, D), (D,)]
        for name, shape in zip(k, shapes):
            if name.endswith("ln_1.w") or name.endswith("ln_2.w"):
                w[name] = np.ones(shape)
            elif len(shape) == 1:
                w[name] = np.zeros(shape)
            else:
                w[name] = rng.normal(0, 0.02 * (2 if "c_fc" in name or "c_attn" in name else 1), shape)
    return cfg, {k: v.astype(np.float32) for k, v in w.items()}


def save_model(path: str, cfg: dict, weights: dict) -> None:
    blob = encode_tensors({"__config__": np.frombuffer(json.dumps(cfg).encode(), np.uint8).astype(np.int32),
                           **weights})
    with open(path, "wb") as f:
        f.write(blob)


def load_model(path: str) -> tuple[dict, dict]:
    with open(path, "rb") as f:
        t = decode_tensors(f.read())
    cfg = json.loads(bytes(t.pop("__config__").astype(np.uint8)).decode())
    return cfg, t


def shard_tensors(cfg: dict, weights: dict, start: int, end: int) -> dict:
    """Only the tensors a device needs for layers [start, end)."""
    keys = [k for i in range(start, end) for k in layer_keys(i)]
    if start == 0:
        keys += ["wte", "wpe"]
    if end == cfg["n_layer"]:
        keys += ["ln_f.w", "ln_f.b"] + ([] if start == 0 else ["wte"])  # tied LM head
    return {k: weights[k] for k in keys}


def shard_bytes(cfg: dict, start: int, end: int) -> int:
    D, V, C = cfg["n_embd"], cfg["vocab"], cfg["n_ctx"]
    per_layer = 4 * (12 * D * D + 13 * D)
    extra = (4 * (V * D + C * D) if start == 0 else 0) + (4 * V * D if end == cfg["n_layer"] and start else 0)
    return per_layer * (end - start) + extra


def _layernorm(x, w, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


def _gelu(x):
    return 0.5 * x * (1 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))


class Shard:
    """Runs layers [start, end). The first shard embeds tokens, the last returns logits.

    Keeps a per-job KV cache so each new token only ships one hidden vector
    between devices instead of the whole sequence.
    """

    MAX_JOBS = 8

    def __init__(self, cfg: dict, tensors: dict, start: int, end: int):
        self.cfg, self.w, self.start, self.end = cfg, tensors, start, end
        self.first, self.last = start == 0, end == cfg["n_layer"]
        self.caches: OrderedDict[str, dict] = OrderedDict()

    def forward(self, job: str, x: np.ndarray, pos: int) -> np.ndarray:
        cache = self.caches.pop(job, None)
        if cache is None or cache["len"] != pos:
            if pos != 0:
                raise ValueError(f"no KV cache for job {job} at position {pos}")
            cache = {"len": 0, "kv": {}}
        self.caches[job] = cache
        while len(self.caches) > self.MAX_JOBS:
            self.caches.popitem(last=False)

        w = self.w
        if self.first:
            ids = x.astype(np.int64)
            x = w["wte"][ids] + w["wpe"][pos : pos + len(ids)]
        x = x.astype(np.float32)
        for i in range(self.start, self.end):
            x = self._block(i, x, pos, cache["kv"])
        cache["len"] = pos + x.shape[0]
        if self.last:
            h = _layernorm(x[-1:], w["ln_f.w"], w["ln_f.b"])
            return (h @ w["wte"].T)[0].astype(np.float32)
        return x

    def drop(self, job: str) -> None:
        self.caches.pop(job, None)

    def _block(self, i: int, x: np.ndarray, pos: int, kv: dict) -> np.ndarray:
        w, p = self.w, f"h.{i}."
        H = self.cfg["n_head"]
        T, D = x.shape
        hd = D // H
        h = _layernorm(x, w[p + "ln_1.w"], w[p + "ln_1.b"])
        qkv = h @ w[p + "attn.c_attn.w"] + w[p + "attn.c_attn.b"]
        q, k, v = (a.reshape(T, H, hd).transpose(1, 0, 2) for a in np.split(qkv, 3, axis=-1))
        if i in kv:
            k = np.concatenate([kv[i][0], k], axis=1)
            v = np.concatenate([kv[i][1], v], axis=1)
        kv[i] = (k, v)
        S = k.shape[1]
        att = q @ k.transpose(0, 2, 1) / np.sqrt(hd)
        mask = np.arange(S)[None, :] > (pos + np.arange(T))[:, None]
        att = np.where(mask[None], -1e10, att)
        att = np.exp(att - att.max(-1, keepdims=True))
        att /= att.sum(-1, keepdims=True)
        y = (att @ v).transpose(1, 0, 2).reshape(T, D)
        x = x + y @ w[p + "attn.c_proj.w"] + w[p + "attn.c_proj.b"]
        h = _layernorm(x, w[p + "ln_2.w"], w[p + "ln_2.b"])
        x = x + _gelu(h @ w[p + "mlp.c_fc.w"] + w[p + "mlp.c_fc.b"]) @ w[p + "mlp.c_proj.w"] + w[p + "mlp.c_proj.b"]
        return x.astype(np.float32)


def plan_layers(n_layer: int, workers: list[dict], cfg: dict | None = None) -> list[tuple[str, int, int]]:
    """Split layers into contiguous ranges, proportional to each worker's speed score,
    without exceeding any worker's memory budget. Returns [(node_id, start, end)]."""
    ws = sorted(workers, key=lambda w: -w.get("score", 1.0))
    if not ws:
        raise RuntimeError("no workers available")
    ws = ws[:n_layer]
    total = sum(max(w.get("score", 1.0), 1e-3) for w in ws)
    counts = [max(1, round(n_layer * max(w.get("score", 1.0), 1e-3) / total)) for w in ws]
    while sum(counts) > n_layer:
        counts[counts.index(max(counts))] -= 1
    while sum(counts) < n_layer:
        counts[counts.index(min(counts))] += 1

    if cfg is not None:  # respect memory budgets by moving layers to workers with room
        per_layer = shard_bytes(cfg, 1, 2)
        embed_layers = -(-(shard_bytes(cfg, 0, 1) - per_layer) // per_layer)  # embeddings, in "layers"
        ws, counts = list(ws), list(counts)
        mems = [w.get("mem_mb", 1 << 20) * 1024 * 1024 for w in ws]
        if len(ws) > 1 and mems[0] // per_layer < embed_layers + 1:  # first device also holds the embeddings
            big = mems.index(max(mems))
            ws[0], ws[big] = ws[big], ws[0]
            mems[0], mems[big] = mems[big], mems[0]
        caps = [int(m // per_layer) for m in mems]
        caps[0] -= embed_layers
        if len(ws) > 1:
            caps[-1] -= embed_layers  # the last device holds a copy of the (tied) output embedding
        for _ in range(n_layer * len(ws)):
            over = [i for i, c in enumerate(counts) if c > max(caps[i], 0)]
            if not over:
                break
            room = [i for i, c in enumerate(counts) if c < caps[i]]
            if not room:
                raise RuntimeError("the connected devices do not have enough memory for this model")
            counts[over[0]] -= 1
            counts[room[0]] += 1
        if counts[0] == 0:
            raise RuntimeError("no connected device has enough memory to hold the model's embeddings")
        keep = [i for i, c in enumerate(counts) if c > 0]
        ws, counts = [ws[i] for i in keep], [counts[i] for i in keep]

    plan, start = [], 0
    for w, c in zip(ws, counts):
        plan.append((w["node_id"], start, start + c))
        start += c
    return plan
