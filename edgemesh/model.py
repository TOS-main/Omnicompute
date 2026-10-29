"""Decoder-only transformers in plain numpy, split into layer shards.

Two architecture families cover most open models on Hugging Face:
  * "gpt2":  GPT-2, DistilGPT-2 (LayerNorm, learned positions, GELU)
  * "llama": Llama 2/3.x, Mistral, Qwen2/2.5/3, SmolLM, TinyLlama, Phi-3 and
             their fine-tunes (RMSNorm, RoPE, grouped-query attention, SwiGLU)

Weights use internal names (see `tensor_shapes`) in [in, out] layout, so every
projection is `x @ W`. They may be float16 to halve memory and bandwidth;
compute is float32. The browser worker (web/model.js) implements the same math.
"""

from __future__ import annotations

import json
from collections import OrderedDict

import numpy as np

from .protocol import decode_tensors, encode_tensors

# ---------------------------------------------------------------- weights


def tensor_shapes(cfg: dict) -> tuple[dict, dict]:
    """(global tensors, per-layer tensors with "{i}" in the name) -> shapes."""
    D, V = cfg["n_embd"], cfg["vocab"]
    if cfg.get("arch", "gpt2") == "gpt2":
        g = {"wte": (V, D), "wpe": (cfg["n_ctx"], D), "ln_f.w": (D,), "ln_f.b": (D,)}
        per = {"ln_1.w": (D,), "ln_1.b": (D,), "attn.c_attn.w": (D, 3 * D), "attn.c_attn.b": (3 * D,),
               "attn.c_proj.w": (D, D), "attn.c_proj.b": (D,), "ln_2.w": (D,), "ln_2.b": (D,),
               "mlp.c_fc.w": (D, 4 * D), "mlp.c_fc.b": (4 * D,), "mlp.c_proj.w": (4 * D, D), "mlp.c_proj.b": (D,)}
    else:
        H, KV, hd, F = cfg["n_head"], cfg["n_kv_head"], cfg["head_dim"], cfg["ffn"]
        g = {"embed": (V, D), "norm_f": (D,)}
        if not cfg.get("tie_embeddings"):
            g["lm_head"] = (D, V)
        per = {"attn_norm": (D,), "q.w": (D, H * hd), "k.w": (D, KV * hd), "v.w": (D, KV * hd),
               "o.w": (H * hd, D), "mlp_norm": (D,), "gate.w": (D, F), "up.w": (D, F), "down.w": (F, D)}
        if cfg.get("qkv_bias"):
            per.update({"q.b": (H * hd,), "k.b": (KV * hd,), "v.b": (KV * hd,)})
        if cfg.get("qk_norm"):
            per.update({"q_norm": (hd,), "k_norm": (hd,)})
    return g, {"h.{i}." + k: v for k, v in per.items()}


def layer_keys(cfg: dict, i: int) -> list[str]:
    return [k.format(i=i) for k in tensor_shapes(cfg)[1]]


def shard_keys(cfg: dict, start: int, end: int) -> list[str]:
    """The tensors a device needs for layers [start, end)."""
    gpt2 = cfg.get("arch", "gpt2") == "gpt2"
    keys = [k for i in range(start, end) for k in layer_keys(cfg, i)]
    if start == 0:
        keys += ["wte", "wpe"] if gpt2 else ["embed"]
    if end == cfg["n_layer"]:
        if gpt2:
            keys += ["ln_f.w", "ln_f.b", "wte"]
        else:
            keys += ["norm_f", "embed" if cfg.get("tie_embeddings") else "lm_head"]
    return list(dict.fromkeys(keys))


def shard_bytes(cfg: dict, start: int, end: int, bytes_per_param: int = 4) -> int:
    g, per = tensor_shapes(cfg)
    shapes = {**g, **{k.format(i=i): v for i in range(cfg["n_layer"]) for k, v in per.items()}}
    return sum(int(np.prod(shapes[k])) for k in shard_keys(cfg, start, end)) * bytes_per_param


class DictSource:
    """Model weights held in memory (random demo models, .emm files)."""

    def __init__(self, cfg: dict, weights: dict):
        self.cfg, self.weights = cfg, weights

    def get(self, name: str) -> np.ndarray:
        return self.weights[name]


def shard_tensors(source, start: int, end: int) -> dict:
    return {k: source.get(k) for k in shard_keys(source.cfg, start, end)}


def random_model(n_layer=8, n_embd=128, n_head=4, n_ctx=256, vocab=256, seed=0, arch="gpt2",
                 n_kv_head=None, qkv_bias=False, qk_norm=False, tie=True) -> tuple[dict, dict]:
    """A randomly initialised model; useful for testing the mesh without downloads."""
    rng = np.random.default_rng(seed)
    cfg = {"arch": arch, "n_layer": n_layer, "n_embd": n_embd, "n_head": n_head, "n_ctx": n_ctx,
           "vocab": vocab, "tokenizer": "bytes", "name": f"random-{arch}-{n_layer}x{n_embd}"}
    if arch == "llama":
        cfg.update({"n_kv_head": n_kv_head or n_head, "head_dim": n_embd // n_head, "ffn": 3 * n_embd,
                    "rope_theta": 10000.0, "rms_eps": 1e-6, "tie_embeddings": tie,
                    "qkv_bias": qkv_bias, "qk_norm": qk_norm})
    g, per = tensor_shapes(cfg)
    norms = ("norm_f", "attn_norm", "mlp_norm", "q_norm", "k_norm", "ln_1.w", "ln_2.w", "ln_f.w")
    w = {}
    for name, shape in [*g.items(), *[(k.format(i=i), s) for i in range(n_layer) for k, s in per.items()]]:
        if name.endswith(norms):
            w[name] = 1 + rng.normal(0, 0.05, shape)
        elif len(shape) == 1:
            w[name] = rng.normal(0, 0.02, shape)
        else:
            w[name] = rng.normal(0, 0.1 if name in ("wte", "embed") else 0.05, shape)
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


# ---------------------------------------------------------------- math


def _f32(w: np.ndarray) -> np.ndarray:
    return w if w.dtype == np.float32 else w.astype(np.float32)


def _layernorm(x, w, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * _f32(w) + _f32(b)


def _rmsnorm(x, w, eps):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * _f32(w)


def _gelu(x):
    return 0.5 * x * (1 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))


def rope_inv_freq(cfg: dict) -> np.ndarray:
    hd = cfg["head_dim"]
    inv = 1.0 / (cfg.get("rope_theta", 10000.0) ** (np.arange(0, hd, 2, dtype=np.float64) / hd))
    rs = cfg.get("rope_scaling") or {}
    kind = rs.get("rope_type", rs.get("type"))
    if kind == "linear":
        inv = inv / rs["factor"]
    elif kind == "freqs":  # per-frequency factors given directly (GGUF's rope_freqs tensor)
        inv = inv / np.asarray(rs["factors"], np.float64)[: len(inv)]
    elif kind == "llama3":  # Llama 3.1+ long-context frequency scaling
        factor, lo, hi = rs["factor"], rs.get("low_freq_factor", 1.0), rs.get("high_freq_factor", 4.0)
        old = rs.get("original_max_position_embeddings", 8192)
        wavelen = 2 * np.pi / inv
        smooth = (old / wavelen - lo) / (hi - lo)
        scaled = np.where(wavelen > old / lo, inv / factor, inv)
        mid = (wavelen <= old / lo) & (wavelen >= old / hi)
        inv = np.where(mid, (1 - smooth) * scaled / factor + smooth * scaled, scaled)
    return inv


def _rope(x: np.ndarray, pos: int, inv: np.ndarray) -> np.ndarray:
    """x: [heads, T, hd], HF "rotate_half" convention."""
    T = x.shape[1]
    ang = np.arange(pos, pos + T, dtype=np.float64)[:, None] * inv[None, :]
    cos, sin = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
    cos, sin = np.concatenate([cos, cos], -1), np.concatenate([sin, sin], -1)
    half = x.shape[-1] // 2
    rot = np.concatenate([-x[..., half:], x[..., :half]], -1)
    return x * cos + rot * sin


def _attend(q, k, v, pos):
    """q: [H, T, hd], k/v: [KV, S, hd] -> [T, H*hd], causal, grouped-query."""
    H, T, hd = q.shape
    groups = H // k.shape[0]
    if groups > 1:
        k, v = np.repeat(k, groups, axis=0), np.repeat(v, groups, axis=0)
    S = k.shape[1]
    att = q @ k.transpose(0, 2, 1) / np.sqrt(hd)
    mask = np.arange(S)[None, :] > (pos + np.arange(T))[:, None]
    att = np.where(mask[None], -1e30, att)
    att = np.exp(att - att.max(-1, keepdims=True))
    att /= att.sum(-1, keepdims=True)
    return (att @ v).transpose(1, 0, 2).reshape(T, H * hd)


class Shard:
    """Runs layers [start, end). The first shard embeds tokens, the last returns logits.

    Keeps a per-job KV cache so each new token only ships one hidden vector
    between devices instead of the whole sequence.
    """

    MAX_JOBS = 8

    def __init__(self, cfg: dict, tensors: dict, start: int, end: int):
        self.cfg, self.w, self.start, self.end = cfg, tensors, start, end
        self.first, self.last = start == 0, end == cfg["n_layer"]
        self.llama = cfg.get("arch", "gpt2") == "llama"
        self.inv_freq = rope_inv_freq(cfg) if self.llama else None
        self.caches: OrderedDict[str, dict] = OrderedDict()
        self.gpu = None  # optional gpu.GPU holding the weight matrices (see use_gpu)
        self.cpu_part: dict[str, np.ndarray] = {}  # columns the CPU computes alongside the GPU
        self.cpu_share, self._pool = 0.0, None

    def use_gpu(self, gpu, cpu_share: float = 0.0) -> None:
        """Move every weight matrix (and a tied output head) to the GPU, freeing the CPU copies.
        cpu_share > 0 keeps that share of each matrix's columns on the CPU, which computes them
        while the GPU does the rest (both at once, so a step takes the slower of the two)."""
        names = [n for n, t in self.w.items() if t.ndim == 2 and (n.endswith(".w") or n == "lm_head")]
        tied = None if "lm_head" in self.w or not self.last else "embed" if self.llama else "wte"
        mats = [(n, lambda n=n: self.w.pop(n)) for n in names] + ([(tied + "^T", lambda: self.w[tied].T)] if tied else [])
        self.cpu_part = {}
        for name, take in mats:
            w = take()
            n_cpu = int(w.shape[1] * cpu_share) // 32 * 32
            if n_cpu:
                self.cpu_part[name] = np.ascontiguousarray(_f32(w[:, :n_cpu]))
                w = w[:, n_cpu:]
            gpu.put(name, np.ascontiguousarray(w))
        self.cpu_share = cpu_share if self.cpu_part else 0.0
        if self.cpu_part and self._pool is None:
            from concurrent.futures import ThreadPoolExecutor

            self._pool = ThreadPoolExecutor(1, thread_name_prefix="edgemesh-cpu-part")
        self.gpu = gpu

    def _gpu_mm(self, x: np.ndarray, names: list[str]) -> list[np.ndarray]:
        """x @ W on the GPU, with the CPU computing its columns at the same time (see use_gpu)."""
        part = [n for n in names if n in self.cpu_part]
        if not part:
            return self.gpu.mm_many(x, names)
        fut = self._pool.submit(lambda: [x @ self.cpu_part[n] for n in part])
        ys = self.gpu.mm_many(x, names)
        left = dict(zip(part, fut.result()))
        return [np.concatenate([left[n], y], axis=1) if n in left else y for n, y in zip(names, ys)]

    def _mms(self, x: np.ndarray, names: list[str]) -> list[np.ndarray]:
        """[x @ W + b for each weight name], on the GPU when the weights are there."""
        w = self.w
        if self.gpu is not None and all(self.gpu.has(n) for n in names):
            ys = self._gpu_mm(x, names)
        else:
            ys = [x @ _f32(w[n]) for n in names]
        out = []
        for n, y in zip(names, ys):
            b = w.get(n[:-2] + ".b") if n.endswith(".w") else None
            out.append(y + _f32(b) if b is not None else y)
        return out

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
            if self.llama:
                x = _f32(w["embed"][ids])
            else:
                x = _f32(w["wte"][ids]) + _f32(w["wpe"][pos : pos + len(ids)])
        x = x.astype(np.float32)
        block = self._llama_block if self.llama else self._gpt2_block
        for i in range(self.start, self.end):
            x = block(i, x, pos, cache["kv"])
        cache["len"] = pos + x.shape[0]
        if self.last:
            if self.llama:
                h = _rmsnorm(x[-1:], w["norm_f"], self.cfg["rms_eps"])
                tied = "embed"
            else:
                h = _layernorm(x[-1:], w["ln_f.w"], w["ln_f.b"])
                tied = "wte"
            name = "lm_head" if self.gpu is not None and self.gpu.has("lm_head") else tied + "^T"
            if self.gpu is not None and self.gpu.has(name):
                return self._gpu_mm(h, [name])[0][0].astype(np.float32)
            head = w["lm_head"] if "lm_head" in w else w[tied].T
            return (h @ _f32(head))[0].astype(np.float32)
        return x

    def drop(self, job: str) -> None:
        self.caches.pop(job, None)

    @staticmethod
    def _cache(kv: dict, i: int, k: np.ndarray, v: np.ndarray):
        if i in kv:
            k = np.concatenate([kv[i][0], k], axis=1)
            v = np.concatenate([kv[i][1], v], axis=1)
        kv[i] = (k, v)
        return k, v

    def _gpt2_block(self, i: int, x: np.ndarray, pos: int, kv: dict) -> np.ndarray:
        w, p = self.w, f"h.{i}."
        H = self.cfg["n_head"]
        T, D = x.shape
        h = _layernorm(x, w[p + "ln_1.w"], w[p + "ln_1.b"])
        qkv, = self._mms(h, [p + "attn.c_attn.w"])
        q, k, v = (a.reshape(T, H, D // H).transpose(1, 0, 2) for a in np.split(qkv, 3, axis=-1))
        k, v = self._cache(kv, i, k, v)
        x = x + self._mms(_attend(q, k, v, pos), [p + "attn.c_proj.w"])[0]
        h = _layernorm(x, w[p + "ln_2.w"], w[p + "ln_2.b"])
        x = x + self._mms(_gelu(self._mms(h, [p + "mlp.c_fc.w"])[0]), [p + "mlp.c_proj.w"])[0]
        return x.astype(np.float32)

    def _llama_block(self, i: int, x: np.ndarray, pos: int, kv: dict) -> np.ndarray:
        w, p, c = self.w, f"h.{i}.", self.cfg
        H, KV, hd, eps = c["n_head"], c["n_kv_head"], c["head_dim"], c["rms_eps"]
        T = x.shape[0]
        h = _rmsnorm(x, w[p + "attn_norm"], eps)
        q, k, v = self._mms(h, [p + "q.w", p + "k.w", p + "v.w"])
        q = q.reshape(T, H, hd).transpose(1, 0, 2)
        k = k.reshape(T, KV, hd).transpose(1, 0, 2)
        v = v.reshape(T, KV, hd).transpose(1, 0, 2)
        if p + "q_norm" in w:  # Qwen3: per-head RMSNorm on q and k
            q, k = _rmsnorm(q, w[p + "q_norm"], eps), _rmsnorm(k, w[p + "k_norm"], eps)
        q, k = _rope(q, pos, self.inv_freq), _rope(k, pos, self.inv_freq)
        k, v = self._cache(kv, i, k.astype(np.float32), v)
        x = x + self._mms(_attend(q, k, v, pos), [p + "o.w"])[0]
        h = _rmsnorm(x, w[p + "mlp_norm"], eps)
        g, u = self._mms(h, [p + "gate.w", p + "up.w"])
        x = x + self._mms(g / (1 + np.exp(-g)) * u, [p + "down.w"])[0]
        return x.astype(np.float32)


# ---------------------------------------------------------------- planning


def plan_layers(n_layer: int, workers: list[dict], cfg: dict | None = None,
                bytes_per_param: int = 4) -> list[tuple[str, int, int]]:
    """Split layers into contiguous ranges, in proportion to each worker's speed score,
    without exceeding any worker's memory budget. Returns [(node_id, start, end)].

    The last device also computes the output head (a vocab-sized matrix, often as big as
    several layers), so the fastest device goes last and gets correspondingly fewer layers.
    A worker may report "weight_factor" to scale how much memory its weights take."""
    if not workers:
        raise RuntimeError("no workers available")
    ws = sorted(workers, key=lambda w: -w.get("score", 1.0))[:n_layer]
    ws = ws[1:] + ws[:1]  # fastest last
    scores = [max(w.get("score", 1.0), 1e-3) for w in ws]
    head_units = 0.0
    if cfg is not None and len(ws) > 1:
        per_layer = max(shard_bytes(cfg, 1, 2) if n_layer > 1 else shard_bytes(cfg, 0, 1), 1)
        head_units = shard_bytes(cfg, n_layer, n_layer) / per_layer  # head compute, in layers
    units = n_layer + head_units
    want = [units * sc / sum(scores) for sc in scores]
    want[-1] -= head_units
    counts = [max(1, round(x)) for x in want]
    while sum(counts) > n_layer:
        i = max(range(len(counts)), key=lambda j: counts[j] - want[j])
        if counts[i] <= 1:
            i = counts.index(max(counts))
        counts[i] -= 1
    while sum(counts) < n_layer:
        counts[min(range(len(counts)), key=lambda j: counts[j] - want[j])] += 1

    if cfg is not None:  # respect memory budgets by moving layers to workers with room
        per_layer = shard_bytes(cfg, 1, 2, bytes_per_param) if n_layer > 2 else \
            shard_bytes(cfg, 0, 1, bytes_per_param) - shard_bytes(cfg, 0, 0, bytes_per_param)
        first_extra = shard_bytes(cfg, 0, 0, bytes_per_param)
        last_extra = shard_bytes(cfg, n_layer, n_layer, bytes_per_param)
        embed_layers = -(-first_extra // per_layer)
        head_layers = -(-last_extra // per_layer)
        mems = [w.get("mem_mb", 1 << 20) * 1024 * 1024 / w.get("weight_factor", 1) for w in ws]
        if len(ws) > 1 and mems[0] // per_layer < embed_layers + 1:  # first device also holds the embeddings
            big = max(range(len(ws) - 1), key=lambda i: mems[i])
            ws[0], ws[big] = ws[big], ws[0]
            mems[0], mems[big] = mems[big], mems[0]
            counts[0], counts[big] = counts[big], counts[0]
        if len(ws) > 1 and mems[-1] // per_layer < head_layers + 1:  # last device also holds the head
            big = max(range(1, len(ws)), key=lambda i: mems[i])
            ws[-1], ws[big] = ws[big], ws[-1]
            mems[-1], mems[big] = mems[big], mems[-1]
            counts[-1], counts[big] = counts[big], counts[-1]
        caps = [int(m // per_layer) for m in mems]
        caps[0] -= embed_layers
        if len(ws) > 1:
            caps[-1] -= head_layers
        for _ in range(n_layer * len(ws)):
            over = [i for i, c in enumerate(counts) if c > max(caps[i], 0)]
            if not over:
                break
            room = [i for i, c in enumerate(counts) if c < caps[i]]
            if not room:
                need = shard_bytes(cfg, 0, n_layer, bytes_per_param) / 2 ** 30
                raise RuntimeError(f"the connected devices do not have enough memory for this model "
                                   f"(needs about {need:.1f} GB in total)")
            counts[over[0]] -= 1
            counts[room[0]] += 1
        if 0 in counts:  # a device ended up with nothing (too little memory): plan again without it
            keep = [w for w, c in zip(ws, counts) if c > 0]
            if not keep:
                raise RuntimeError("no connected device has enough memory to hold the model's embeddings")
            return plan_layers(n_layer, keep, cfg, bytes_per_param)

    plan, start = [], 0
    for w, c in zip(ws, counts):
        plan.append((w["node_id"], start, start + c))
        start += c
    return plan
