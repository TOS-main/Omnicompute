"""GGUF model files (llama.cpp, LM Studio, Ollama exports), presented like a Hugging Face
checkpoint: Omnicompute's internal tensor names and [in, out] layout, plus the tokenizer and chat
template stored in the file.

Quantized tensors (Q4_0, Q4_K, Q5_K, Q6_K, Q8_0, ...) are expanded to float16 as each device's
layers are sent, using the `gguf` package's NumPy decoders, so the file stays small on disk and
in the download, while devices compute as they do for safetensors models.

Supported architectures: llama (Llama 2/3.x, Mistral, TinyLlama, SmolLM), qwen2, qwen3, phi3,
and gemma4 (dense Gemma 4: E2B, E4B, 31B; see model.py for its layout). Gemma 4's per-layer
embedding table is the biggest tensor in the file but is only ever looked up by row, so it
travels and stays in its quantized form ("ple_embed.q") instead of being expanded.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ARCHS = {"llama", "mistral", "qwen2", "qwen3", "phi3", "gemma4"}


def _field(reader, key, default=None):
    f = reader.fields.get(key)
    if f is None:
        return default
    v = f.contents()
    return v


def _unpermute(w: np.ndarray, n_head: int) -> np.ndarray:
    """Undo llama.cpp's reordering of Llama q/k rows (its RoPE pairs neighbours, HF's pairs halves)."""
    shape = w.shape
    return w.reshape(n_head, shape[0] // n_head // 2, 2, *shape[1:]).swapaxes(1, 2).reshape(shape)


def gguf_config(reader, name: str, max_ctx: int = 8192) -> dict:
    arch = _field(reader, "general.architecture")
    if arch not in ARCHS:
        raise ValueError(f"GGUF architecture '{arch}' is not supported yet (supported: {', '.join(sorted(ARCHS))})")
    a = lambda k, d=None: _field(reader, f"{arch}.{k}", d)  # noqa: E731
    D, H = int(a("embedding_length")), int(a("attention.head_count"))
    names = {t.name for t in reader.tensors}
    if arch == "gemma4":
        return _gemma4_config(reader, a, name, names, D, H, max_ctx)
    tokens = _field(reader, "tokenizer.ggml.tokens") or []
    rs = None
    t = next((t for t in reader.tensors if t.name == "rope_freqs.weight"), None)
    if t is not None and getattr(t, "data", None) is None:  # header only (estimates): factors unknown
        rs = {"rope_type": "freqs", "factors": []}
    elif t is not None:  # Llama 3.1+: per-frequency scaling stored as a tensor
        rs = {"rope_type": "freqs", "factors": [float(x) for x in np.asarray(t.data, np.float32).ravel()]}
    elif "rope_factors_long.weight" in names:
        raise ValueError("long-context (LongRoPE) GGUF models are not supported yet")
    elif a("rope.scaling.type") == "linear":
        rs = {"rope_type": "linear", "factor": float(a("rope.scaling.factor", 1.0))}
    ctx = int(a("context_length", 2048))
    hd = int(a("attention.key_length") or D // H)
    return {
        "arch": "llama", "hf_type": arch, "name": name, "tokenizer": "gguf",
        "n_layer": int(a("block_count")), "n_embd": D, "n_head": H,
        "n_kv_head": int(a("attention.head_count_kv") or H), "head_dim": hd,
        "ffn": int(a("feed_forward_length")), "vocab": int(a("vocab_size") or len(tokens)),
        "n_ctx": ctx, "ctx_default": min(ctx, max_ctx),
        "rope_theta": float(a("rope.freq_base", 10000.0)), "rope_scaling": rs,
        "rms_eps": float(a("attention.layer_norm_rms_epsilon", 1e-6)),
        "tie_embeddings": "output.weight" not in names,
        "qkv_bias": "blk.0.attn_q.bias" in names or "blk.0.attn_qkv.bias" in names,
        "qk_norm": "blk.0.attn_q_norm.weight" in names,
    }


def _gemma4_config(reader, a, name, names, D, H, max_ctx) -> dict:
    if any(".ffn_gate_inp." in n for n in names):
        raise ValueError("Gemma 4 mixture-of-experts models (26B-A4B) are not supported yet; E2B, E4B and 31B are")
    L = int(a("block_count"))
    per = lambda v: [int(x) for x in v] if isinstance(v, list) else [int(v)] * L  # noqa: E731
    swa = [bool(x) for x in (a("attention.sliding_window_pattern") or [True] * L)]
    t = next((t for t in reader.tensors if t.name == "rope_freqs.weight"), None)
    factors = [] if t is None or getattr(t, "data", None) is None else [float(x) for x in np.asarray(t.data, np.float32).ravel()]
    ctx = int(a("context_length", 131072))
    tokens = _field(reader, "tokenizer.ggml.tokens") or []
    shared = int(a("attention.shared_kv_layers", 0) or 0)
    return {
        "arch": "gemma4", "hf_type": "gemma4", "name": name, "tokenizer": "gguf",
        "n_layer": L, "n_embd": D, "n_head": H, "vocab": int(a("vocab_size") or len(tokens)),
        "head_dim": int(a("attention.key_length")), "head_dim_swa": int(a("attention.key_length_swa")),
        "n_kv_heads": per(a("attention.head_count_kv")), "ffns": per(a("feed_forward_length")),
        "swa": swa, "window": int(a("attention.sliding_window", 512)),
        "rope_theta": float(a("rope.freq_base", 1e6)), "rope_theta_swa": float(a("rope.freq_base_swa", 1e4)),
        "rope_factors": factors, "rms_eps": float(a("attention.layer_norm_rms_epsilon", 1e-6)),
        "softcap": float(a("final_logit_softcapping", 0.0) or 0.0),
        "ple_dim": int(a("embedding_length_per_layer_input", 0) or 0), "n_kv_shared": shared,
        "k_eq_v": [f"blk.{i}.attn_k.weight" in names and f"blk.{i}.attn_v.weight" not in names for i in range(L)],
        "tie_embeddings": "output.weight" not in names,
        "n_ctx": ctx, "ctx_default": min(ctx, max_ctx),
    }


GEMMA4_PARTS = {"attn_norm": "attn_norm", "q_norm": "attn_q_norm", "k_norm": "attn_k_norm",
                "post_attn_norm": "post_attention_norm", "mlp_norm": "ffn_norm", "post_mlp_norm": "post_ffw_norm",
                "ple_post_norm": "post_norm", "out_scale": "layer_output_scale",
                "q": "attn_q", "k": "attn_k", "v": "attn_v", "o": "attn_output", "gate": "ffn_gate", "up": "ffn_up",
                "down": "ffn_down", "ple_gate": "inp_gate", "ple_out": "proj"}


class GGUFSource:
    def __init__(self, path: str | Path, name: str | None = None, dtype=np.float16):
        from gguf import GGUFReader

        self.path = Path(path)
        self.folder = self.path.parent
        self.reader = GGUFReader(str(self.path))
        self.tensors = {t.name: t for t in self.reader.tensors}
        self.cfg = gguf_config(self.reader, name or self.path.stem)
        self.dtype = dtype

    def meta(self, key, default=None):
        return _field(self.reader, key, default)

    def _t(self, name: str) -> np.ndarray:
        from gguf import GGMLQuantizationType, quants

        t = self.tensors[name]
        if t.tensor_type in (GGMLQuantizationType.F32, GGMLQuantizationType.F16):
            w = np.asarray(t.data)
        else:
            w = quants.dequantize(np.asarray(t.data), t.tensor_type)
        return w.reshape([int(n) for n in reversed(t.shape)]) if w.ndim and w.size == np.prod(t.shape) else w

    def raw(self, name: str) -> tuple[np.ndarray, int]:
        """A tensor's bytes as stored (one row per output), and its GGML type."""
        t = self.tensors[name]
        return np.ascontiguousarray(np.asarray(t.data).reshape(int(t.shape[-1]), -1).view(np.uint8)), int(t.tensor_type)

    def get(self, name: str) -> np.ndarray:
        if name in ("ple_embed.q", "ple_embed.type"):  # Gemma 4: kept quantized (see the module notes)
            q, kind = self.raw("per_layer_token_embd.weight")
            return q if name == "ple_embed.q" else np.array([kind], np.int32)
        w = np.asarray(self._get(name), np.float32)
        if w.ndim >= 2 and self.dtype == np.float16:
            return np.ascontiguousarray(np.clip(w, -65504, 65504).astype(np.float16))
        return np.ascontiguousarray(w)

    def _gemma4(self, name: str) -> np.ndarray:
        glob = {"embed": "token_embd.weight", "norm_f": "output_norm.weight", "ple_norm": "per_layer_proj_norm.weight"}
        if name in glob:
            return self._t(glob[name])
        if name == "lm_head":
            return self._t("output.weight").T
        if name == "ple_proj.w":
            return self._t("per_layer_model_proj.weight").T
        _, i, part = name.split(".", 2)
        w = self._t(f"blk.{i}.{GEMMA4_PARTS[part.removesuffix('.w')]}.weight")
        return w.T if part.endswith(".w") else w.reshape(-1)

    def _get(self, name: str) -> np.ndarray:
        c = self.cfg
        if c["arch"] == "gemma4":
            return self._gemma4(name)
        if name == "embed":
            return self._t("token_embd.weight")
        if name == "norm_f":
            return self._t("output_norm.weight")
        if name == "lm_head":
            return self._t("output.weight").T
        _, i, part = name.split(".", 2)
        b = f"blk.{i}."
        simple = {"attn_norm": "attn_norm.weight", "mlp_norm": "ffn_norm.weight",
                  "q_norm": "attn_q_norm.weight", "k_norm": "attn_k_norm.weight"}
        if part in simple:
            return self._t(b + simple[part])
        proj, kind = part.split(".")
        suffix = ".weight" if kind == "w" else ".bias"
        H, KV, hd, F = c["n_head"], c["n_kv_head"], c["head_dim"], c["ffn"]
        if proj in ("q", "k", "v") and b + "attn_qkv.weight" in self.tensors:  # Phi-3: fused q/k/v
            full = self._t(b + "attn_qkv" + suffix)
            lo, hi = {"q": (0, H * hd), "k": (H * hd, (H + KV) * hd), "v": ((H + KV) * hd, (H + 2 * KV) * hd)}[proj]
            w = full[lo:hi]
        elif proj in ("gate", "up") and b + "ffn_gate.weight" not in self.tensors:  # Phi-3: gate and up fused
            full = self._t(b + "ffn_up.weight")
            w = full[:F] if proj == "gate" else full[F:]
        else:
            w = self._t(b + {"q": "attn_q", "k": "attn_k", "v": "attn_v", "o": "attn_output", "gate": "ffn_gate",
                             "up": "ffn_up", "down": "ffn_down"}[proj] + suffix)
        if c["hf_type"] == "llama" and proj in ("q", "k"):
            w = _unpermute(w, H if proj == "q" else KV)
        return w.T if kind == "w" else w  # GGUF rows are outputs, like HF's [out, in]

    def make_tokenizer(self):
        from .tokenizer import GGUFTokenizer

        return GGUFTokenizer(self)


# ---------------------------------------------------------------- header only (remote files)

class _Field:
    def __init__(self, v):
        self.v = v

    def contents(self):
        return self.v


class _TensorInfo:
    data = None

    def __init__(self, name, shape):
        self.name, self.shape = name, shape


class _Header:
    """GGUF metadata and tensor names parsed from the start of a file, without its weights."""

    def __init__(self, read):
        import struct

        self.read, self.buf, self.pos = read, b"", 0
        self.u = lambda fmt: struct.unpack_from("<" + fmt, self._need(struct.calcsize("<" + fmt)), 0)[0]
        if self._need(4) != b"GGUF":
            raise ValueError("not a GGUF file")
        self.pos = 4
        self.u("I")  # version
        n_tensors, n_kv = self.u("Q"), self.u("Q")
        self.fields = {}
        for _ in range(n_kv):
            k = self._str()
            self.fields[k] = _Field(self._value(self.u("I")))
        self.tensors = []
        for _ in range(n_tensors):
            name = self._str()
            dims = [self.u("Q") for _ in range(self.u("I"))]
            self.u("I"), self.u("Q")  # type, offset
            self.tensors.append(_TensorInfo(name, dims))

    def _need(self, n):
        while len(self.buf) < self.pos + n:
            more = self.read(len(self.buf), len(self.buf) + max(4 << 20, len(self.buf)))
            if not more:
                raise ValueError("GGUF header is cut short")
            self.buf += more
        out = self.buf[self.pos:self.pos + n]
        self.pos += n
        return out

    def _str(self):
        return self._need(self.u("Q")).decode("utf-8", errors="replace")

    def _value(self, t):
        fmts = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
        if t in fmts:
            return self.u(fmts[t])
        if t == 8:
            return self._str()
        if t == 9:
            et, n = self.u("I"), self.u("Q")
            return [self._value(et) for _ in range(n)]
        raise ValueError(f"unknown GGUF value type {t}")


def remote_gguf_config(read, name: str) -> dict:
    """Omnicompute config of a GGUF file from its first few megabytes; read(start, end) -> bytes."""
    return gguf_config(_Header(read), name)


def local_gguf_config(path: str) -> tuple[dict, int]:
    p = Path(path).expanduser()
    with open(p, "rb") as f:
        def read(a, b):
            f.seek(a)
            return f.read(b - a)

        return remote_gguf_config(read, f"gguf:{p}"), p.stat().st_size


def find_gguf(roots: list[Path]) -> list[Path]:
    out = []
    for r in roots:
        if r.is_dir():
            out += [p for p in r.rglob("*.gguf") if p.is_file() and "mmproj" not in p.name.lower()]
    return sorted(set(out))


def lmstudio_dirs() -> list[Path]:
    """Where LM Studio (and the llama.cpp cache) keep downloaded GGUF files."""
    home = Path.home()
    return [home / ".lmstudio" / "models", home / ".cache" / "lm-studio" / "models", home / ".cache" / "llama.cpp"]
