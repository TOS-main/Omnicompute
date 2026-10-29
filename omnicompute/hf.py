"""Load models straight from Hugging Face (or a local folder in the same layout).

`omnicompute host --model hf:Qwen/Qwen2.5-0.5B-Instruct`, or paste a repo id on the
hub page. Only the hub downloads; it reads tensors lazily from the
.safetensors files (memory-mapped) and sends each device just its layers, as
float16. No torch or transformers needed.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import struct
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

log = logging.getLogger("omnicompute.hf")

HF = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
LLAMA_LIKE = {"llama", "mistral", "qwen2", "qwen3", "phi3"}  # SmolLM/SmolLM2, TinyLlama use "llama"
SUPPORTED = sorted(LLAMA_LIKE | {"gpt2"})
META_FILES = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
              "special_tokens_map.json"]


# ---------------------------------------------------------------- config


def omnicompute_config(hf: dict, name: str, max_ctx: int = 8192) -> dict:
    mt = hf.get("model_type")
    if mt == "gpt2":
        return {"arch": "gpt2", "n_layer": hf["n_layer"], "n_embd": hf["n_embd"], "n_head": hf["n_head"],
                "n_ctx": hf.get("n_positions", hf.get("n_ctx", 1024)), "vocab": hf["vocab_size"],
                "tokenizer": "hf", "name": name, "hf_type": mt}
    if mt not in LLAMA_LIKE:
        raise ValueError(f"model type '{mt}' is not supported yet (supported: {', '.join(SUPPORTED)})")
    rp = hf.get("rope_parameters") or {}  # newer transformers versions nest RoPE settings here
    rs = hf.get("rope_scaling") or (rp if rp.get("rope_type") not in (None, "default") else None)
    if rs and rs.get("rope_type", rs.get("type")) not in (None, "default", "linear", "llama3"):
        raise ValueError(f"rope scaling '{rs.get('rope_type', rs.get('type'))}' is not supported yet")
    H = hf["num_attention_heads"]
    return {
        "arch": "llama", "hf_type": mt, "name": name, "tokenizer": "hf",
        "n_layer": hf["num_hidden_layers"], "n_embd": hf["hidden_size"], "n_head": H,
        "n_kv_head": hf.get("num_key_value_heads") or H,
        "head_dim": hf.get("head_dim") or hf["hidden_size"] // H,
        "ffn": hf["intermediate_size"], "vocab": hf["vocab_size"],
        "n_ctx": hf.get("max_position_embeddings", 2048),  # the model's own limit
        "ctx_default": min(hf.get("max_position_embeddings", 2048), max_ctx),
        "rope_theta": hf.get("rope_theta", rp.get("rope_theta", 10000.0)), "rope_scaling": rs,
        "rms_eps": hf.get("rms_norm_eps", 1e-6),
        "tie_embeddings": bool(hf.get("tie_word_embeddings", False)),
        "qkv_bias": mt == "qwen2" or bool(hf.get("attention_bias", False)),
        "qk_norm": mt == "qwen3",
    }


# ---------------------------------------------------------------- safetensors


class _SafeTensors:
    """Index over one or more .safetensors files; tensors are read on demand via memmap."""

    def __init__(self, folder: Path):
        files = sorted(folder.glob("*.safetensors"))
        if not files:
            raise FileNotFoundError(f"no .safetensors weights in {folder}")
        self.index: dict[str, tuple] = {}
        for f in files:
            with open(f, "rb") as fh:
                (n,) = struct.unpack("<Q", fh.read(8))
                header = json.loads(fh.read(n))
            for name, t in header.items():
                if name != "__metadata__":
                    self.index[name] = (f, 8 + n + t["data_offsets"][0], t["dtype"], tuple(t["shape"]))

    def __contains__(self, name: str) -> bool:
        return name in self.index

    def get(self, name: str) -> np.ndarray:
        f, off, dtype, shape = self.index[name]
        if dtype == "BF16":
            raw = np.memmap(f, "<u2", "r", off, shape)
            return (raw.astype(np.uint32) << 16).view(np.float32)
        np_dtype = {"F32": "<f4", "F16": "<f2", "F64": "<f8"}.get(dtype)
        if np_dtype is None:
            raise ValueError(f"tensor {name} has unsupported dtype {dtype}")
        return np.asarray(np.memmap(f, np_dtype, "r", off, shape))


class HFSource:
    """A Hugging Face checkpoint folder, presented in Omnicompute's internal tensor layout."""

    def __init__(self, folder: str | Path, name: str | None = None, dtype=np.float16):
        self.folder = Path(folder)
        hf = json.loads((self.folder / "config.json").read_text(encoding="utf-8"))
        self.hf_config = hf
        self.cfg = omnicompute_config(hf, name or self.folder.name)
        gen = self.folder / "generation_config.json"
        if gen.exists():  # the model author's suggested sampling settings, used as the web page's defaults
            g = json.loads(gen.read_text(encoding="utf-8"))
            self.cfg["gen_defaults"] = {k: g[k] for k in ("max_new_tokens", "temperature", "top_p") if k in g}
        self.st = _SafeTensors(self.folder)
        self.dtype = dtype
        self.prefix = next((p for p in ("", "model.", "transformer.") if self._has(p)), "")

    def _has(self, prefix: str) -> bool:
        probe = "wte.weight" if self.cfg["arch"] == "gpt2" else "embed_tokens.weight"
        return prefix + probe in self.st

    def _t(self, name: str) -> np.ndarray:
        return self.st.get(self.prefix + name)

    def get(self, name: str) -> np.ndarray:
        w = self._gpt2(name) if self.cfg["arch"] == "gpt2" else self._llama(name)
        w = np.asarray(w, np.float32)
        if w.ndim >= 2 and self.dtype == np.float16:  # big matrices travel and live as float16
            return np.ascontiguousarray(np.clip(w, -65504, 65504).astype(np.float16))
        return np.ascontiguousarray(w)

    def _gpt2(self, name: str) -> np.ndarray:
        if name in ("wte", "wpe"):
            return self._t(name + ".weight")
        if name.startswith("ln_f."):
            return self._t("ln_f." + ("weight" if name.endswith(".w") else "bias"))
        base, kind = name.rsplit(".", 1)  # GPT-2's Conv1D weights are already [in, out]
        return self._t(base + (".weight" if kind == "w" else ".bias"))

    def _llama(self, name: str) -> np.ndarray:
        c = self.cfg
        if name == "embed":
            return self._t("embed_tokens.weight")
        if name == "norm_f":
            return self._t("norm.weight")
        if name == "lm_head":
            return self.st.get("lm_head.weight").T
        _, i, part = name.split(".", 2)
        L = f"layers.{i}."
        simple = {"attn_norm": "input_layernorm.weight", "mlp_norm": "post_attention_layernorm.weight",
                  "q_norm": "self_attn.q_norm.weight", "k_norm": "self_attn.k_norm.weight"}
        if part in simple:
            return self._t(L + simple[part])
        proj, kind = part.split(".")
        suffix = ".weight" if kind == "w" else ".bias"
        H, KV, hd, F = c["n_head"], c["n_kv_head"], c["head_dim"], c["ffn"]
        if proj in ("q", "k", "v") and (L + "self_attn.qkv_proj.weight") in self._prefixed():  # Phi-3 fused qkv
            full = self._t(L + "self_attn.qkv_proj" + suffix)
            a, b = {"q": (0, H * hd), "k": (H * hd, (H + KV) * hd), "v": ((H + KV) * hd, (H + 2 * KV) * hd)}[proj]
            part_w = full[a:b]
        elif proj in ("gate", "up") and (L + "mlp.gate_up_proj.weight") in self._prefixed():  # Phi-3 fused MLP
            full = self._t(L + "mlp.gate_up_proj.weight")
            part_w = full[:F] if proj == "gate" else full[F:]
        else:
            module = {"q": "self_attn.q_proj", "k": "self_attn.k_proj", "v": "self_attn.v_proj",
                      "o": "self_attn.o_proj", "gate": "mlp.gate_proj", "up": "mlp.up_proj",
                      "down": "mlp.down_proj"}[proj]
            part_w = self._t(L + module + suffix)
        return part_w.T if kind == "w" else part_w  # HF Linear is [out, in]

    def _prefixed(self):
        return _PrefixView(self.st, self.prefix)


class _PrefixView:
    def __init__(self, st, prefix):
        self.st, self.prefix = st, prefix

    def __contains__(self, name):
        return self.prefix + name in self.st


# ---------------------------------------------------------------- download


class Download:
    """Progress of one model download, shared with the web page."""

    def __init__(self, repo: str):
        self.repo = repo
        self.state = "starting"  # starting | downloading | loading | ready | error
        self.total = self.done = 0
        self.file = ""
        self.error = ""
        self.folder: Path | None = None

    def as_dict(self) -> dict:
        return {"repo": self.repo, "state": self.state, "total": self.total, "done": self.done,
                "file": self.file, "error": self.error}


def _request(url: str, token: str | None, headers: dict | None = None):
    h = {"User-Agent": "omnicompute/0.1", **(headers or {})}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=60)


def list_repo_files(repo: str, token: str | None = None, revision: str = "main") -> dict[str, int]:
    """{path: size} for files at the top of the repo."""
    try:
        with _request(f"{HF}/api/models/{repo}/tree/{revision}", token) as r:
            items = json.load(r)
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            raise PermissionError(f"{repo} is gated or private: accept its license on huggingface.co "
                                  "and give the hub a Hugging Face token") from e
        if e.code == 404:
            raise FileNotFoundError(f"no model called {repo} on Hugging Face") from e
        raise
    return {i["path"]: (i.get("lfs") or {}).get("size", i.get("size", 0)) for i in items if i.get("type") == "file"}


GGUF_PREFERENCE = ["Q4_K_M", "Q4_K_S", "Q5_K_M", "Q8_0", "Q6_K", "Q5_K_S", "Q4_0", "IQ4_XS", "Q3_K_L", "F16"]


def split_spec(repo: str) -> tuple[str, str | None]:
    """'org/name:Q8_0' -> ('org/name', 'Q8_0'): which file of a GGUF repo to use."""
    name, _, want = repo.partition(":")
    return name, want or None


def pick_gguf(files: dict, want: str | None = None) -> str | None:
    """The GGUF file to use from a repo: the asked-for quantization, else a good default."""
    ggufs = [f for f in files if f.endswith(".gguf") and "mmproj" not in f.lower()]
    if not ggufs:
        return None
    ggufs = [f for f in ggufs if "-of-" not in f or "00001-of-00001" in f]  # models split in parts: not yet
    if not ggufs:
        raise ValueError("this repo only has GGUF models split into several files, which Omnicompute can't read yet")
    for q in [want] if want else GGUF_PREFERENCE:
        hit = [f for f in ggufs if q.lower() in f.lower()]
        if hit:
            return min(hit, key=len)
    if want:
        raise ValueError(f"no {want} file in this repo; files: {', '.join(f.split('/')[-1] for f in ggufs)}")
    return min(ggufs, key=lambda f: files[f])


def remote_config(repo: str, token: str | None = None) -> tuple[dict, int]:
    """(Omnicompute config, total download size in bytes) without downloading the weights."""
    repo, want = split_spec(repo)
    files = list_repo_files(repo, token)
    weights = [f for f in files if f.endswith(".safetensors")]
    gg = None if weights else pick_gguf(files, want)
    if gg:  # read just the start of the file, where GGUF keeps the model's settings
        from .gguf_source import remote_gguf_config

        return remote_gguf_config(lambda a, b: _range(f"{HF}/{repo}/resolve/main/{gg}", token, a, b), repo), files[gg]
    if not weights:
        raise ValueError(f"{repo} has no .safetensors or .gguf weights (only .bin files)")
    with _request(f"{HF}/{repo}/resolve/main/config.json", token) as r:
        hf = json.load(r)
    size = sum(files[f] for f in weights + [m for m in META_FILES if m in files])
    return omnicompute_config(hf, repo), size


def _range(url: str, token: str | None, start: int, end: int) -> bytes:
    with _request(url, token, {"Range": f"bytes={start}-{end - 1}"}) as r:
        return r.read(end - start)  # never the whole file, even if the server ignores the range


def download(repo: str, dest_root: Path, token: str | None = None, progress: Download | None = None) -> Path:
    progress = progress or Download(repo)
    repo, want = split_spec(repo)
    files = list_repo_files(repo, token)
    weights = [f for f in files if f.endswith(".safetensors")]
    if not weights:
        gg = pick_gguf(files, want)
        if gg is None:
            raise ValueError(f"{repo} has no .safetensors or .gguf weights (only .bin files)")
        wanted = [gg]  # a GGUF file carries its own config and tokenizer
    elif "config.json" not in files or "tokenizer.json" not in files:
        raise ValueError(f"{repo} is missing config.json or tokenizer.json")
    else:
        wanted = [f for f in META_FILES if f in files] + weights
    folder = dest_root / repo.replace("/", "--")
    folder.mkdir(parents=True, exist_ok=True)
    progress.total = sum(files[f] for f in wanted)
    progress.done = sum(min(files[f], (folder / f).stat().st_size) for f in wanted if (folder / f).exists())
    progress.state = "downloading"
    for name in wanted:
        target = folder / name
        if target.exists() and target.stat().st_size == files[name]:
            continue
        progress.file = name
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_suffix(target.suffix + ".part")
        have = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        with _request(f"{HF}/{repo}/resolve/main/{name}", token, headers) as r, open(part, "ab" if have else "wb") as out:
            if have and r.status != 206:  # server ignored the range: start over
                out.truncate(0)
                progress.done -= have
            while chunk := r.read(1 << 20):
                out.write(chunk)
                progress.done += len(chunk)
        os.replace(part, target)
    progress.folder = folder
    return folder


def _gguf_in(folder: Path) -> Path | None:
    files = [f for f in folder.rglob("*.gguf") if "mmproj" not in f.name.lower()]
    return min(files, key=lambda f: len(str(f))) if files else None


def _complete(folder: Path) -> bool:
    """True when a downloaded model folder has everything needed to load it offline."""
    if not (folder / "config.json").exists() and _gguf_in(folder) is not None:
        return not any(f.suffix == ".part" for f in folder.rglob("*"))
    if not (folder / "config.json").exists() or not (folder / "tokenizer.json").exists():
        return False
    if any(f.suffix == ".part" for f in folder.iterdir()):
        return False
    index = folder / "model.safetensors.index.json"
    if index.exists():
        return all((folder / f).exists() for f in set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values()))
    return any(folder.glob("*.safetensors"))


def _open(folder: Path, name: str):
    if not (folder / "config.json").exists() and _gguf_in(folder) is not None:
        from .gguf_source import GGUFSource

        return GGUFSource(_gguf_in(folder), name=name)
    return HFSource(folder, name=name)


def resolve(spec: str, cache: Path, token: str | None = None, progress: Download | None = None):
    """'hf:org/name[:QUANT]', 'org/name', a local folder, or a .gguf file ('gguf:/path/x.gguf')
    -> HFSource or GGUFSource (downloading if needed)."""
    path = Path(spec.removeprefix("gguf:").removeprefix("hf:")).expanduser()
    if (spec.startswith("gguf:") or path.suffix == ".gguf") and path.is_file():
        from .gguf_source import GGUFSource

        if progress:
            progress.state = "loading"
        return GGUFSource(path, name=f"gguf:{path}")
    if spec.startswith("gguf:"):
        raise FileNotFoundError(f"no GGUF file at {path}")
    if not spec.startswith("hf:") and path.is_dir():
        return _open(path, path.name)
    repo = spec.removeprefix("hf:")
    local = cache / split_spec(repo)[0].replace("/", "--")
    if local.is_dir() and _complete(local) and (split_spec(repo)[1] is None or not (local / "config.json").exists()
                                                and split_spec(repo)[1].lower() in str(_gguf_in(local)).lower()):
        if progress:  # already downloaded: no network needed
            progress.state = "loading"
        return _open(local, repo)
    folder = download(repo, cache, token, progress)
    if progress:
        progress.state = "loading"
    return _open(folder, repo)


def gguf_dirs(cache: Path) -> list[Path]:
    """Folders searched for GGUF files: Omnicompute's own (uploads, imports) and LM Studio's."""
    from .gguf_source import lmstudio_dirs

    extra = [Path(p).expanduser() for p in os.environ.get("OMNICOMPUTE_GGUF_DIRS", "").split(os.pathsep) if p]
    return [cache / "gguf", *lmstudio_dirs(), *extra]


def downloaded(cache: Path) -> list[dict]:
    """Models already on this computer: [{"repo", "bytes", "complete"}], plus GGUF files found in
    Omnicompute's gguf folder and LM Studio's model folders ("repo" is then "gguf:<path>")."""
    out = []
    if cache.is_dir():
        for folder in sorted(cache.iterdir()):
            if not folder.is_dir() or "--" not in folder.name:
                continue
            files = [f for f in folder.rglob("*") if f.is_file()]
            out.append({"repo": folder.name.replace("--", "/", 1), "bytes": sum(f.stat().st_size for f in files),
                        "complete": _complete(folder), "format": "gguf" if _gguf_in(folder) else "safetensors"})
    from .gguf_source import find_gguf

    own = (cache / "gguf").resolve()
    for f in find_gguf(gguf_dirs(cache)):
        out.append({"repo": f"gguf:{f}", "name": f.name, "bytes": f.stat().st_size, "complete": True, "format": "gguf",
                    "where": "uploaded" if own in f.resolve().parents else "LM Studio" if "lm" in str(f).lower() else str(f.parent),
                    "removable": own in f.resolve().parents})
    return out


def remove(cache: Path, repo: str) -> int:
    """Delete a downloaded model. Returns the bytes freed."""
    if repo.startswith("gguf:"):  # only files Omnicompute itself added; LM Studio's stay LM Studio's
        f = Path(repo[5:]).resolve()
        if (cache / "gguf").resolve() not in f.parents or not f.is_file():
            raise FileNotFoundError("only GGUF files uploaded to Omnicompute can be removed here")
        size = f.stat().st_size
        f.unlink()
        return size
    folder = (cache / repo.replace("/", "--")).resolve()
    if folder.parent != cache.resolve() or not folder.is_dir():
        raise FileNotFoundError(f"{repo} is not downloaded")
    freed = sum(f.stat().st_size for f in folder.rglob("*") if f.is_file())
    shutil.rmtree(folder)
    return freed


class Loader:
    """Runs downloads in a background thread so the hub stays responsive."""

    def __init__(self, cache: Path, token: str | None = None):
        self.cache, self.token = cache, token
        self.current: Download | None = None
        self.lock = threading.Lock()

    def start(self, repo: str, token: str | None, on_ready) -> Download:
        with self.lock:
            if self.current and self.current.state in ("starting", "downloading", "loading"):
                raise RuntimeError(f"already loading {self.current.repo}")
            d = self.current = Download(repo)

        def run():
            t0 = time.monotonic()
            try:
                spec = repo if repo.startswith("gguf:") else "hf:" + repo
                src = resolve(spec, self.cache, token or self.token, d)
                on_ready(src)
                d.state = "ready"
                log.info("%s ready in %.0fs", repo, time.monotonic() - t0)
            except Exception as e:
                d.state, d.error = "error", str(e)
                log.warning("loading %s failed: %s", repo, e)

        threading.Thread(target=run, daemon=True).start()
        return d
