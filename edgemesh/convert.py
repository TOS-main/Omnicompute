"""Turn a Hugging Face GPT-2 checkpoint into an EdgeMesh model file.

Works with any GPT-2-architecture repo that ships model.safetensors
(gpt2, gpt2-medium, gpt2-large, gpt2-xl, distilgpt2, ...). Reads safetensors
directly so no torch/transformers install is needed.
"""

from __future__ import annotations

import json
import os
import struct
import urllib.request

import numpy as np

from .model import save_model

FILES = ["config.json", "model.safetensors", "vocab.json", "merges.txt"]
REPOS = {"gpt2": "openai-community/gpt2", "gpt2-medium": "openai-community/gpt2-medium",
         "gpt2-large": "openai-community/gpt2-large", "gpt2-xl": "openai-community/gpt2-xl",
         "distilgpt2": "distilbert/distilgpt2"}


def read_safetensors(path: str) -> dict:
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
        data = f.read()
    out = {}
    for name, t in header.items():
        if name == "__metadata__":
            continue
        a, b = t["data_offsets"]
        raw = data[a:b]
        if t["dtype"] == "F32":
            arr = np.frombuffer(raw, "<f4")
        elif t["dtype"] == "F16":
            arr = np.frombuffer(raw, "<f2").astype(np.float32)
        elif t["dtype"] == "BF16":
            arr = (np.frombuffer(raw, "<u2").astype(np.uint32) << 16).view(np.float32)
        else:
            continue  # e.g. bool attention-mask buffers
        out[name] = arr.reshape(t["shape"])
    return out


def from_hf_dir(src: str, out_path: str) -> str:
    with open(os.path.join(src, "config.json")) as f:
        hf = json.load(f)
    t = {k.removeprefix("transformer."): v for k, v in read_safetensors(os.path.join(src, "model.safetensors")).items()}
    cfg = {"n_layer": hf["n_layer"], "n_embd": hf["n_embd"], "n_head": hf["n_head"],
           "n_ctx": hf.get("n_positions", hf.get("n_ctx", 1024)), "vocab": hf["vocab_size"],
           "tokenizer": "gpt2", "vocab_file": "vocab.json", "merges_file": "merges.txt",
           "name": hf.get("_name_or_path") or os.path.basename(os.path.normpath(src))}
    w = {"wte": t["wte.weight"], "wpe": t["wpe.weight"], "ln_f.w": t["ln_f.weight"], "ln_f.b": t["ln_f.bias"]}
    for i in range(cfg["n_layer"]):
        for part in ("ln_1", "ln_2", "attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"):
            w[f"h.{i}.{part}.w"] = t[f"h.{i}.{part}.weight"]
            w[f"h.{i}.{part}.b"] = t[f"h.{i}.{part}.bias"]
    save_model(out_path, cfg, {k: np.ascontiguousarray(v, np.float32) for k, v in w.items()})
    return out_path


def convert_gpt2(out_dir: str, variant: str = "gpt2") -> str:
    """Download (if needed) and convert. Returns the model file path, which `edgemesh host --model` takes."""
    os.makedirs(out_dir, exist_ok=True)
    repo = REPOS.get(variant, variant)
    for name in FILES:
        dest = os.path.join(out_dir, name)
        if not os.path.exists(dest):
            url = f"https://huggingface.co/{repo}/resolve/main/{name}"
            print(f"downloading {url}")
            urllib.request.urlretrieve(url, dest + ".part")
            os.replace(dest + ".part", dest)
    return from_hf_dir(out_dir, os.path.join(out_dir, f"{variant.replace('/', '_')}.emm"))
