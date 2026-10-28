import json
import struct

import numpy as np

from edgemesh.convert import from_hf_dir
from edgemesh.model import Shard, load_model, random_model
from edgemesh.tokenizer import GPT2Tokenizer, _bytes_to_unicode


def write_safetensors(path, tensors):
    header, off, blobs = {}, 0, []
    for k, v in tensors.items():
        raw = np.ascontiguousarray(v, "<f4").tobytes()
        header[k] = {"dtype": "F32", "shape": list(v.shape), "data_offsets": [off, off + len(raw)]}
        off += len(raw)
        blobs.append(raw)
    h = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h + b"".join(blobs))


def test_convert_hf_layout(tmp_path):
    cfg, w = random_model(n_layer=2, n_embd=32, n_head=4, n_ctx=64)
    hf = {"transformer.wte.weight": w["wte"], "transformer.wpe.weight": w["wpe"],
          "transformer.ln_f.weight": w["ln_f.w"], "transformer.ln_f.bias": w["ln_f.b"]}
    for k, v in w.items():
        if k.startswith("h."):
            hf["transformer." + k[:-2] + (".weight" if k.endswith(".w") else ".bias")] = v
    write_safetensors(tmp_path / "model.safetensors", hf)
    (tmp_path / "config.json").write_text(json.dumps({"n_layer": 2, "n_embd": 32, "n_head": 4,
                                                      "n_positions": 64, "vocab_size": 256}))
    cfg2, w2 = load_model(from_hf_dir(str(tmp_path), str(tmp_path / "out.emm")))
    ids = np.array([3, 4, 5], np.int32)
    np.testing.assert_array_equal(Shard(cfg, w, 0, 2).forward("j", ids, 0), Shard(cfg2, w2, 0, 2).forward("j", ids, 0))
    assert cfg2["tokenizer"] == "gpt2"


def test_gpt2_tokenizer_roundtrip(tmp_path):
    chars = list(_bytes_to_unicode().values())
    vocab = {c: i for i, c in enumerate(chars)}
    merges = [("h", "e"), ("l", "l"), ("he", "ll"), ("Ġ", "w")]
    for a, b in merges:
        vocab[a + b] = len(vocab)
    (tmp_path / "vocab.json").write_text(json.dumps(vocab))
    (tmp_path / "merges.txt").write_text("#version: 0.2\n" + "\n".join(f"{a} {b}" for a, b in merges))
    tok = GPT2Tokenizer(str(tmp_path / "vocab.json"), str(tmp_path / "merges.txt"))
    ids = tok.encode("hello world, ünïcode!")
    assert ids[0] == vocab["hell"] and vocab["Ġw"] in ids
    assert tok.decode(ids) == "hello world, ünïcode!"
