"""GGUF files: written here the way llama.cpp's converter writes them, then loaded back."""

from pathlib import Path

import numpy as np
import pytest

gguf = pytest.importorskip("gguf")

from edgemesh.bpe import BYTE_TO_CHAR  # noqa: E402
from edgemesh.gguf_source import GGUFSource  # noqa: E402
from edgemesh.model import Shard, random_model  # noqa: E402
from edgemesh.tokenizer import load_tokenizer  # noqa: E402
from test_mesh import model_file  # noqa: E402,F401

SPECIAL = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
TEMPLATE = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
            "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")


def _permute(w, n_head):  # llama.cpp convert_hf_to_gguf.py LlamaModel.permute
    return w.reshape(n_head, 2, w.shape[0] // n_head // 2, *w.shape[1:]).swapaxes(1, 2).reshape(w.shape)


def write_gguf(path: Path, cfg: dict, w: dict, arch: str, qtype) -> list[str]:
    from gguf import GGMLQuantizationType as Q

    gw = gguf.GGUFWriter(str(path), arch)
    gw.add_block_count(cfg["n_layer"])
    gw.add_context_length(cfg["n_ctx"])
    gw.add_embedding_length(cfg["n_embd"])
    gw.add_feed_forward_length(cfg["ffn"])
    gw.add_head_count(cfg["n_head"])
    gw.add_head_count_kv(cfg["n_kv_head"])
    gw.add_rope_freq_base(cfg["rope_theta"])
    gw.add_layer_norm_rms_eps(cfg["rms_eps"])
    tokens = [BYTE_TO_CHAR[b] for b in range(256)]
    merges = []
    for a, b in [("h", "e"), ("l", "l"), ("he", "ll"), ("Ġ", "w"), ("o", "r")]:
        merges.append(f"{a} {b}")
        tokens.append(a + b)
    tokens += SPECIAL
    tokens += [f"[PAD{i}]" for i in range(cfg["vocab"] - len(tokens))]
    gw.add_tokenizer_model("gpt2")
    gw.add_tokenizer_pre("qwen2")
    gw.add_token_list(tokens)
    gw.add_token_types([3 if t in SPECIAL else 1 for t in tokens])
    gw.add_token_merges(merges)
    gw.add_eos_token_id(tokens.index("<|im_end|>"))
    gw.add_chat_template(TEMPLATE)

    def put(name, a):
        a = np.ascontiguousarray(a, np.float32)
        if a.ndim == 1 or qtype == Q.F32:
            gw.add_tensor(name, a)
        elif qtype == Q.F16:
            gw.add_tensor(name, a.astype(np.float16))
        else:
            gw.add_tensor(name, gguf.quants.quantize(a, qtype), raw_dtype=qtype)

    put("token_embd.weight", w["embed"])
    put("output_norm.weight", w["norm_f"])
    if "lm_head" in w:
        put("output.weight", w["lm_head"].T)
    H, KV = cfg["n_head"], cfg["n_kv_head"]
    for i in range(cfg["n_layer"]):
        p, b = f"h.{i}.", f"blk.{i}."
        put(b + "attn_norm.weight", w[p + "attn_norm"])
        put(b + "ffn_norm.weight", w[p + "mlp_norm"])
        for ours, theirs in [("q", "attn_q"), ("k", "attn_k"), ("v", "attn_v"), ("o", "attn_output"),
                             ("gate", "ffn_gate"), ("up", "ffn_up"), ("down", "ffn_down")]:
            m = w[p + ours + ".w"].T  # GGUF rows are outputs
            bias = w.get(p + ours + ".b")
            if arch == "llama" and ours in ("q", "k"):
                m = _permute(m, H if ours == "q" else KV)
                bias = None if bias is None else _permute(bias, H if ours == "q" else KV)
            put(b + theirs + ".weight", m)
            if bias is not None:
                put(b + theirs + ".bias", bias)
        if cfg.get("qk_norm"):
            put(b + "attn_q_norm.weight", w[p + "q_norm"])
            put(b + "attn_k_norm.weight", w[p + "k_norm"])
    gw.write_header_to_file()
    gw.write_kv_data_to_file()
    gw.write_tensors_to_file()
    gw.close()
    return tokens


@pytest.mark.parametrize("arch,kw", [("llama", dict(n_kv_head=2, tie=False)),
                                     ("qwen2", dict(n_kv_head=2, qkv_bias=True, tie=True)),
                                     ("qwen3", dict(n_kv_head=2, qk_norm=True, tie=True))])
@pytest.mark.parametrize("qname", ["F32", "Q8_0", "Q4_0"])
def test_gguf_matches_the_original_model(tmp_path, arch, kw, qname):
    from gguf import GGMLQuantizationType as Q

    q = getattr(Q, qname)
    cfg, w = random_model(n_layer=2, n_embd=256, n_head=4, vocab=320, arch="llama", **kw)
    write_gguf(tmp_path / "m.gguf", cfg, w, arch, q)
    src = GGUFSource(tmp_path / "m.gguf")
    c = src.cfg
    assert (c["n_layer"], c["n_kv_head"], c["head_dim"], c["qkv_bias"], c["qk_norm"], c["tie_embeddings"]) == \
        (2, 2, 64, bool(kw.get("qkv_bias")), bool(kw.get("qk_norm")), kw["tie"])
    w16 = {k: v.astype(np.float16).astype(np.float32) if v.ndim == 2 else v for k, v in w.items()}
    ref = Shard(cfg, w16, 0, 2)  # EdgeMesh sends big matrices as float16
    mine = Shard(c, {k: src.get(k).astype(np.float32) for k in ref.w}, 0, 2)
    ids = np.array([3, 1, 4, 1, 5], np.int32)
    a, b = mine.forward("j", ids, 0), ref.forward("j", ids, 0)
    if qname == "Q4_0":  # 4-bit rounding on random weights: close, not equal
        assert np.corrcoef(a.ravel(), b.ravel())[0, 1] > 0.9
        return
    tol = {"F32": 1e-4, "Q8_0": 0.03}[qname]
    assert np.abs(a - b).max() <= tol * np.abs(b).max()
    assert (a.argmax(-1) == b.argmax(-1)).all()


def test_gguf_tokenizer_and_chat_template(tmp_path):
    from gguf import GGMLQuantizationType as Q

    cfg, w = random_model(n_layer=1, n_embd=128, n_head=4, vocab=300, arch="llama", n_kv_head=2)
    tokens = write_gguf(tmp_path / "t.gguf", cfg, w, "qwen2", Q.F16)
    tok = load_tokenizer(GGUFSource(tmp_path / "t.gguf"))
    ids = tok.encode("hello world")
    assert [tokens[i] for i in ids] == ["hell", "o", "Ġw", "or", "l", "d"]
    assert tok.decode(ids) == "hello world"
    chat = tok.encode([{"role": "user", "content": "hi"}], chat=True)
    assert chat[0] == tokens.index("<|im_start|>") and chat[-1] == tokens.index(BYTE_TO_CHAR[10])
    assert tokens.index("<|im_end|>") in tok.eos_ids
    assert tok.decode(chat) == "user\nhi\nassistant\n"  # special tokens are left out


async def test_upload_and_load_a_gguf_then_generate(tmp_path, model_file):
    """A real Llama (transformers), written as GGUF like llama.cpp does, uploaded in pieces through
    the hub, loaded, and run across two devices: same tokens as transformers."""
    import asyncio
    import json

    from aiohttp import ClientSession

    from edgemesh.hf import HFSource
    from edgemesh.model import shard_keys
    from test_mesh import KEY, start_mesh

    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    hcfg = transformers.LlamaConfig(vocab_size=300, hidden_size=128, intermediate_size=256, num_hidden_layers=3,
                                    num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
                                    tie_word_embeddings=False, rope_theta=10000.0)
    torch.manual_seed(2)
    model = transformers.AutoModelForCausalLM.from_config(hcfg).eval()
    (tmp_path / "hf").mkdir()
    model.save_pretrained(tmp_path / "hf", safe_serialization=True)
    (tmp_path / "hf" / "tokenizer.json").write_text("{}")
    src = HFSource(tmp_path / "hf", dtype=np.float32)
    w = {k: src.get(k) for k in shard_keys(src.cfg, 0, src.cfg["n_layer"])}
    from gguf import GGMLQuantizationType as Q

    write_gguf(tmp_path / "tiny-llama.gguf", src.cfg, w, "llama", Q.F32)
    blob = (tmp_path / "tiny-llama.gguf").read_bytes()

    hub, nodes = await start_mesh(model_file[0], [("a", False, 2), ("b", False, 1)])
    hub.loader.cache = tmp_path / "cache"
    base = f"http://127.0.0.1:{hub.port}"
    try:
        async with ClientSession(headers={"X-Mesh-Key": KEY}) as s:
            piece, off = 50_000, 0
            while True:  # the web page uploads like this, in pieces
                last = off + piece >= len(blob)
                q = f"name=tiny-llama.gguf&offset={off}" + ("&done=1" if last else "")
                async with s.post(f"{base}/api/models/upload?{q}", data=blob[off:off + piece]) as r:
                    j = await r.json()
                off = j["have"]
                if last:
                    break
            assert off == len(blob)
            async with s.get(base + "/api/models") as r:
                listed = (await r.json())["models"]
            assert any(m["repo"] == j["repo"] and m["removable"] for m in listed)
            async with s.post(base + "/api/model", json={"repo": j["repo"]}) as r:
                assert r.status == 200, await r.text()
            for _ in range(200):
                async with s.get(base + "/api/model") as r:
                    st = await r.json()
                if st["download"]["state"] in ("ready", "error"):
                    break
                await asyncio.sleep(0.05)
            assert st["download"]["state"] == "ready", st
            prompt = "hello world"
            async with s.post(base + "/api/generate", json={"prompt": prompt, "max_new": 6}) as r:
                events = [json.loads(line) async for line in r.content]
        tokens = [e["id"] for e in events if e["type"] == "token"]
        assert len(events[0]["plan"]) == 2
        ids = hub.orchestrator.tokenizer.encode(prompt)
        with torch.no_grad():
            ref = model.generate(torch.tensor([ids]), max_new_tokens=6, do_sample=False,
                                 eos_token_id=None, pad_token_id=0)[0, len(ids):].tolist()
        assert tokens == ref
    finally:
        for n in nodes:
            await n.close()
        await hub.close()
