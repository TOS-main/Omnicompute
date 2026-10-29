"""Gemma 4 from a GGUF file, checked against the transformers reference implementation.

A tiny random Gemma 4 (sliding and full layers with different head sizes, per-layer embeddings,
shared key/value layers, final logit softcapping) is written as GGUF the way llama.cpp's
converter does, then run whole and split across devices."""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
gguf = pytest.importorskip("gguf")

from omnicompute.gguf_source import GGUFSource  # noqa: E402
from omnicompute.model import Shard, shard_tensors  # noqa: E402

LAYER_TYPES = ["sliding_attention", "sliding_attention", "full_attention", "sliding_attention", "sliding_attention",
               "full_attention"]


def _tokenizer(folder):
    from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE(byte_fallback=True))
    tok.normalizer = normalizers.Replace(" ", "▁")
    tok.pre_tokenizer = pre_tokenizers.Split("\n", "isolated")
    tok.decoder = decoders.Sequence([decoders.Replace("▁", " "), decoders.ByteFallback(), decoders.Fuse()])
    special = ["<pad>", "<eos>", "<bos>", "<unk>", "<|turn>", "<turn|>"]
    trainer = trainers.BpeTrainer(vocab_size=700, special_tokens=special + [f"<0x{i:02X}>" for i in range(256)],
                                  show_progress=False)
    tok.train_from_iterator(["Hello world, how are you today?", "The quick brown fox jumps over the lazy dog.",
                             "def f(x):\n    return x + 1\n", "tides follow the moon"] * 50, trainer)
    tok.save(str(folder / "tokenizer.json"))
    return tok


def _write_gguf(model, tok, path, ple_q8=False):
    """What llama.cpp's convert_hf_to_gguf.py writes for Gemma 4 (Gemma4Model), for a text model."""
    c = model.config
    L = c.num_hidden_layers
    w = gguf.GGUFWriter(str(path), "gemma4")
    full, slide = c.per_layer_config[LAYER_TYPES.index("full_attention")], c.per_layer_config[0]
    full_hd, full_kv, swa_hd, swa_kv = full.head_dim, full.num_key_value_heads, slide.head_dim, slide.num_key_value_heads
    swa = [t == "sliding_attention" for t in c.layer_types]
    w.add_context_length(c.max_position_embeddings)
    w.add_embedding_length(c.hidden_size)
    w.add_block_count(L)
    w.add_feed_forward_length(c.intermediate_size)
    w.add_head_count(c.num_attention_heads)
    w.add_head_count_kv([swa_kv if s else full_kv for s in swa])
    w.add_layer_norm_rms_eps(c.rms_norm_eps)
    w.add_key_length(full_hd)
    w.add_value_length(full_hd)
    w.add_key_length_swa(swa_hd)
    w.add_value_length_swa(swa_hd)
    w.add_rope_freq_base(c.rope_parameters["full_attention"]["rope_theta"])
    w.add_rope_freq_base_swa(c.rope_parameters["sliding_attention"]["rope_theta"])
    w.add_rope_dimension_count(full_hd)
    w.add_rope_dimension_count_swa(swa_hd)
    w.add_sliding_window(c.sliding_window)
    w.add_sliding_window_pattern(swa)
    w.add_final_logit_softcapping(c.final_logit_softcapping)
    w.add_embedding_length_per_layer_input(c.hidden_size_per_layer_input)
    w.add_shared_kv_layers(c.num_kv_shared_layers)
    # tokenizer
    vocab = json.loads(tok.to_str())["model"]
    inv = {i: t for t, i in tok.get_vocab().items()}
    tokens = [inv[i] for i in range(len(inv))]
    w.add_tokenizer_model("gemma4")
    w.add_token_list(tokens)
    w.add_token_scores([0.0] * len(tokens))
    w.add_token_types([3 if t.startswith("<") and t.endswith(">") and not t.startswith("<0x") else
                       6 if t.startswith("<0x") else 1 for t in tokens])
    w.add_token_merges([" ".join(m) if isinstance(m, list) else m for m in vocab["merges"]])
    w.add_bos_token_id(2)
    w.add_eos_token_id(1)
    w.add_add_bos_token(True)
    w.add_chat_template("{{ bos_token }}{% for m in messages %}<|turn>{{ m['role'] }}\n{{ m['content'] }}<turn|>\n"
                        "{% endfor %}<|turn>model\n")
    # tensors
    pf = c.rope_parameters["full_attention"]["partial_rotary_factor"]
    rot = int(full_hd * pf / 2)
    w.add_tensor("rope_freqs.weight", np.array([1.0] * rot + [1e30] * (full_hd // 2 - rot), np.float32))
    names = gguf.get_tensor_name_map(gguf.MODEL_ARCH.GEMMA4, L)
    for n, p in model.model.state_dict().items():
        a = p.detach().float().numpy()
        g = names.get_name("model." + n, try_suffixes=(".weight",))
        if g is None and n.endswith("layer_scalar"):
            g = f"blk.{n.split('.')[1]}.layer_output_scale"
        assert g is not None, n
        g = g if g.endswith(".weight") else g + ".weight"
        if g == "per_layer_token_embd.weight" and ple_q8:
            w.add_tensor(g, gguf.quants.quantize(a, gguf.GGMLQuantizationType.Q8_0),
                         raw_dtype=gguf.GGMLQuantizationType.Q8_0)
        else:
            w.add_tensor(g, a)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


@pytest.fixture(scope="module", params=[False, True], ids=["v_proj", "k_eq_v"])
def gemma4(request, tmp_path_factory):
    from transformers import Gemma4ForCausalLM, Gemma4TextConfig

    folder = tmp_path_factory.mktemp("gemma4")
    tok = _tokenizer(folder)
    V = tok.get_vocab_size()
    torch.manual_seed(0)
    cfg = Gemma4TextConfig(
        vocab_size=V, hidden_size=64, intermediate_size=96, num_hidden_layers=6, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, global_head_dim=32, num_global_key_value_heads=1,
        max_position_embeddings=512, sliding_window=8, layer_types=LAYER_TYPES, final_logit_softcapping=30.0,
        vocab_size_per_layer_input=V, hidden_size_per_layer_input=16, num_kv_shared_layers=2,
        attention_k_eq_v=request.param, pad_token_id=0, eos_token_id=1, bos_token_id=2, tie_word_embeddings=True)
    model = Gemma4ForCausalLM(cfg).eval()
    with torch.no_grad():
        for n, p in model.named_parameters():
            p.copy_(1 + 0.3 * torch.randn_like(p) if "norm" in n else 0.08 * torch.randn_like(p))
        for layer in model.model.layers:
            layer.layer_scalar.fill_(0.9)
    path = folder / "tiny-gemma4.gguf"
    _write_gguf(model, tok, path)
    return model, tok, path


def _reference(model, ids):
    with torch.no_grad():
        return model(torch.tensor([ids])).logits[0, -1].float().numpy()


def test_gemma4_matches_transformers_whole_and_split(gemma4):
    model, tok, path = gemma4
    src = GGUFSource(path, dtype=np.float32)
    cfg = src.cfg
    assert cfg["arch"] == "gemma4" and cfg["swa"] == [True, True, False, True, True, False]
    ids = [2] + [(i * 37) % 390 + 5 for i in range(13)]
    assert len(ids) > cfg["window"]  # the sliding window matters
    ref = _reference(model, ids)
    ref_next = _reference(model, ids + [7])
    for cuts in ([0, 6], [0, 2, 5, 6], [0, 3, 4, 6], [0, 1, 2, 3, 4, 5, 6]):
        shards = [Shard(cfg, shard_tensors(src, a, b), a, b) for a, b in zip(cuts, cuts[1:])]

        def run(x, pos):
            t = {"x": np.array(x, np.int32)}
            for s in shards:
                t = s.forward_ex("job", t, pos)
            return t["x"]

        np.testing.assert_allclose(run(ids, 0), ref, atol=2e-3, rtol=1e-3)
        np.testing.assert_allclose(run([7], len(ids)), ref_next, atol=2e-3, rtol=1e-3)  # the KV caches


def test_gemma4_tokenizer_matches(gemma4):
    from omnicompute.tokenizer import GGUFTokenizer

    model, tok, path = gemma4
    ours = GGUFTokenizer(GGUFSource(path))
    text = "Hello world, how are you today? def f(x): return x + 1 ünïcode"
    assert ours.tok.encode(text).ids == [2] + tok.encode(text).ids
    assert ours.decode(ours.tok.encode(text).ids) == text
    pure = GGUFTokenizer.__new__(GGUFTokenizer)
    import omnicompute.tokenizer as T

    fast = T._fast_gemma4
    try:
        T._fast_gemma4 = lambda *a: None  # the pure-Python path (phones without `tokenizers`)
        pure = GGUFTokenizer(GGUFSource(path))
    finally:
        T._fast_gemma4 = fast
    assert pure.tok.encode(text).ids == ours.tok.encode(text).ids
    assert 5 in ours.eos_ids or tok.token_to_id("<turn|>") in ours.eos_ids
    chat = ours.encode([{"role": "user", "content": "hi"}], chat=True)
    assert chat[0] == 2 and tok.token_to_id("<|turn>") in chat


def test_gemma4_quantized_per_layer_table_stays_quantized(gemma4, tmp_path):
    model, tok, _ = gemma4
    path = tmp_path / "q8.gguf"
    _write_gguf(model, tok, path, ple_q8=True)
    src = GGUFSource(path, dtype=np.float32)
    t = shard_tensors(src, 0, 6)
    assert t["ple_embed.q"].dtype == np.uint8  # sent and kept as Q8_0 rows
    ids = [2] + [(i * 37) % 390 + 5 for i in range(13)]
    out = Shard(src.cfg, t, 0, 6).forward_ex("j", {"x": np.array(ids, np.int32)}, 0)["x"]
    np.testing.assert_allclose(out, _reference(model, ids), atol=0.05)


async def test_gemma4_across_three_devices_matches_transformers(gemma4):
    import tempfile
    from pathlib import Path

    from omnicompute.hub import Hub
    from omnicompute.limits import LimitStore
    from omnicompute.node import MeshNode
    from omnicompute.worker import Worker

    model, tok, path = gemma4
    key = "gemma4-test-key"
    hub = Hub(key, "127.0.0.1", 0, GGUFSource(path))
    await hub.start()
    nodes = []
    try:
        for name, score in (("a", 2), ("b", 1), ("c", 3)):
            n = MeshNode(name, key, f"http://127.0.0.1:{hub.port}", {"role": "worker", "kind": "native", "score": score, "mem_mb": 1024})
            Worker(LimitStore(Path(tempfile.mkdtemp()) / "l.json", {"cpu": 100, "always": True})).attach(n)
            await n.start()
            nodes.append(n)
        events = [ev async for ev in hub.orchestrator.generate("Hello world", 10)]
        assert len(events[0]["plan"]) == 3
        ids = [e["id"] for e in events if e["type"] == "token"]
        prompt = hub.orchestrator.tokenizer.encode("Hello world")
        with torch.no_grad():
            want = model.generate(torch.tensor([prompt]), max_new_tokens=len(ids), do_sample=False)[0, len(prompt):].tolist()
        assert ids == want[:len(ids)]
    finally:
        for n in nodes:
            await n.close()
        await hub.close()
