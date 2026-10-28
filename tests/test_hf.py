"""Checks our numpy models against the reference Hugging Face transformers implementation,
using tiny randomly initialised checkpoints saved in the normal HF format."""

import json

import numpy as np
import pytest

from edgemesh.hf import HFSource
from edgemesh.model import Shard, shard_tensors

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

COMMON = dict(vocab_size=97, hidden_size=64, intermediate_size=96, num_hidden_layers=3,
              num_attention_heads=4, max_position_embeddings=128)
CONFIGS = {
    "llama-gqa-llama3rope": ("LlamaConfig", dict(num_key_value_heads=2, rope_theta=500000.0, tie_word_embeddings=True,
                                                  rope_scaling={"rope_type": "llama3", "factor": 8.0,
                                                                "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                                                                "original_max_position_embeddings": 32})),
    "mistral": ("MistralConfig", dict(num_key_value_heads=2, sliding_window=None)),
    "qwen2": ("Qwen2Config", dict(num_key_value_heads=2, tie_word_embeddings=True)),
    "qwen3": ("Qwen3Config", dict(num_key_value_heads=2, head_dim=32)),
    "phi3": ("Phi3Config", dict(num_key_value_heads=4, pad_token_id=0)),
    "gpt2": ("GPT2Config", dict(vocab_size=97, n_embd=64, n_layer=3, n_head=4, n_positions=128)),
}


def make_checkpoint(tmp_path, kind, dtype=torch.float32):
    cls_name, extra = CONFIGS[kind]
    cfg_cls = getattr(transformers, cls_name)
    cfg = cfg_cls(**extra) if kind == "gpt2" else cfg_cls(**{**COMMON, **extra})
    torch.manual_seed(0)
    model = transformers.AutoModelForCausalLM.from_config(cfg).eval()
    with torch.no_grad():  # make norms non-trivial so a wrong norm implementation shows up
        for n, p in model.named_parameters():
            if "norm" in n or "ln_" in n:
                p.add_(torch.randn_like(p) * 0.1)
    model = model.to(dtype)
    model.save_pretrained(tmp_path, safe_serialization=True)
    return model


@pytest.mark.parametrize("kind", list(CONFIGS))
def test_matches_transformers(tmp_path, kind):
    model = make_checkpoint(tmp_path, kind)
    ids = [5, 17, 3, 88, 42, 7]
    with torch.no_grad():
        ref = model(torch.tensor([ids + [11, 12]])).logits[0].float().numpy()

    src = HFSource(tmp_path, dtype=np.float32)
    n = src.cfg["n_layer"]
    shards = [Shard(src.cfg, shard_tensors(src, s, e), s, e) for s, e in [(0, 1), (1, n)]]

    def run(x, pos):
        for sh in shards:
            x = sh.forward("j", x, pos)
        return x

    np.testing.assert_allclose(run(np.array(ids, np.int32), 0), ref[len(ids) - 1], atol=2e-4, rtol=2e-3)
    np.testing.assert_allclose(run(np.array([11], np.int32), len(ids)), ref[len(ids)], atol=2e-4, rtol=2e-3)
    np.testing.assert_allclose(run(np.array([12], np.int32), len(ids) + 1), ref[len(ids) + 1], atol=2e-4, rtol=2e-3)


def test_bf16_checkpoint_in_float16(tmp_path):
    model = make_checkpoint(tmp_path, "qwen2", torch.bfloat16)
    ids = [1, 2, 3, 4]
    with torch.no_grad():
        ref = model.float()(torch.tensor([ids])).logits[0, -1].numpy()
    src = HFSource(tmp_path)  # default: float16 weights
    assert src.get("h.0.q.w").dtype == np.float16
    out = Shard(src.cfg, shard_tensors(src, 0, src.cfg["n_layer"]), 0, src.cfg["n_layer"]).forward("j", np.array(ids, np.int32), 0)
    np.testing.assert_allclose(out, ref, atol=2e-2, rtol=2e-2)


def test_tokenizer_chat_template_and_eos(tmp_path):
    tokenizers = pytest.importorskip("tokenizers")
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

    from edgemesh.tokenizer import HFTokenizer

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(["hello world, how are you today?"] * 50,
                            trainers.BpeTrainer(vocab_size=300, special_tokens=["<|im_start|>", "<|im_end|>"],
                                                initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({
        "eos_token": "<|im_end|>",
        "chat_template": "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
                         "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"}))
    t = HFTokenizer(tmp_path)
    assert t.eos_ids == {tok.token_to_id("<|im_end|>")}
    ids = t.encode("hello", chat=True)
    assert ids[0] == tok.token_to_id("<|im_start|>")
    assert t.decode(t.encode("hello world")) == "hello world"
