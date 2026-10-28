import numpy as np

from edgemesh.model import DictSource, Shard, plan_layers, random_model, shard_tensors
from edgemesh.protocol import decode_tensors, encode_tensors


def full_logits(cfg, w, ids):
    return Shard(cfg, w, 0, cfg["n_layer"]).forward("j", np.array(ids, np.int32), 0)


def test_tensor_roundtrip():
    t = {"a": np.arange(7, dtype=np.float32), "b": np.arange(6, dtype=np.int32).reshape(2, 3)}
    back = decode_tensors(encode_tensors(t))
    assert all(np.array_equal(t[k], back[k]) and t[k].dtype == back[k].dtype for k in t)


import pytest


@pytest.mark.parametrize("kw", [{}, {"arch": "llama", "n_kv_head": 2, "qkv_bias": True},
                                {"arch": "llama", "qk_norm": True, "tie": False}])
def test_split_matches_full_model_with_kv_cache(kw):
    cfg, w = random_model(n_layer=6, n_embd=32, n_head=4, n_ctx=64, **kw)
    ids = [5, 9, 200, 3]
    shards = [Shard(cfg, shard_tensors(DictSource(cfg, w), s, e), s, e) for s, e in [(0, 2), (2, 5), (5, 6)]]

    def split(x, pos):
        for sh in shards:
            x = sh.forward("job", x, pos)
        return x

    # prompt in one go, then one token at a time through the cache
    np.testing.assert_allclose(split(np.array(ids, np.int32), 0), full_logits(cfg, w, ids), rtol=1e-4, atol=1e-4)
    seq = list(ids)
    for nxt in [7, 8]:
        out = split(np.array([nxt], np.int32), len(seq))
        seq.append(nxt)
        np.testing.assert_allclose(out, full_logits(cfg, w, seq), rtol=1e-4, atol=1e-4)


def test_plan_is_contiguous_and_weighted():
    plan = plan_layers(12, [{"node_id": "fast", "score": 30}, {"node_id": "slow", "score": 10}])
    assert plan == [("fast", 0, 9), ("slow", 9, 12)]


def test_plan_respects_memory():
    cfg, _ = random_model(n_layer=8, n_embd=128)
    # "fast" has room for ~3 layers, one of which goes to the embeddings it holds as first device
    plan = plan_layers(8, [{"node_id": "fast", "score": 100, "mem_mb": 3}, {"node_id": "big", "score": 1}], cfg)
    assert plan == [("fast", 0, 2), ("big", 2, 8)]
    # too small to even hold the embeddings: the big device goes first instead
    plan = plan_layers(8, [{"node_id": "tiny", "score": 100, "mem_mb": 1}, {"node_id": "big", "score": 1}], cfg)
    assert plan[0][0] == "big"
