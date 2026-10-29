import numpy as np

from omnicompute.model import DictSource, Shard, plan_layers, random_model, shard_tensors
from omnicompute.protocol import decode_tensors, encode_tensors


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
    assert plan == [("slow", 0, 3), ("fast", 3, 12)]  # fastest last: it also computes the output head


def test_plan_respects_memory():
    cfg, _ = random_model(n_layer=8, n_embd=128)
    # "fast" has room for ~3 layers, one of which goes to the output head it holds as last device
    plan = plan_layers(8, [{"node_id": "fast", "score": 100, "mem_mb": 3}, {"node_id": "big", "score": 1}], cfg)
    assert plan == [("big", 0, 6), ("fast", 6, 8)]
    # too small to even hold the embeddings: the big device goes first instead
    plan = plan_layers(8, [{"node_id": "tiny", "score": 100, "mem_mb": 1}, {"node_id": "big", "score": 1}], cfg)
    assert plan[0][0] == "big"


def test_best_plan_skips_devices_that_only_add_latency():
    from omnicompute.estimate import best_plan

    cfg, _ = random_model(n_layer=8, n_embd=128)
    pc = {"node_id": "pc", "score": 50, "mem_mb": 8000, "kind": "native", "rtt_ms": 1}
    phone = {"node_id": "phone", "score": 0.5, "mem_mb": 1000, "kind": "browser", "rtt_ms": 60}
    plan, est = best_plan(cfg, [pc, phone])
    assert [p[0] for p in plan] == ["pc"] and est["tok_per_s"] > 0
    plan, _ = best_plan(cfg, [pc, {**phone, "limits": {"always": True}}])
    assert {p[0] for p in plan} == {"pc", "phone"}
    plan, _ = best_plan(cfg, [{**pc, "mem_mb": 1}, phone])  # too big for the pc alone: the phone is needed
    assert "phone" in {p[0] for p in plan}


def test_ws_parts_roundtrip():
    import os

    from omnicompute.wsparts import Joiner, split

    data = os.urandom(3 * (1 << 20) + 5)
    parts = split(data)
    assert len(parts) == 4
    j = Joiner()
    out = [j.feed(p) for p in reversed(parts)]
    assert out[:3] == [None] * 3 and out[3] == data
    assert j.feed(b"small") == b"small"
