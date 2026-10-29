import shutil
import subprocess

import numpy as np
import pytest

from omnicompute.model import Shard, random_model
from omnicompute.protocol import decode_tensors, encode_tensors, pack

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="needs node.js")

VARIANTS = {
    "gpt2": {},
    "llama-gqa-bias": {"arch": "llama", "n_kv_head": 2, "qkv_bias": True},
    "qwen3-untied": {"arch": "llama", "qk_norm": True, "tie": False},
    "llama3-rope": {"arch": "llama", "n_kv_head": 1},
}


@pytest.mark.parametrize("name", list(VARIANTS))
@pytest.mark.parametrize("half", [False, True])
def test_browser_shard_matches_numpy(tmp_path, name, half):
    cfg, w = random_model(n_layer=4, n_embd=32, n_head=4, n_ctx=64, **VARIANTS[name])
    if name == "llama3-rope":
        cfg["rope_theta"] = 500000.0
        cfg["rope_scaling"] = {"rope_type": "llama3", "factor": 8.0, "original_max_position_embeddings": 16}
    if half:
        w = {k: v.astype(np.float16) if v.ndim == 2 else v for k, v in w.items()}
    ids = np.array([1, 50, 7, 99, 3], np.int32)
    req = tmp_path / "req.bin"
    req.write_bytes(pack({"cfg": cfg, "splits": [[0, 1], [1, 3], [3, 4]], "next": 42},
                         encode_tensors({**w, "__ids__": ids})))
    out = decode_tensors(subprocess.run(["node", "tests/js_shard_check.js", str(req)],
                                        check=True, capture_output=True).stdout)
    full = Shard(cfg, w, 0, 4)
    np.testing.assert_allclose(out["prompt"], full.forward("j", ids, 0), atol=3e-4, rtol=1e-3)
    np.testing.assert_allclose(out["step"], full.forward("j", np.array([42], np.int32), len(ids)), atol=3e-4, rtol=1e-3)
