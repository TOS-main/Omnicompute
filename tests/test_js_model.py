import shutil
import subprocess

import numpy as np
import pytest

from edgemesh.model import Shard, random_model
from edgemesh.protocol import decode_tensors, encode_tensors, pack

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="needs node.js")


def test_browser_shard_matches_numpy(tmp_path):
    cfg, w = random_model(n_layer=4, n_embd=32, n_head=4, n_ctx=64)
    ids = np.array([1, 50, 7, 99, 3], np.int32)
    req = tmp_path / "req.bin"
    req.write_bytes(pack({"cfg": cfg, "splits": [[0, 1], [1, 3], [3, 4]], "next": 42},
                         encode_tensors({**w, "__ids__": ids})))
    out = decode_tensors(subprocess.run(["node", "tests/js_shard_check.js", str(req)],
                                        check=True, capture_output=True).stdout)
    full = Shard(cfg, w, 0, 4)
    np.testing.assert_allclose(out["prompt"], full.forward("j", ids, 0), atol=2e-4, rtol=1e-3)
    np.testing.assert_allclose(out["step"], full.forward("j", np.array([42], np.int32), len(ids)), atol=2e-4, rtol=1e-3)
