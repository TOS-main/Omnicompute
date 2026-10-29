"""The fast GGUF engine (llama.cpp's llama-server), with a stand-in server."""

import json
import os
import stat
import sys
from pathlib import Path

import pytest

posix_only = pytest.mark.skipif(os.name == "nt", reason="the stand-in server is started through a shell script")


@pytest.fixture
def fake_server(tmp_path, monkeypatch):
    exe = tmp_path / "llama-server"
    exe.write_text(f"#!/bin/sh\nexec {sys.executable} {Path(__file__).parent / 'fake_llama_server.py'} \"$@\"\n")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("OMNICOMPUTE_LLAMA_SERVER", str(exe))
    monkeypatch.setenv("OMNICOMPUTE_LLAMA", "on")
    return exe


def _gguf(tmp_path):
    from gguf import GGMLQuantizationType as Q
    from test_gguf import write_gguf

    from omnicompute.model import random_model

    cfg, w = random_model(n_layer=2, n_embd=64, n_head=4, vocab=320, arch="llama", tie=True)
    path = tmp_path / "tiny.gguf"
    write_gguf(path, cfg, w, "llama", Q.F32)
    return path


@posix_only
async def test_gguf_runs_on_llama_cpp_with_lm_studio_settings(tmp_path, fake_server):
    from omnicompute.gguf_source import GGUFSource
    from omnicompute.hub import Hub

    path = _gguf(tmp_path)
    hub = Hub("llama-key", "127.0.0.1", 0, GGUFSource(path), models_dir=str(tmp_path / "models"))
    await hub.start()
    try:
        o = hub.orchestrator
        assert o.engine is not None
        events = [ev async for ev in o.generate([{"role": "user", "content": "2+2?"}], 100, chat=True)]
        kinds = [e["type"] for e in events]
        assert "status" in kinds and kinds[-1] == "done", events
        done = events[-1]
        assert done["text"] == "2 + 2 = 4" and done["thinking"] == "two plus two is four"
        assert done["stats"]["engine"] == "llama.cpp" and done["stats"]["tok_per_s"] == 64.0
        args = json.loads(Path(str(path) + ".args.json").read_text())
        assert args[args.index("-ngl") + 1] == "999" and "--jinja" in args and "-fa" in args  # all layers on the GPU
        # CPU only, 3 threads: restarts with those settings
        o.set_engine({"gpu_layers": 0, "threads": 3})
        events = [ev async for ev in o.generate("hi", 10)]
        assert events[-1]["text"] == "hello world"
        args = json.loads(Path(str(path) + ".args.json").read_text())
        assert args[args.index("-ngl") + 1] == "0" and args[args.index("-t") + 1] == "3"
        # the mesh instead
        o.set_engine({"engine": "mesh"})
        assert o.engine is None
    finally:
        await hub.close()


async def test_loading_another_model_stops_the_answer_being_written(tmp_path, monkeypatch):
    """A slow mesh answer must not hold up a switch until it finishes."""
    import asyncio

    from test_mesh import KEY, start_mesh

    from omnicompute.model import DictSource, random_model

    cfg, w = random_model(n_layer=2, n_embd=32, n_head=2, n_ctx=4096)
    hub, nodes = await start_mesh(DictSource(cfg, w), [("pc", False, 1000)])
    try:
        o = hub.orchestrator
        gen = o.generate("hello", 4000)
        got = []

        async def run():
            async for ev in gen:
                got.append(ev)

        task = asyncio.ensure_future(run())
        while not any(e["type"] == "token" for e in got):
            await asyncio.sleep(0.01)
        cfg2, w2 = random_model(n_layer=2, n_embd=32, n_head=2, n_ctx=64)
        await asyncio.wait_for(hub._switch(DictSource(cfg2, w2)), 10)
        await asyncio.wait_for(task, 10)
        assert got[-1]["type"] == "done" and got[-1]["stats"]["stopped"] == "switched"
        assert hub.orchestrator.cfg is cfg2
    finally:
        for n in nodes:
            await n.close()
        await hub.close()


@pytest.mark.skipif(not os.environ.get("OMNICOMPUTE_TEST_REAL_LLAMA"), reason="downloads llama.cpp and a model (CI)")
async def test_real_llama_cpp_download_and_run(tmp_path, monkeypatch):
    import urllib.request

    from omnicompute.gguf_source import GGUFSource
    from omnicompute.llamacpp import Engine

    monkeypatch.delenv("OMNICOMPUTE_LLAMA_SERVER", raising=False)
    model = tmp_path / "stories260K.gguf"
    urllib.request.urlretrieve("https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories260K.gguf", model)
    eng = Engine(tmp_path, model, GGUFSource(model).cfg)
    eng.start({"context": 256, "gpu_layers": "auto", "flash_attn": True})
    assert await eng.ready(), (eng.error, eng._tail())
    out, timings = "", {}
    async for kind, piece in eng.generate("Once upon a time", 24, 0.0, False):
        if kind == "timings":
            timings = piece
        else:
            out += piece
    eng.stop()
    assert out.strip() and timings.get("predicted_per_second"), (out, timings)
