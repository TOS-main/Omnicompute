"""Contribute-only key, GPU backend, power limits, shared chat, model management, installers."""

import argparse
import asyncio
import io
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

import aiohttp
import numpy as np
import pytest

from edgemesh import limits as L
from edgemesh.crypto import contribute_key, is_contribute_key
from edgemesh.limits import LimitStore
from edgemesh.model import Shard, random_model
from edgemesh.node import MeshNode, parse_join_url
from edgemesh.worker import Worker
from test_mesh import KEY, collect, model_file, reference, start_mesh  # noqa: F401


def _gpu():
    from edgemesh.gpu import open_gpu

    return open_gpu(allow_software=True)


# ---------------------------------------------------------------- contribute-only key


def test_contribute_key_shape():
    ck = contribute_key("some mesh key")
    assert is_contribute_key(ck) and contribute_key(ck) == ck and not is_contribute_key("some mesh key")
    assert parse_join_url(f"http://h:1/contribute?c={ck}") == ("http://h:1", ck)


async def test_contributor_joins_computes_but_gets_no_admin(model_file):
    path, cfg, w = model_file
    hub, nodes = await start_mesh(path, [("owner", False, 2)])
    ck = hub.contribute_key
    friend = MeshNode("friend", ck, f"http://127.0.0.1:{hub.port}",
                      {"role": "worker", "kind": "native", "score": 3, "mem_mb": 1024})
    Worker(LimitStore(Path(tempfile.mkdtemp()) / "l.json", {"cpu": 100, "always": True})).attach(friend)
    await friend.start()
    try:
        assert hub.clients["friend"].contributor and not hub.clients["owner"].contributor
        events, ids = await collect(hub, "hello", 8)
        assert [p[0] for p in events[0]["plan"]] == ["owner", "friend"]
        assert ids == reference(cfg, w, "hello", 8)
        assert friend.path_to("owner") == "direct"  # contributors still hole-punch with the others
        # each device's part in the answer
        contrib = events[-1]["stats"]["contribution"]
        assert {c["node_id"] for c in contrib} == {"owner", "friend"} and abs(sum(c["share"] for c in contrib) - 100) < 1

        # the contribute key opens nothing else
        async with aiohttp.ClientSession() as s:
            for path_ in ("/api/status", "/api/chat", "/api/models"):
                async with s.get(f"http://127.0.0.1:{hub.port}{path_}", headers={"X-Mesh-Key": ck}) as r:
                    assert r.status == 401
            async with s.get(f"http://127.0.0.1:{hub.port}/api/status", headers={"X-Mesh-Key": KEY}) as r:
                st = await r.json()
                assert st["contribute_key"] == ck
                assert next(d for d in st["devices"] if d["node_id"] == "friend")["helped"]["tokens"] == 8

        # ...and other devices ignore control messages that don't come from the hub
        with pytest.raises(RuntimeError, match="only accepted from the hub"):
            await friend.request("owner", {"op": "set_limits", "limits": {"cpu": 0}}, timeout=5)
        # a contributor can't take over a device's name
        impostor = MeshNode("owner", ck, f"http://127.0.0.1:{hub.port}", {"role": "worker"})
        with pytest.raises(PermissionError):
            await impostor.start()
    finally:
        await friend.close()
        for n in nodes:
            await n.close()
        await hub.close()


# ---------------------------------------------------------------- GPU


def test_gpu_shard_matches_cpu():
    gpu = _gpu()
    if gpu is None:
        pytest.skip("no WebGPU adapter (pip install wgpu; Linux needs mesa-vulkan-drivers)")
    gpu.max_bind = 64 * 1024  # force big matrices into several chunks
    for kw in [dict(arch="gpt2"), dict(arch="llama", n_kv_head=2, qkv_bias=True, qk_norm=True, tie=True),
               dict(arch="llama", n_kv_head=2, tie=False)]:
        cfg, w = random_model(n_layer=3, **kw)
        w16 = {k: v.astype(np.float16).astype(np.float32) for k, v in w.items()}
        cpu = Shard(cfg, dict(w16), 0, 3)
        gpu.clear()
        g = Shard(cfg, dict(w16), 0, 3)
        g.use_gpu(gpu)
        assert not any(k.endswith(".w") and v.ndim == 2 for k, v in g.w.items())  # the CPU copies are freed
        ids = np.array([3, 1, 4, 1], np.int32)
        np.testing.assert_allclose(g.forward("j", ids, 0), cpu.forward("j", ids, 0), atol=1e-4)
        np.testing.assert_allclose(g.forward("j", np.array([5], np.int32), 4),
                                   cpu.forward("j", np.array([5], np.int32), 4), atol=1e-4)


async def test_mesh_with_a_gpu_worker(model_file, monkeypatch):
    gpu = _gpu()
    if gpu is None:
        pytest.skip("no WebGPU adapter")
    monkeypatch.setenv("EDGEMESH_GPU", "force")
    path, cfg, w = model_file
    hub, nodes = await start_mesh(path, [("cpu-box", False, 2)])
    n = MeshNode("gpu-box", KEY, f"http://127.0.0.1:{hub.port}",
                 {"role": "worker", "kind": "native", "score": 3, "cpu_score": 1, "mem_mb": 1024,
                  "gpu": {"name": gpu.name, "score": 3}})
    worker = Worker(LimitStore(Path(tempfile.mkdtemp()) / "l.json", {"cpu": 100, "always": True}), gpu).attach(n)
    await n.start()
    try:
        assert hub.clients["gpu-box"].info["uses"] == "gpu"
        events, ids = await collect(hub, "gpu!", 8)
        assert "gpu-box" in [p[0] for p in events[0]["plan"]]
        assert ids == reference(cfg, w, "gpu!", 8)  # fp16 weights on the GPU, same tokens
        assert next(iter(worker.shards.values())).gpu is gpu
    finally:
        await n.close()
        for x in nodes:
            await x.close()
        await hub.close()


async def test_a_gpu_that_computes_wrong_falls_back_to_the_cpu(model_file, monkeypatch):
    from edgemesh import gpu as G

    gpu = _gpu()
    if gpu is None:
        pytest.skip("no WebGPU adapter")
    monkeypatch.setenv("EDGEMESH_GPU", "force")

    def bad_put(self, name, w):
        raise G.GPUMismatch(f"{name}: off by 1e3")

    monkeypatch.setattr(G.GPU, "put", bad_put)
    path, cfg, w = model_file
    hub, nodes = await start_mesh(path, [])
    n = MeshNode("bad-gpu", KEY, f"http://127.0.0.1:{hub.port}",
                 {"role": "worker", "kind": "native", "score": 3, "cpu_score": 1, "mem_mb": 1024,
                  "gpu": {"name": gpu.name, "score": 3}})
    worker = Worker(LimitStore(Path(tempfile.mkdtemp()) / "l.json", {"cpu": 100, "always": True}), gpu).attach(n)
    await n.start()
    try:
        events, ids = await collect(hub, "gpu?", 8)
        assert "bad-gpu" in [p[0] for p in events[0]["plan"]]
        assert ids == reference(cfg, w, "gpu?", 8)  # right answer from the CPU
        assert worker.gpu_broken and next(iter(worker.shards.values())).gpu is None
    finally:
        await n.close()
        for x in nodes:
            await x.close()
        await hub.close()


# ---------------------------------------------------------------- power limits


def test_battery_and_charging_rules():
    lim = L.validate({"cpu": 80, "cores": 6, "battery": {"cpu": 30, "cores": 2}, "charging": {"cpu": 100, "cores": None},
                      "schedule": [{"from": "09:00", "to": "17:00", "cpu": 50}]})
    noon, night = datetime(2026, 1, 1, 12), datetime(2026, 1, 1, 22)
    assert L.effective(lim, night, None) == (80, "default")
    assert L.effective(lim, night, "charging") == (100, "charging")
    assert L.effective(lim, noon, "charging") == (50, "schedule 09:00-17:00")  # time windows win over charging
    assert L.effective(lim, noon, "battery") == (30, "on battery")  # battery caps everything
    assert L.effective_cores(lim, "battery") == 2 and L.effective_cores(lim, "charging") == 6
    assert L.effective_cores(lim, None) == 6
    with pytest.raises(ValueError):
        L.validate({"battery": {"cpu": 130}})


# ---------------------------------------------------------------- hub features


async def test_shared_chat_models_capacity_and_installers(model_file, tmp_path):
    path, cfg, w = model_file
    hub, nodes = await start_mesh(path, [("a", False, 2)])
    hub.loader.cache = tmp_path / "models"
    good = hub.loader.cache / "org--tiny"
    good.mkdir(parents=True)
    for f in ("config.json", "tokenizer.json", "model.safetensors"):
        (good / f).write_bytes(b"x" * 100)
    (hub.loader.cache / "org--half").mkdir()
    (hub.loader.cache / "org--half" / "model.safetensors.part").write_bytes(b"x" * 10)
    base, auth = f"http://127.0.0.1:{hub.port}", {"X-Mesh-Key": KEY}
    try:
        async with aiohttp.ClientSession() as s:
            body = {"messages": [{"role": "user", "content": "hi"}], "max_new": 5, "record": {"user": "hi", "by": "phone"}}
            async with s.post(base + "/api/generate", json=body, headers=auth) as r:
                lines = [json.loads(x) async for x in r.content]
            async with s.get(base + "/api/chat", headers=auth) as r:
                chat = (await r.json())["messages"]
            assert chat[0] == {"role": "user", "content": "hi", "by": "phone"}
            assert chat[1]["content"] == lines[-1]["text"] and "live" not in chat[1] and chat[1]["stats"]["contribution"]
            async with s.post(base + "/api/chat/clear", headers=auth) as r:
                assert r.status == 200
            async with s.get(base + "/api/chat", headers=auth) as r:
                assert (await r.json())["messages"] == []

            async with s.get(base + "/api/models", headers=auth) as r:
                models = {m["repo"]: m for m in (await r.json())["models"]}
            assert models["org/tiny"]["complete"] and not models["org/half"]["complete"]
            async with s.delete(base + "/api/models?repo=org/tiny", headers=auth) as r:
                assert (await r.json())["freed"] == 300
            assert not good.exists()
            async with s.delete(base + "/api/models?repo=../../etc", headers=auth) as r:
                assert r.status == 404

            async with s.get(base + "/api/status", headers=auth) as r:
                cap = (await r.json())["capacity"]
            assert cap["devices"] == 1 and cap["max_params"] == 1024 * 2**20 // 4
            assert cap["params_at_1"] == min(cap["max_params"], int(2e9 / 2))

            ck = hub.contribute_key
            async with s.get(base + "/install.sh?c=" + ck) as r:
                sh = await r.text()
            assert f'HUB="{base}"' in sh and f'KEY="{ck}"' in sh and "@" not in sh.split("set -e")[1].split("PY=")[0]
            async with s.get(base + "/install.ps1?c=" + ck) as r:
                assert f'$Key = "{ck}"' in await r.text()
            async with s.get(base + "/install.sh?c=wrong") as r:
                assert r.status == 401
            async with s.get(base + "/download/edgemesh.zip?c=" + ck) as r:
                z = zipfile.ZipFile(io.BytesIO(await r.read()))
            names = z.namelist()
            assert "edgemesh-src/pyproject.toml" in names and "edgemesh-src/edgemesh/cli.py" in names
            assert "edgemesh-src/edgemesh/web/contribute.html" in names
            assert "edgemesh-src/edgemesh/installers/install.sh" in names
            async with s.get(base + "/contribute") as r:
                assert "Lend this device" in await r.text()
    finally:
        for n in nodes:
            await n.close()
        await hub.close()


def test_package_zip_installs_with_pip(tmp_path):
    from edgemesh.hub import _package_zip

    zipfile.ZipFile(io.BytesIO(_package_zip())).extractall(tmp_path)
    target = tmp_path / "site"
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", "--no-build-isolation",
                    "--target", str(target), str(tmp_path / "edgemesh-src")], check=True)
    assert (target / "edgemesh" / "web" / "worker.js").exists()
    assert (target / "edgemesh" / "installers" / "install.ps1").exists()


async def test_background_contributor_starts_and_stops(model_file, tmp_path, monkeypatch):
    if sys.platform == "win32":
        pytest.skip("uses the Linux autostart path")
    path, cfg, w = model_file
    hub, nodes = await start_mesh(path, [])
    home, cfgdir = tmp_path / "home", tmp_path / "config"
    env = {**os.environ, "EDGEMESH_HOME": str(home), "XDG_CONFIG_HOME": str(cfgdir), "EDGEMESH_SCORE": "1",
           "EDGEMESH_GPU": "0", "PYTHONPATH": str(Path(__file__).parent.parent)}
    link = f"http://127.0.0.1:{hub.port}/contribute?c={hub.contribute_key}"
    try:
        out = subprocess.run([sys.executable, "-m", "edgemesh", "contribute", link, "--name", "bg-box"],
                             env=env, capture_output=True, text=True, timeout=30).stdout
        assert "in the background as 'bg-box'" in out
        for _ in range(400):  # a cold Python start plus imports can take a while on a busy machine
            if "bg-box" in hub.clients:
                break
            await asyncio.sleep(0.1)
        assert hub.clients["bg-box"].contributor
        desktop = (cfgdir / "autostart" / "edgemesh.desktop").read_text()
        assert "NoDisplay=true" in desktop and (home / "contribute.sh").exists()
        subprocess.run([sys.executable, "-m", "edgemesh", "contribute", "--stop"], env=env, check=True, timeout=30)
        for _ in range(300):
            if "bg-box" not in hub.clients:
                break
            await asyncio.sleep(0.1)
        assert "bg-box" not in hub.clients and not (cfgdir / "autostart" / "edgemesh.desktop").exists()
        assert "joined" in (home / "contribute.log").read_text()
    finally:
        for n in nodes:
            await n.close()
        await hub.close()


async def test_remove_device_uninstalls_it(model_file, tmp_path):
    """"Remove" on the main computer: the installer-made setup deletes itself and stays gone."""
    if sys.platform == "win32":
        pytest.skip("uses the Linux autostart path")
    path, cfg, w = model_file
    hub, nodes = await start_mesh(path, [])
    install = tmp_path / "install"  # what install.sh makes: a folder with the marker and a venv
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", "--system-site-packages", str(install / "venv")], check=True)
    (install / "installed-by-edgemesh-installer").write_text("x")
    home, cfgdir = tmp_path / "home", tmp_path / "config"
    env = {**os.environ, "EDGEMESH_HOME": str(home), "XDG_CONFIG_HOME": str(cfgdir), "EDGEMESH_SCORE": "1",
           "EDGEMESH_GPU": "0", "PYTHONPATH": str(Path(__file__).parent.parent)}
    link = f"http://127.0.0.1:{hub.port}/contribute?c={hub.contribute_key}"
    try:
        subprocess.run([str(install / "venv" / "bin" / "python"), "-m", "edgemesh", "contribute", link, "--name", "lab-pc"],
                       env=env, check=True, capture_output=True, timeout=30)
        for _ in range(400):
            if "lab-pc" in hub.clients:
                break
            await asyncio.sleep(0.1)
        assert (cfgdir / "autostart" / "edgemesh.desktop").exists()
        async with aiohttp.ClientSession() as s:
            async with s.post(f"http://127.0.0.1:{hub.port}/api/devices/lab-pc/remove", headers={"X-Mesh-Key": KEY}) as r:
                res = await r.json()
        assert res["ok"] and str(install) in res["removed"]
        for _ in range(100):
            if not install.exists() and not home.exists():
                break
            await asyncio.sleep(0.1)
        assert not install.exists() and not home.exists()  # uninstalled
        assert not (cfgdir / "autostart" / "edgemesh.desktop").exists()  # won't start with the computer
        assert "lab-pc" not in hub.clients
        again = MeshNode("lab-pc", hub.contribute_key, f"http://127.0.0.1:{hub.port}", {"role": "worker"})
        with pytest.raises(PermissionError, match="removed"):
            await again.start()
    finally:
        for n in nodes:
            await n.close()
        await hub.close()


async def test_host_moves_to_a_free_port_or_reuses_a_running_hub(model_file, monkeypatch):
    import socket

    from edgemesh.cli import _start_on_free_port
    from edgemesh.hub import Hub

    path, cfg, w = model_file
    busy = socket.socket()
    busy.bind(("127.0.0.1", 0))
    busy.listen()
    port = busy.getsockname()[1]
    first = Hub(KEY, "127.0.0.1", port, path)
    opened = []
    monkeypatch.setattr("webbrowser.open", lambda url: opened.append(url))
    try:
        assert await _start_on_free_port(first, KEY, port)
        assert first.port == port + 1  # something else had the port
        second = Hub(KEY, "127.0.0.1", port, path)
        assert not await _start_on_free_port(second, KEY, port)  # same mesh already running: open it instead
        assert opened == [f"http://127.0.0.1:{port + 1}/?k={KEY}"]
    finally:
        busy.close()
        await first.close()


# ---------------------------------------------------------------- keys stay the same

def test_join_links_survive_copy_paste():
    assert parse_join_url('"http://10.0.0.5:8001/?k=abc_DEF-123".') == ("http://10.0.0.5:8001", "abc_DEF-123")
    assert parse_join_url("<10.0.0.5:8000/contribute?c=c0123>") == ("http://10.0.0.5:8000", "c0123")


def test_mesh_key_is_kept_and_can_be_set(tmp_path, monkeypatch, capsys):
    from edgemesh import cli

    monkeypatch.setattr(cli, "HOME", tmp_path)
    first = cli._mesh_key(None)
    assert cli._mesh_key(None) == first  # same key after a restart
    assert cli._mesh_key("my-own-key-1", save=True) == "my-own-key-1"
    assert cli._mesh_key(None) == "my-own-key-1"  # yours from then on
    asyncio.run(cli.cmd_links(argparse.Namespace(key=None)))
    out = capsys.readouterr().out
    assert "?k=my-own-key-1" in out and "contribute?c=c" in out and "install.ps1?c=c" in out
