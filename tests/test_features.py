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

from omnicompute import limits as L
from omnicompute.crypto import contribute_key, is_contribute_key
from omnicompute.limits import LimitStore
from omnicompute.model import Shard, random_model
from omnicompute.node import MeshNode, parse_join_url
from omnicompute.worker import Worker
from test_mesh import KEY, collect, model_file, reference, start_mesh  # noqa: F401


def _gpu(bits=16):
    from omnicompute.gpu import GPU, open_gpu

    gpu = open_gpu(allow_software=True)
    if isinstance(gpu, GPU):
        gpu.bits = bits  # 16: exact enough to compare tokens with the CPU
    return gpu


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
        assert friend.path_to("owner") == "relay"  # helpers never get anyone's IP address
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


def test_8bit_gpu_weights_stay_close_to_the_model():
    gpu = _gpu(bits=8)
    if gpu is None or gpu.backend != "webgpu":
        pytest.skip("no WebGPU adapter")
    cfg, w = random_model(n_layer=3, arch="llama", n_kv_head=2, tie=True)
    cpu = Shard(cfg, dict(w), 0, 3)
    g = Shard(cfg, dict(w), 0, 3)
    g.use_gpu(gpu)
    ids = np.array([3, 1, 4, 1, 5, 9], np.int32)
    a, b = g.forward("j", ids, 0), cpu.forward("j", ids, 0)
    assert np.abs(a - b).max() < 0.05 * np.abs(b).max()
    assert (a.argmax(-1) == b.argmax(-1)).mean() >= 0.8


def test_cuda_backend_through_torch(monkeypatch):
    """The CUDA code path, run by CPU torch (there is no NVIDIA card here)."""
    pytest.importorskip("torch")
    from omnicompute.gpu import CudaGPU, open_gpu

    monkeypatch.setenv("OMNICOMPUTE_GPU", "torch-cpu")
    gpu = open_gpu()
    assert isinstance(gpu, CudaGPU)
    for kw in [dict(arch="gpt2"), dict(arch="llama", n_kv_head=2, qkv_bias=True, tie=True)]:
        cfg, w = random_model(n_layer=2, **kw)
        w16 = {k: v.astype(np.float16).astype(np.float32) for k, v in w.items()}
        cpu = Shard(cfg, dict(w16), 0, 2)
        gpu.clear()
        g = Shard(cfg, dict(w16), 0, 2)
        g.use_gpu(gpu)
        ids = np.array([3, 1, 4], np.int32)
        np.testing.assert_allclose(g.forward("j", ids, 0), cpu.forward("j", ids, 0), atol=1e-4)
    assert gpu.benchmark() > 0


async def test_mesh_with_a_gpu_worker(model_file, monkeypatch):
    gpu = _gpu()
    if gpu is None:
        pytest.skip("no WebGPU adapter")
    monkeypatch.setenv("OMNICOMPUTE_GPU", "force")
    monkeypatch.setenv("OMNICOMPUTE_WEIGHTS", "16")
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
    from omnicompute import gpu as G

    gpu = _gpu()
    if gpu is None:
        pytest.skip("no WebGPU adapter")
    monkeypatch.setenv("OMNICOMPUTE_GPU", "force")

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
            assert f'HUB="{base}"' in sh and f'KEY="{ck}"' in sh and "@ORIGIN@" not in sh and "@KEY@" not in sh.split("set -e")[1]
            async with s.get(base + "/install.ps1?c=" + ck) as r:
                assert f'$Key = "{ck}"' in await r.text()
            async with s.get(base + "/install.sh?c=wrong") as r:
                assert r.status == 401
            async with s.get(base + "/download/omnicompute.zip?c=" + ck) as r:
                z = zipfile.ZipFile(io.BytesIO(await r.read()))
            names = z.namelist()
            assert "omnicompute-src/pyproject.toml" in names and "omnicompute-src/omnicompute/cli.py" in names
            assert "omnicompute-src/omnicompute/web/contribute.html" in names
            assert "omnicompute-src/omnicompute/installers/install.sh" in names
            async with s.get(base + "/contribute") as r:
                assert "Lend this device" in await r.text()
    finally:
        for n in nodes:
            await n.close()
        await hub.close()


def test_package_zip_installs_with_pip(tmp_path):
    from omnicompute.hub import _package_zip

    zipfile.ZipFile(io.BytesIO(_package_zip())).extractall(tmp_path)
    target = tmp_path / "site"
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", "--no-build-isolation",
                    "--target", str(target), str(tmp_path / "omnicompute-src")], check=True)
    assert (target / "omnicompute" / "web" / "worker.js").exists()
    assert (target / "omnicompute" / "installers" / "install.ps1").exists()


async def test_background_contributor_starts_and_stops(model_file, tmp_path, monkeypatch):
    if not sys.platform.startswith("linux"):
        pytest.skip("uses the Linux autostart path")
    path, cfg, w = model_file
    hub, nodes = await start_mesh(path, [])
    home, cfgdir = tmp_path / "home", tmp_path / "config"
    env = {**os.environ, "OMNICOMPUTE_HOME": str(home), "XDG_CONFIG_HOME": str(cfgdir), "OMNICOMPUTE_SCORE": "1",
           "OMNICOMPUTE_GPU": "0", "PYTHONPATH": str(Path(__file__).parent.parent)}
    link = f"http://127.0.0.1:{hub.port}/contribute?c={hub.contribute_key}"
    try:
        out = subprocess.run([sys.executable, "-m", "omnicompute", "contribute", link, "--name", "bg-box"],
                             env=env, capture_output=True, text=True, timeout=30).stdout
        assert "in the background as 'bg-box'" in out
        for _ in range(400):  # a cold Python start plus imports can take a while on a busy machine
            if "bg-box" in hub.clients:
                break
            await asyncio.sleep(0.1)
        assert hub.clients["bg-box"].contributor
        desktop = (cfgdir / "autostart" / "omnicompute.desktop").read_text(encoding="utf-8")
        assert "NoDisplay=true" in desktop and (home / "contribute.sh").exists()
        subprocess.run([sys.executable, "-m", "omnicompute", "contribute", "--stop"], env=env, check=True, timeout=30)
        for _ in range(300):
            if "bg-box" not in hub.clients:
                break
            await asyncio.sleep(0.1)
        assert "bg-box" not in hub.clients and not (cfgdir / "autostart" / "omnicompute.desktop").exists()
        assert "joined" in (home / "contribute.log").read_text(encoding="utf-8")
    finally:
        for n in nodes:
            await n.close()
        await hub.close()


async def test_remove_device_uninstalls_it(model_file, tmp_path):
    """"Remove" on the main computer: the installer-made setup deletes itself and stays gone."""
    if not sys.platform.startswith("linux"):
        pytest.skip("uses the Linux autostart path")
    path, cfg, w = model_file
    hub, nodes = await start_mesh(path, [])
    install = tmp_path / "install"  # what install.sh makes: a folder with the marker and a venv
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", "--system-site-packages", str(install / "venv")], check=True)
    (install / "installed-by-edgemesh-installer").write_text("x")
    home, cfgdir = tmp_path / "home", tmp_path / "config"
    env = {**os.environ, "OMNICOMPUTE_HOME": str(home), "XDG_CONFIG_HOME": str(cfgdir), "OMNICOMPUTE_SCORE": "1",
           "OMNICOMPUTE_GPU": "0", "PYTHONPATH": str(Path(__file__).parent.parent)}
    link = f"http://127.0.0.1:{hub.port}/contribute?c={hub.contribute_key}"
    try:
        subprocess.run([str(install / "venv" / "bin" / "python"), "-m", "omnicompute", "contribute", link, "--name", "lab-pc"],
                       env=env, check=True, capture_output=True, timeout=30)
        for _ in range(400):
            if "lab-pc" in hub.clients:
                break
            await asyncio.sleep(0.1)
        assert (cfgdir / "autostart" / "omnicompute.desktop").exists()
        async with aiohttp.ClientSession() as s:
            async with s.post(f"http://127.0.0.1:{hub.port}/api/devices/lab-pc/remove", headers={"X-Mesh-Key": KEY}) as r:
                res = await r.json()
        assert res["ok"] and str(install) in res["removed"]
        for _ in range(100):
            if not install.exists() and not home.exists():
                break
            await asyncio.sleep(0.1)
        assert not install.exists() and not home.exists()  # uninstalled
        assert not (cfgdir / "autostart" / "omnicompute.desktop").exists()  # won't start with the computer
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

    from omnicompute.cli import _start_on_free_port
    from omnicompute.hub import Hub

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
    from omnicompute import cli

    monkeypatch.setattr(cli, "HOME", tmp_path)
    first = cli._mesh_key(None)
    assert cli._mesh_key(None) == first  # same key after a restart
    assert cli._mesh_key("my-own-key-1", save=True) == "my-own-key-1"
    assert cli._mesh_key(None) == "my-own-key-1"  # yours from then on
    asyncio.run(cli.cmd_links(argparse.Namespace(key=None)))
    out = capsys.readouterr().out
    assert "?k=my-own-key-1" in out and "contribute?c=c" in out and 'OMNICOMPUTE_CODE="c' in out and "go.html#c=c" in out


async def test_termux_host_script_needs_the_full_key(model_file):
    import aiohttp

    path, cfg, w = model_file
    hub, nodes = await start_mesh(path, [])
    base = f"http://127.0.0.1:{hub.port}"
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(f"{base}/termux.sh?k={hub.contribute_key}") as r:
                assert r.status == 401  # a contribute key can't take over hosting...
            async with s.get(f"{base}/termux.sh?c={hub.contribute_key}") as r:
                worker = await r.text()  # ...but can make the phone a native worker
            async with s.get(f"{base}/termux.sh?k={KEY}") as r:
                text = await r.text()
        assert f'KEY="{KEY}"' in text and f'HUB="{base}"' in text and 'MODE="host"' in text
        assert f'KEY="{hub.contribute_key}"' in worker and 'MODE="contribute"' in worker and KEY not in worker
    finally:
        for x in nodes:
            await x.close()
        await hub.close()


def test_permanent_link_pieces_and_github_installers():
    """The beacon script is the same on the web page and the GitHub Pages permanent link, and the
    installers run straight from GitHub with just a code (no main device on the same network)."""
    root = Path(__file__).resolve().parent.parent
    assert (root / "docs" / "beacon.js").read_text(encoding="utf-8") == (root / "omnicompute" / "web" / "beacon.js").read_text(encoding="utf-8")
    for name in ("install.sh", "termux-host.sh"):
        text = (root / "omnicompute" / "installers" / name).read_text(encoding="utf-8")
        assert "OMNICOMPUTE_CODE" in text and "archive/refs/heads/main.zip" in text
    ps = (root / "omnicompute" / "installers" / "install.ps1").read_text(encoding="utf-8")
    assert "OMNICOMPUTE_CODE" in ps and "archive/refs/heads/main.zip" in ps and "--cores" in ps
    assert "--cores" in (root / "omnicompute" / "installers" / "install.sh").read_text(encoding="utf-8")


def test_cores_flag_is_saved_and_scales_the_score(tmp_path):
    from omnicompute import limits as L

    store = L.LimitStore(tmp_path / "l.json", {"cores": 4})
    assert store.limits["cores"] == 4
    assert L.LimitStore(tmp_path / "l.json").limits["cores"] == 4  # kept after a restart


def test_install_sh_from_github_downloads_the_repo(tmp_path):
    import pytest

    if sys.platform == "win32":
        pytest.skip("the Mac/Linux installer")
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "curl").write_text(f'#!/bin/sh\necho "$@" > "{tmp_path}/curl-args"\nexit 1\n')
    (bin_ / "curl").chmod(0o755)
    script = Path(__file__).resolve().parent.parent / "omnicompute" / "installers" / "install.sh"
    env = {**os.environ, "PATH": f"{bin_}:{os.environ['PATH']}", "HOME": str(tmp_path), "OMNICOMPUTE_YES": "1",
           "OMNICOMPUTE_CODE": "c" + "ab" * 12, "OMNICOMPUTE_REPO": "someone/fork"}
    out = subprocess.run(["sh", str(script)], env=env, capture_output=True, text=True, timeout=60)
    assert "Downloading Omnicompute from GitHub (someone/fork)" in out.stdout
    assert "https://github.com/someone/fork/archive/refs/heads/main.zip" in (tmp_path / "curl-args").read_text(encoding="utf-8")


def test_android_entry_point_hosts_and_stops(tmp_path, monkeypatch):
    """What the Android app's service calls: host on a thread, find the key, stop again."""
    import socket
    import sys as _sys
    import time
    import urllib.request

    from omnicompute import android

    monkeypatch.setenv("OMNICOMPUTE_BEACON", "off")
    monkeypatch.setattr(_sys, "stdout", _sys.stdout)
    monkeypatch.setattr(_sys, "stderr", _sys.stderr)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    import omnicompute.cli as cli

    monkeypatch.setattr(cli, "HOME", tmp_path / ".omnicompute")
    android.start(str(tmp_path), port=port, public=False)
    try:
        for _ in range(100):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/hello", timeout=1) as r:
                    assert json.load(r)["omnicompute"]
                break
            except OSError:
                time.sleep(0.1)
        else:
            raise AssertionError((tmp_path / ".omnicompute" / "host.log").read_text(encoding="utf-8"))
        assert android.running() and android.mesh_key(str(tmp_path))
        assert "go.html#c=c" in android.permanent_link(str(tmp_path))
    finally:
        android.stop()
    assert not android.running()


def test_gpu_limit_and_changing_limits_from_the_command_line(tmp_path, monkeypatch, capsys):
    from omnicompute import cli
    from omnicompute import limits as L

    lim = L.validate({"cpu": 40, "gpu_limit": 90})
    assert L.effective_for(lim, gpu=True) == (90, "GPU limit") and L.effective_for(lim, gpu=False)[0] == 40
    night = L.validate({"cpu": 40, "gpu_limit": 90, "schedule": [{"from": "00:00", "to": "23:59", "cpu": 0}]})
    assert L.effective_for(night, gpu=True)[0] == 0  # a pause window pauses the GPU too

    monkeypatch.setattr(cli, "HOME", tmp_path)
    running = L.LimitStore(tmp_path / "limits-desk.json", {"cpu": 80})
    cli.main(["limits", "--cpu", "30", "--gpu", "70", "--cores", "4"])
    assert "CPU 30%, GPU 70%, cores 4" in capsys.readouterr().out
    assert running.reload_if_changed() and running.limits["cpu"] == 30 and running.limits["gpu_limit"] == 70
    assert not running.reload_if_changed()
    cli.main(["limits", "--gpu", "same", "--cores", "all"])
    running.reload_if_changed()
    assert running.limits["gpu_limit"] is None and running.limits["cores"] is None


def test_browser_and_native_gpu_shaders_are_the_same():
    from omnicompute import gpu

    js = Path(__file__).resolve().parent.parent / "omnicompute" / "web" / "gpu.js"
    out = subprocess.run(["node", "-e", f"const g = require({str(js)!r}); console.log(JSON.stringify("
                          "[1, 4].map((t) => [g.shaderF16(t), g.shaderQ8(t)])))"], capture_output=True, text=True)
    if out.returncode:
        import pytest
        pytest.skip("node is not installed")
    assert json.loads(out.stdout) == [[gpu._shader_f16(t), gpu._shader_q8(t)] for t in (1, 4)]


def test_cpu_and_gpu_split_each_matrix(monkeypatch):
    """With both engines on, the CPU computes its share of each matrix's columns alongside the GPU."""
    pytest.importorskip("torch")
    from types import SimpleNamespace

    from omnicompute import limits as L
    from omnicompute.gpu import open_gpu
    from omnicompute.worker import Worker

    monkeypatch.setenv("OMNICOMPUTE_GPU", "torch-cpu")
    gpu = open_gpu()
    for kw in [dict(arch="gpt2"), dict(arch="llama", n_kv_head=2, tie=True)]:
        cfg, w = random_model(n_layer=2, n_embd=256, **kw)
        w16 = {k: v.astype(np.float16).astype(np.float32) for k, v in w.items()}
        cpu = Shard(cfg, dict(w16), 0, 2)
        gpu.clear()
        g = Shard(cfg, dict(w16), 0, 2)
        g.use_gpu(gpu, cpu_share=0.4)
        assert g.cpu_part and all(m.shape[1] % 32 == 0 for m in g.cpu_part.values())
        ids = np.array([3, 1, 4], np.int32)
        np.testing.assert_allclose(g.forward("j", ids, 0), cpu.forward("j", ids, 0), atol=1e-4)
        np.testing.assert_allclose(g.forward("j", np.array([5], np.int32), 3),
                                   cpu.forward("j", np.array([5], np.int32), 3), atol=1e-4)

    monkeypatch.delenv("OMNICOMPUTE_GPU")
    info = {"cpu_score": 30.0, "cores_total": 8, "gpu": {"score": 70.0}}
    wk = Worker(gpu=gpu)
    wk.node = SimpleNamespace(info=info)
    store = SimpleNamespace(limits=L.validate({"cpu": 100}))
    wk.limits = store
    assert wk.engines() == ("both", 0.3) and wk.limit_info()["uses"] == "gpu+cpu"
    store.limits = L.validate({"cpu": 100, "use_cpu": False})
    assert wk.engines() == ("gpu", 0.0)
    store.limits = L.validate({"cpu": 100, "gpu": False})
    assert wk.engines() == ("cpu", 1.0)
    store.limits = L.validate({"cpu": 100, "gpu": False, "use_cpu": False})  # never both off
    assert store.limits["use_cpu"] and wk.engines() == ("cpu", 1.0)
    store.limits = L.validate({"cpu": 10, "gpu_limit": 100})  # a small CPU share isn't worth splitting
    assert wk.engines() == ("gpu", 0.0)


def test_chat_history_keeps_searches_and_reopens_chats(tmp_path):
    from omnicompute.chats import ChatHistory

    h = ChatHistory(tmp_path / "chats.json")
    h.keep("a", [{"role": "user", "content": "How do tides work?"}, {"role": "assistant", "content": "The moon", "live": True}])
    h.keep("b", [{"role": "user", "content": "Write a haiku"}])
    h.keep("c", [])  # empty chats are not kept
    h.rename("b", "Poems")
    again = ChatHistory(tmp_path / "chats.json")
    assert {c["title"] for c in again.listing()} == {"How do tides work?", "Poems"}
    assert [c["id"] for c in again.listing("moon")] == ["a"]
    assert "live" not in again.messages("a")[1]
    again.keep("b", [{"role": "user", "content": "Write a haiku"}, {"role": "assistant", "content": "..."}])
    assert again.chats["b"]["title"] == "Poems"  # a renamed chat keeps its name
    again.delete("a")
    assert [c["id"] for c in ChatHistory(tmp_path / "chats.json").listing()] == ["b"]


def test_helpers_never_get_the_first_or_last_layers():
    from omnicompute.model import plan_layers

    ws = [{"node_id": "helper-fast", "score": 100, "contributor": True}, {"node_id": "laptop", "score": 5},
          {"node_id": "helper", "score": 50, "contributor": True}, {"node_id": "phone", "score": 2}]
    plan = plan_layers(24, ws)
    assert {plan[0][0], plan[-1][0]} == {"laptop", "phone"} and len(plan) == 4
    assert plan_layers(24, ws[:3])[0][0] == "laptop"  # one own device: it takes the prompt's end


async def test_android_app_helps_a_computers_mesh_from_its_contribute_link(tmp_path, monkeypatch):
    """The app's "Help my main computer" choice: a native worker joins with the pasted link."""
    import sys as _sys

    from omnicompute import android
    from omnicompute.hub import Hub
    from omnicompute.model import random_model, save_model

    monkeypatch.setenv("OMNICOMPUTE_BEACON", "off")
    monkeypatch.setattr(_sys, "stdout", _sys.stdout)
    monkeypatch.setattr(_sys, "stderr", _sys.stderr)
    import omnicompute.cli as cli

    monkeypatch.setattr(cli, "HOME", tmp_path / ".omnicompute")
    path = str(tmp_path / "m.emm")
    save_model(path, *random_model(n_layer=2, n_embd=32, n_head=2, n_ctx=64))
    hub = Hub("android-help-key", "127.0.0.1", 0, path)
    await hub.start()
    link = f"http://127.0.0.1:{hub.port}/contribute?c={hub.contribute_key}"
    assert android.check_link(link) == "" and android.check_link("hello")
    android.join(str(tmp_path), link, name="chromebook")
    try:
        for _ in range(100):
            if any(w["node_id"] == "chromebook" for w in hub.workers()):
                break
            await asyncio.sleep(0.1)
        else:
            raise AssertionError(android.status(str(tmp_path)))
        assert hub.clients["chromebook"].contributor
        assert not (tmp_path / ".omnicompute" / "mesh.key").exists()  # helping makes no mesh of its own
    finally:
        await asyncio.get_running_loop().run_in_executor(None, android.stop)
        await hub.close()
