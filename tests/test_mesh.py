import asyncio
import tempfile
from pathlib import Path

import numpy as np
import pytest

from omnicompute.hub import Hub
from omnicompute.limits import LimitStore
from omnicompute.model import Shard, random_model, save_model
from omnicompute.node import MeshNode
from omnicompute.worker import Worker

KEY = "test-key"


@pytest.fixture
def model_file(tmp_path):
    cfg, w = random_model(n_layer=6, n_embd=64, n_head=4, n_ctx=128)
    path = tmp_path / "m.emm"
    save_model(str(path), cfg, w)
    return str(path), cfg, w


async def start_mesh(model_path, workers):
    hub = Hub(KEY, "127.0.0.1", 0, model_path)
    await hub.start()
    nodes = []
    for name, relay_only, score in workers:
        n = MeshNode(name, KEY, f"http://127.0.0.1:{hub.port}",
                     {"role": "worker", "kind": "native", "score": score, "mem_mb": 1024},
                     allow_direct=not relay_only)
        # full speed, and always in the split (the planner would otherwise use just one fast device)
        Worker(LimitStore(Path(tempfile.mkdtemp()) / "limits.json", {"cpu": 100, "always": True})).attach(n)
        await n.start()
        nodes.append(n)
    return hub, nodes


def reference(cfg, w, prompt, n):
    ids = list(prompt.encode())
    sh = Shard(cfg, w, 0, cfg["n_layer"])
    out = []
    for _ in range(n):
        logits = sh.forward(str(len(ids)), np.array(ids, np.int32), 0)
        logits[:32] = logits[127:] = -1e9  # orchestrator keeps byte-model output printable
        nxt = int(np.argmax(logits))
        ids.append(nxt)
        out.append(nxt)
    return out


async def collect(hub, prompt, n):
    events = [ev async for ev in hub.orchestrator.generate(prompt, n)]
    return events, [e["id"] for e in events if e["type"] == "token"]


async def test_generation_across_devices_matches_single_device(model_file):
    path, cfg, w = model_file
    # the fastest device goes last (it also computes the output head): chain a -> b -> c
    hub, nodes = await start_mesh(path, [("a", False, 2), ("b", False, 1), ("c", True, 3)])
    try:
        events, ids = await collect(hub, "hello mesh", 12)
        assert ids == reference(cfg, w, "hello mesh", 12)
        plan = events[0]["plan"]
        assert [p[0] for p in plan] == ["a", "b", "c"] and plan[-1][2] == 6
        # a -> b punched a direct UDP path; c is relay-only so b -> c went through the hub
        a, b, c = nodes
        assert a.path_to("b") == "direct"  # first token went via relay while punching
        assert b.path_to("c") == "relay"
        assert a.stats["direct"] > 0 and b.stats["relay"] > 0
    finally:
        for n in nodes:
            await n.close()
        await hub.close()


async def test_replans_when_a_device_drops(model_file):
    path, cfg, w = model_file
    hub, nodes = await start_mesh(path, [("a", False, 2), ("b", False, 1)])
    try:
        gen = hub.orchestrator.generate("drop test", 10)
        got = []
        async for ev in gen:
            if ev["type"] == "token":
                got.append(ev["id"])
                if len(got) == 3:
                    await nodes[1].close()  # b disappears mid-generation
                    await asyncio.sleep(0.2)
            if ev["type"] == "replan":
                assert [p[0] for p in ev["plan"]] == ["a"]
        assert got == reference(cfg, w, "drop test", 10)
    finally:
        await nodes[0].close()
        await hub.close()


async def test_wrong_key_is_rejected(model_file):
    hub = Hub(KEY, "127.0.0.1", 0, model_file[0])
    await hub.start()
    try:
        n = MeshNode("x", "not-the-key", f"http://127.0.0.1:{hub.port}")
        with pytest.raises(PermissionError):
            await n.start()
    finally:
        await hub.close()


async def test_large_message_over_udp(model_file):
    hub, (a, b) = await start_mesh(model_file[0], [("a", False, 1), ("b", False, 1)])
    try:
        got = asyncio.get_running_loop().create_future()

        async def handler(peer, header, payload):
            got.set_result(payload)

        b.handler = handler
        assert await a.connect_direct("b")
        blob = np.random.default_rng(1).bytes(3_000_000)
        assert await a.notify("b", {"op": "x"}, blob) == "direct"
        assert await asyncio.wait_for(got, 20) == blob
    finally:
        await a.close()
        await b.close()
        await hub.close()
