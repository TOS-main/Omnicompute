"""The permanent link: the host posts its sealed address, devices find it after it moves."""

import asyncio
import json

import pytest
from aiohttp import web

from edgemesh import beacon
from edgemesh.crypto import contribute_key
from edgemesh.hub import Hub
from edgemesh.node import parse_join_url
from test_mesh import KEY, model_file  # noqa: F401


async def fake_ntfy():
    topics: dict[str, list] = {}

    async def post(request):
        topics.setdefault(request.match_info["topic"], []).append(
            {"event": "message", "message": await request.text()})
        return web.json_response({"ok": True})

    async def get(request):
        return web.Response(text="".join(json.dumps(e) + "\n" for e in topics.get(request.match_info["topic"], [])))

    app = web.Application()
    app.add_routes([web.post("/{topic}", post), web.get("/{topic}/json", get)])
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    return runner, f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}", topics


def test_sealed_and_topic_follow_the_contribute_key():
    ck = contribute_key(KEY)
    assert beacon.topic(KEY) == beacon.topic(ck)  # both links find the host
    box = beacon.seal(KEY, {"url": "https://x.trycloudflare.com"})
    assert "trycloudflare" not in box
    assert beacon.unseal(ck, box) == {"url": "https://x.trycloudflare.com"}
    assert beacon.unseal("another-mesh-key", box) is None


def test_permanent_links_parse():
    ck = contribute_key(KEY)
    assert parse_join_url(f"https://bob.github.io/edgemesh/go.html#c={ck}") == ("beacon:", ck)
    assert parse_join_url(f"https://bob.github.io/edgemesh/go.html#k={KEY}") == ("beacon:", KEY)
    assert parse_join_url(ck) == ("beacon:", ck)


async def test_a_worker_follows_the_host_to_a_new_address(model_file, monkeypatch, tmp_path):
    from edgemesh import cli

    runner, url, topics = await fake_ntfy()
    monkeypatch.setenv("EDGEMESH_BEACON", url)
    monkeypatch.setattr(cli, "HOME", tmp_path)
    monkeypatch.setenv("EDGEMESH_SCORE", "5")
    hub = Hub(KEY, "127.0.0.1", 0, model_file[0])
    await hub.start()
    assert await beacon.publish(KEY, {"url": f"http://127.0.0.1:{hub.port}"})
    assert (await beacon.lookup(contribute_key(KEY)))["url"] == f"http://127.0.0.1:{hub.port}"
    task = asyncio.ensure_future(cli._run_worker("beacon:", contribute_key(KEY), "roamer", True, 0.2))
    hub2 = None
    try:
        for _ in range(100):
            if "roamer" in hub.clients:
                break
            await asyncio.sleep(0.1)
        assert "roamer" in hub.clients
        await hub.close()  # the host restarts somewhere else (new tunnel address)
        hub2 = Hub(KEY, "127.0.0.1", 0, model_file[0])
        await hub2.start()
        await beacon.publish(KEY, {"url": f"http://127.0.0.1:{hub2.port}"})
        for _ in range(300):
            if "roamer" in hub2.clients:
                break
            await asyncio.sleep(0.1)
        assert "roamer" in hub2.clients
    finally:
        task.cancel()
        if hub2:
            await hub2.close()
        await runner.cleanup()
