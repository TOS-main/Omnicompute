"""--public: the tunnel helper starts cloudflared and picks the public address out of its log.
(Uses a stand-in cloudflared; the real one needs internet access.)"""

import os
import stat
import sys

import pytest

from edgemesh import tunnel

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="stand-in script is a shell script")


async def test_tunnel_reads_public_url(tmp_path, monkeypatch):
    fake = tmp_path / "cloudflared"
    fake.write_text("#!/bin/sh\necho 'INF Requesting new quick Tunnel' >&2\n"
                    "echo 'INF |  https://brave-otter-12.trycloudflare.com  |' >&2\nsleep 30\n")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    proc, url = await tunnel.start(8000, tmp_path)
    try:
        assert url == "https://brave-otter-12.trycloudflare.com"
    finally:
        proc.kill()
        await proc.wait()


def test_asset_names():
    assert tunnel._asset().startswith("cloudflared-")
