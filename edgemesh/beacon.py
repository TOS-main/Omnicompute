"""A permanent address for a mesh whose host moves around.

A free Cloudflare tunnel gets a new random address every time the host restarts, and a phone
host changes Wi-Fi addresses too. So the host posts where it is now to a public message board
(ntfy.sh by default; EDGEMESH_BEACON points at another ntfy server, "off" turns it off), and
devices look it up there when they can't reach it:

* the topic name is derived from the contribute key, so only people with a link can find it;
* the message (current https and Wi-Fi addresses) is sealed with AES-GCM under a key derived
  from the contribute key, so the board only ever sees ciphertext.

The web page does the same with WebCrypto (web/common.js: findHost), which is how the
permanent link on the project's GitHub page (docs/go.html) forwards to the host.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .crypto import _derive, contribute_key

log = logging.getLogger("edgemesh.beacon")

REPUBLISH_S = 30 * 60  # ntfy.sh keeps messages for 12 hours


def server() -> str | None:
    s = os.environ.get("EDGEMESH_BEACON", "https://ntfy.sh").rstrip("/")
    return None if s.lower() in ("", "0", "off", "none") else s


def topic(key: str) -> str:
    return "edgemesh-" + _derive(contribute_key(key), b"beacon-topic")[:12].hex()


def seal(key: str, info: dict) -> str:
    nonce = os.urandom(12)
    ct = AESGCM(_derive(contribute_key(key), b"beacon-key")).encrypt(nonce, json.dumps(info).encode(), b"edgemesh-beacon")
    return base64.urlsafe_b64encode(nonce + ct).decode().rstrip("=")


def unseal(key: str, text: str) -> dict | None:
    try:
        raw = base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))
        data = AESGCM(_derive(contribute_key(key), b"beacon-key")).decrypt(raw[:12], raw[12:], b"edgemesh-beacon")
        return json.loads(data)
    except Exception:  # someone else's message on the topic, or garbage
        return None


async def publish(key: str, info: dict) -> bool:
    import aiohttp

    base = server()
    if base is None:
        return False
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as s:
            async with s.post(f"{base}/{topic(key)}", data=seal(key, {**info, "t": int(time.time())})) as r:
                return r.status < 300
    except Exception as e:
        log.info("could not post the mesh's address: %s", e)
        return False


async def lookup(key: str) -> dict | None:
    """The newest address the host posted: {"url", "lan", "t"}, or None."""
    import aiohttp

    base = server()
    if base is None:
        return None
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as s:
            async with s.get(f"{base}/{topic(key)}/json", params={"poll": "1", "since": "12h"}) as r:
                text = await r.text()
    except Exception as e:
        log.info("could not look up the mesh's address: %s", e)
        return None
    best = None
    for line in text.splitlines():
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        info = unseal(key, ev.get("message", "")) if ev.get("event", "message") == "message" else None
        if info and info.get("url") and (best is None or info.get("t", 0) >= best.get("t", 0)):
            best = info
    return best


class Publisher:
    """Keeps the mesh's current address posted while the host runs."""

    def __init__(self, key: str, info_fn):
        self.key, self.info_fn = key, info_fn
        self._task: asyncio.Task | None = None
        self.last_ok: float | None = None

    def start(self) -> None:
        if server() is not None:
            self._task = asyncio.ensure_future(self._run())

    async def _run(self) -> None:
        last = None
        while True:
            info = self.info_fn()
            if info.get("url") and (info != last or time.time() - (self.last_ok or 0) > REPUBLISH_S):
                if await publish(self.key, info):
                    last, self.last_ok = info, time.time()
            await asyncio.sleep(20)

    def stop(self) -> None:
        if self._task:
            self._task.cancel()
