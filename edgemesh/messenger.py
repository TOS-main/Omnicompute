"""Request/response and one-way messaging on top of any transport.

Subclasses implement ``send(peer_id, header, payload)`` and call
``_deliver(peer_id, blob)`` for every message that arrives.
"""

from __future__ import annotations

import asyncio
import logging
import os

from .protocol import unpack

log = logging.getLogger("edgemesh.messenger")


class Messenger:
    node_id: str

    def __init__(self, handler=None):
        # handler: async (peer_id, header, payload) -> (header, payload) | None
        self.handler = handler
        self._rpc: dict[str, asyncio.Future] = {}

    async def send(self, peer_id: str, header: dict, payload: bytes = b"") -> str:
        raise NotImplementedError

    async def notify(self, peer_id: str, header: dict, payload: bytes = b"") -> str:
        """One-way message; the handler's return value is ignored."""
        return await self.send(peer_id, {**header, "kind": "msg"}, payload)

    async def request(self, peer_id: str, header: dict, payload: bytes = b"", timeout: float = 300) -> tuple[dict, bytes]:
        rpc = os.urandom(8).hex()
        fut = asyncio.get_running_loop().create_future()
        self._rpc[rpc] = fut
        try:
            await self.send(peer_id, {**header, "rpc": rpc, "kind": "req"}, payload)
            rh, rp = await asyncio.wait_for(fut, timeout)
        finally:
            self._rpc.pop(rpc, None)
        if "error" in rh:
            raise RuntimeError(f"{peer_id}: {rh['error']}")
        return rh, rp

    def _deliver(self, peer_id: str, blob: bytes) -> None:
        header, payload = unpack(blob)
        kind = header.get("kind")
        if kind == "resp":
            fut = self._rpc.get(header.get("rpc"))
            if fut and not fut.done():
                fut.set_result((header, payload))
        elif kind in ("req", "msg"):
            asyncio.ensure_future(self._serve(peer_id, header, payload))

    async def _serve(self, peer_id: str, header: dict, payload: bytes) -> None:
        try:
            if self.handler is None:
                raise RuntimeError("this node does not accept requests")
            result = await self.handler(peer_id, header, payload)
            rh, rp = result if result is not None else ({}, b"")
        except Exception as e:
            log.exception("%s: op %s from %s failed", self.node_id, header.get("op"), peer_id)
            rh, rp = {"error": f"{type(e).__name__}: {e}"}, b""
        if header.get("kind") == "req":
            await self.send(peer_id, {**rh, "rpc": header["rpc"], "kind": "resp"}, rp)
