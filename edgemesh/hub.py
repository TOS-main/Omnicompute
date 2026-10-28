"""The hub: one process, one port number (TCP + UDP), that ties the mesh together.

  * HTTP   /            web page: run prompts, see devices, turn this browser into a worker
  * WS     /ws          control channel for every device (native and browser)
  * UDP    same port    reflector: tells native devices their public ip:port (like STUN)
  * relay               forwards messages between devices that could not hole-punch
  * orchestrator        holds the model, splits it over the workers, runs generations

Run it on any machine the other devices can reach: a laptop on the same
Wi-Fi, a home PC with the port forwarded, or a small cloud VM for devices
spread across Wi-Fi and cellular networks.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
from importlib import resources

from aiohttp import WSMsgType, web
from cryptography.exceptions import InvalidTag

from .crypto import MeshCipher
from .messenger import Messenger
from .protocol import pack, unpack

log = logging.getLogger("edgemesh.hub")

OBSERVE_REQ = b"EMOB"
OBSERVE_RESP = b"EMOA"
HUB_ID = "hub"


class _Client:
    def __init__(self, node_id: str, ws: web.WebSocketResponse, info: dict, native: bool):
        self.node_id, self.ws, self.info, self.native = node_id, ws, info, native
        self.udp_addr: tuple[str, int] | None = None

    async def send(self, header: dict, payload: bytes = b"") -> None:
        await self.ws.send_bytes(pack(header, payload))


class _Reflector(asyncio.DatagramProtocol):
    def __init__(self, hub: "Hub"):
        self.hub = hub

    def connection_made(self, transport):
        self.transport = transport

    def datagram_received(self, data, addr):
        if not data.startswith(OBSERVE_REQ):
            return
        try:
            node_id = self.hub.cipher.open(data[4:], OBSERVE_REQ).decode()
        except (InvalidTag, ValueError):
            return
        client = self.hub.clients.get(node_id)
        if client is not None:
            client.udp_addr = (addr[0], addr[1])
        reply = json.dumps({"addr": [addr[0], addr[1]]}).encode()
        self.transport.sendto(OBSERVE_RESP + self.hub.cipher.seal(reply, OBSERVE_RESP), addr)


class HubEndpoint(Messenger):
    """The hub's own seat in the mesh (used by the orchestrator)."""

    node_id = HUB_ID

    def __init__(self, hub: "Hub"):
        super().__init__()
        self.hub = hub

    async def send(self, peer_id: str, header: dict, payload: bytes = b"") -> str:
        await self.hub.route(HUB_ID, peer_id, pack(header, payload))
        return "relay"


class Hub:
    def __init__(self, secret: str, host: str = "0.0.0.0", port: int = 8000, model_path: str | None = None):
        self.secret = secret
        self.cipher = MeshCipher(secret)
        self.host, self.port = host, port
        self.clients: dict[str, _Client] = {}
        self.endpoint = HubEndpoint(self)
        self.orchestrator = None
        self.model_path = model_path
        self.lan_origin: str | None = None  # shown on the web page when it is opened via localhost
        self._runner = None
        self._udp = None

    # ---------- lifecycle ----------

    async def start(self) -> None:
        if self.model_path:
            from .orchestrator import Orchestrator

            self.orchestrator = Orchestrator(self.endpoint, self.model_path, self.workers)
        app = web.Application(client_max_size=1 << 30)
        app.add_routes([
            web.get("/", self._index),
            web.get("/ws", self._ws),
            web.get("/api/status", self._status),
            web.post("/api/generate", self._generate),
            web.get("/static/{name}", self._static),
        ])
        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.host, self.port)
        await site.start()
        self.port = site._server.sockets[0].getsockname()[1]
        loop = asyncio.get_running_loop()
        self._udp, _ = await loop.create_datagram_endpoint(lambda: _Reflector(self), local_addr=(self.host, self.port))
        log.info("hub listening on tcp+udp port %d", self.port)

    async def close(self) -> None:
        if self._udp:
            self._udp.close()
        for c in list(self.clients.values()):
            await c.ws.close()
        if self._runner:
            await self._runner.cleanup()

    def workers(self) -> list[dict]:
        return [{"node_id": c.node_id, "info": c.info} for c in self.clients.values()]

    # ---------- routing ----------

    async def route(self, src: str, dst: str, blob: bytes, sealed: bytes | None = None) -> None:
        """Deliver a plaintext message blob from src to dst. `sealed` is the already-encrypted
        form, forwarded untouched when both ends are native devices."""
        if dst == HUB_ID:
            self.endpoint._deliver(src, blob)
            return
        target = self.clients.get(dst)
        if target is None:
            log.debug("drop message %s -> %s: not connected", src, dst)
            return
        if target.native:
            payload = sealed if sealed is not None else self.cipher.seal(blob, b"relay")
        else:
            payload = blob  # browser devices: protected by the WebSocket transport (use https)
        try:
            await target.send({"type": "relay", "from": src}, payload)
        except ConnectionError:
            pass

    # ---------- WebSocket control channel ----------

    async def _ws(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(max_msg_size=0, heartbeat=20)
        await ws.prepare(request)
        client = None
        try:
            nonce = os.urandom(16).hex()
            await ws.send_bytes(pack({"type": "challenge", "nonce": nonce}))
            first = await ws.receive(timeout=10)
            hello = json.loads(first.data) if first.type == WSMsgType.TEXT else unpack(first.data)[0]
            node_id = str(hello.get("node_id", ""))[:64]
            if "mac" in hello:  # native: challenge-response, the secret never crosses the wire
                ok = self.cipher.verify(f"{nonce}:{node_id}".encode(), hello["mac"])
                native = True
            else:  # browser: presents the key from its join link
                ok = hmac.compare_digest(str(hello.get("key", "")), self.secret)
                native = False
            if not node_id or node_id == HUB_ID or not ok:
                await ws.send_bytes(pack({"type": "error", "error": "wrong mesh key"}))
                return ws
            old = self.clients.get(node_id)
            if old is not None:
                await old.ws.close()
            client = _Client(node_id, ws, dict(hello.get("info", {})), native)
            self.clients[node_id] = client
            await client.send({"type": "welcome", "udp_port": self.port, "observed_ip": request.remote})
            log.info("%s joined: %s %s", node_id, client.info.get("kind"), client.info.get("platform", ""))
            async for msg in ws:
                if msg.type == WSMsgType.BINARY:
                    header, payload = unpack(msg.data)
                    await self._dispatch(client, header, payload)
        except (asyncio.TimeoutError, ConnectionError, ValueError, TypeError):
            pass
        finally:
            if client is not None and self.clients.get(client.node_id) is client:
                del self.clients[client.node_id]
                log.info("%s left", client.node_id)
                if self.orchestrator is not None:
                    self.orchestrator.node_left(client.node_id)
        return ws

    async def _dispatch(self, client: _Client, header: dict, payload: bytes) -> None:
        kind = header.get("type")
        if kind == "relay":
            dst = str(header.get("to"))
            if client.native:
                target = self.clients.get(dst)
                if target is not None and target.native:
                    await self.route(client.node_id, dst, b"", sealed=payload)
                    return
                try:
                    blob = self.cipher.open(payload, b"relay")
                except InvalidTag:
                    return
            else:
                blob = payload
            await self.route(client.node_id, dst, blob)
        elif kind == "list":
            peers = [{"node_id": c.node_id, "info": c.info} for c in self.clients.values() if c is not client]
            await client.send({"type": "peers", "req": header.get("req"), "peers": peers})
        elif kind == "connect":
            target = self.clients.get(header.get("peer"))
            if target is None or not (target.native and client.native and target.info.get("direct", True)):
                await client.send({"type": "punch", "peer": header.get("peer"), "error": "peer cannot punch"})
                return
            await target.send({"type": "punch", "peer": client.node_id, "candidates": _candidates(client)})
            await client.send({"type": "punch", "peer": target.node_id, "candidates": _candidates(target)})

    # ---------- HTTP ----------

    def _authorized(self, request: web.Request) -> bool:
        key = request.headers.get("X-Mesh-Key") or request.query.get("k", "")
        return hmac.compare_digest(key, self.secret)

    async def _index(self, request: web.Request) -> web.Response:
        return web.Response(text=_web_file("index.html"), content_type="text/html")

    async def _static(self, request: web.Request) -> web.Response:
        name = request.match_info["name"]
        if name not in ("app.js", "worker.js", "model.js", "style.css"):
            raise web.HTTPNotFound()
        ctype = "text/css" if name.endswith(".css") else "application/javascript"
        return web.Response(text=_web_file(name), content_type=ctype)

    async def _status(self, request: web.Request) -> web.Response:
        if not self._authorized(request):
            raise web.HTTPUnauthorized(text="wrong mesh key")
        o = self.orchestrator
        return web.json_response({
            "model": o.cfg if o else None,
            "lan_origin": self.lan_origin,
            "plan": [list(p) for p in o.plan] if o else [],
            "devices": [{"node_id": c.node_id, "kind": c.info.get("kind"), "role": c.info.get("role"),
                         "platform": c.info.get("platform"), "score": c.info.get("score"),
                         "mem_mb": c.info.get("mem_mb"), "public_udp": c.udp_addr}
                        for c in self.clients.values()],
        })

    async def _generate(self, request: web.Request) -> web.StreamResponse:
        """POST {"prompt", "max_new", "temperature"} -> newline-delimited JSON events."""
        if not self._authorized(request):
            raise web.HTTPUnauthorized(text="wrong mesh key")
        if self.orchestrator is None:
            raise web.HTTPServiceUnavailable(text="this hub was started without a model")
        body = await request.json()
        resp = web.StreamResponse(headers={"Content-Type": "application/x-ndjson", "Cache-Control": "no-cache"})
        await resp.prepare(request)
        try:
            async for ev in self.orchestrator.generate(str(body.get("prompt", "")), int(body.get("max_new", 32)),
                                                       float(body.get("temperature", 0.0)), body.get("seed")):
                await resp.write((json.dumps(ev) + "\n").encode())
        except Exception as e:
            log.warning("generation failed: %s", e)
            await resp.write((json.dumps({"type": "error", "error": str(e)}) + "\n").encode())
        await resp.write_eof()
        return resp


def _candidates(c: _Client) -> list:
    """Public (server-reflexive) mapping first, then the device's own LAN addresses."""
    out = []
    if c.udp_addr:
        out.append(list(c.udp_addr))
    for addr in c.info.get("local_candidates", []):
        if list(addr) not in out:
            out.append(list(addr))
    return out


def _web_file(name: str) -> str:
    return resources.files("edgemesh").joinpath("web", name).read_text(encoding="utf-8")
