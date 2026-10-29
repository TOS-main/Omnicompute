"""A mesh device: talks to the rendezvous server and to other devices.

Path selection per peer:
  1. The first message to a peer goes through the hub relay right away, while
     in the background the hub introduces us and both sides fire encrypted UDP
     "punch" packets at every candidate address of the other (public mapping +
     LAN addresses). The first packet that gets through opens a direct path,
     which covers same-LAN, home routers and most cone-type NATs; later
     messages use it.
  2. If nothing arrives within PUNCH_TIMEOUT (symmetric NAT, strict CGNAT, UDP
     blocked), or the direct path later stops acknowledging packets, messages
     keep using the relay. Direct is retried every DIRECT_RETRY seconds.

Over UDP, messages are split into chunks and sent with a small selective-ACK
sliding window (ReliableChannel), since tensors are far larger than one packet.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import time
from urllib.parse import unquote, urlsplit

import aiohttp
from cryptography.exceptions import InvalidTag

from .crypto import MeshCipher, contribute_key, is_contribute_key
from .hub import OBSERVE_REQ, OBSERVE_RESP
from .messenger import Messenger
from .protocol import pack, unpack
from .wsparts import Joiner, split

log = logging.getLogger("edgemesh.node")

PEER_MAGIC = b"EMPK"
PUNCH_TIMEOUT = 3.0
PUNCH_INTERVAL = 0.1
DIRECT_RETRY = 60.0
KEEPALIVE = 10.0


class PathDown(Exception):
    pass


class ReliableChannel:
    """Chunked, acknowledged message delivery over a punched UDP path."""

    CHUNK = 1100  # keeps sealed packets under typical 1280-byte safe MTU
    WINDOW = 256
    MAX_RETRIES = 12
    SEND_TIMEOUT = 120.0

    def __init__(self, node: "MeshNode", peer_id: str, addr: tuple):
        self.node, self.peer_id, self.addr = node, peer_id, addr
        self.seq = 0
        self.inflight: dict[int, list] = {}  # seq -> [packet, sent_at, retries, msg_id]
        self.pending: dict[str, list] = {}  # msg_id -> [remaining seqs, future]
        self.window = asyncio.Semaphore(self.WINDOW)
        self.assembly: dict[str, dict] = {}
        self.delivered: dict[str, float] = {}
        self.srtt = 0.2
        self.alive = True
        self._timer = asyncio.ensure_future(self._retransmit_loop())

    def close(self, reason: str = "closed") -> None:
        if not self.alive:
            return
        self.alive = False
        self._timer.cancel()
        for _, fut, _ in self.pending.values():
            if not fut.done():
                fut.set_exception(PathDown(reason))
        self.pending.clear()
        self.inflight.clear()
        for _ in range(self.WINDOW):  # wake senders blocked on the window
            self.window.release()

    async def send(self, blob: bytes) -> None:
        if not self.alive:
            raise PathDown("channel closed")
        msg_id = os.urandom(6).hex()
        n = max(1, -(-len(blob) // self.CHUNK))
        fut = asyncio.get_running_loop().create_future()
        remaining: set[int] = set()
        entry = [remaining, fut, False]  # [unacked seqs, future, all chunks queued]
        self.pending[msg_id] = entry
        for i in range(n):
            await self.window.acquire()
            if not self.alive:
                self.window.release()
                raise PathDown("channel closed")
            seq = self.seq
            self.seq += 1
            remaining.add(seq)
            pkt = self.node._seal_peer(
                {"t": "d", "from": self.node.node_id, "s": seq, "m": msg_id, "i": i, "n": n},
                blob[i * self.CHUNK : (i + 1) * self.CHUNK],
            )
            self.inflight[seq] = [pkt, time.monotonic(), 0, msg_id]
            self.node._sendto(pkt, self.addr)
        entry[2] = True
        self._maybe_done(msg_id)
        await asyncio.wait_for(fut, self.SEND_TIMEOUT)

    def _maybe_done(self, msg_id: str) -> None:
        p = self.pending.get(msg_id)
        if p is not None and p[2] and not p[0]:
            del self.pending[msg_id]
            if not p[1].done():
                p[1].set_result(None)

    def on_ack(self, seqs: list[int]) -> None:
        now = time.monotonic()
        for s in seqs:
            entry = self.inflight.pop(s, None)
            if entry is None:
                continue
            self.window.release()
            if entry[2] == 0:
                self.srtt = 0.875 * self.srtt + 0.125 * (now - entry[1])
            p = self.pending.get(entry[3])
            if p is not None:
                p[0].discard(s)
                self._maybe_done(entry[3])

    def on_data(self, header: dict, chunk: bytes) -> None:
        self.node._sendto(self.node._seal_peer({"t": "a", "from": self.node.node_id, "s": [header["s"]]}), self.addr)
        msg_id = header["m"]
        if msg_id in self.delivered:
            return
        a = self.assembly.setdefault(msg_id, {"n": header["n"], "parts": {}})
        a["parts"][header["i"]] = chunk
        if len(a["parts"]) == a["n"]:
            del self.assembly[msg_id]
            self.delivered[msg_id] = time.monotonic()
            blob = b"".join(a["parts"][i] for i in range(a["n"]))
            self.node._deliver(self.peer_id, blob)

    async def _retransmit_loop(self) -> None:
        while self.alive:
            await asyncio.sleep(0.05)
            now = time.monotonic()
            rto = min(2.0, max(0.15, 3 * self.srtt))
            for seq, entry in list(self.inflight.items()):
                if now - entry[1] > rto * (1.5 ** entry[2]):
                    entry[2] += 1
                    if entry[2] > self.MAX_RETRIES:
                        log.warning("direct path to %s stopped answering", self.peer_id)
                        self.node._direct_failed(self.peer_id, self)
                        return
                    entry[1] = now
                    self.node._sendto(entry[0], self.addr)
            cutoff = now - 300
            for m in [m for m, t in self.delivered.items() if t < cutoff]:
                del self.delivered[m]




class _PeerState:
    def __init__(self):
        self.channel: ReliableChannel | None = None
        self.punched = asyncio.Event()
        self.last_attempt = -1e9
        self.punch_task: asyncio.Task | None = None
        self.connecting: asyncio.Future | None = None


class _UDP(asyncio.DatagramProtocol):
    def __init__(self, node: "MeshNode"):
        self.node = node

    def datagram_received(self, data, addr):
        self.node._on_datagram(data, addr)

    def error_received(self, exc):
        log.debug("udp error: %s", exc)


def parse_join_url(url: str) -> tuple[str, str]:
    """'http://1.2.3.4:8000/?k=SECRET' -> ('http://1.2.3.4:8000', 'SECRET').
    A contribute-only link ('.../contribute?c=KEY') gives its contribute key instead."""
    url = url.strip().strip("\"'<>()[] .,;")  # copied with quotes or sentence punctuation
    parts = urlsplit(url if "://" in url else "http://" + url)
    q = dict(q.split("=", 1) for q in parts.query.split("&") if "=" in q)
    return f"{parts.scheme}://{parts.netloc}", unquote(q.get("k") or q.get("c", "")).strip()


class MeshNode(Messenger):
    """A native (desktop / server / Termux) device in the mesh."""

    def __init__(self, node_id: str, secret: str, hub_url: str, info: dict | None = None,
                 allow_direct: bool = True, udp_port: int = 0, handler=None):
        super().__init__(handler)
        self.node_id = node_id
        self.cipher = MeshCipher(contribute_key(secret))  # seals device-to-device traffic
        self.auth = MeshCipher(secret)  # proves which key we hold when logging in
        self.contributor = is_contribute_key(secret)
        self.hub_url = hub_url.rstrip("/")
        self.info = {**(info or {}), "direct": allow_direct}
        self.allow_direct = allow_direct
        self.udp_port = udp_port
        self.public_addr: tuple | None = None
        self.peers: dict[str, _PeerState] = {}
        self._server_waiters: dict[str, asyncio.Future] = {}
        self._tasks: list[asyncio.Task] = []
        self._observed = asyncio.Event()
        self._ws_lock = asyncio.Lock()
        self.closed = asyncio.Event()
        self.stats = {"direct": 0, "relay": 0}

    # ---------- lifecycle ----------

    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        self.udp, _ = await loop.create_datagram_endpoint(lambda: _UDP(self), local_addr=("0.0.0.0", self.udp_port))
        port = self.udp.get_extra_info("sockname")[1]
        self.info["local_candidates"] = [[ip, port] for ip in _local_ips()]

        ws_url = self.hub_url.replace("https://", "wss://").replace("http://", "ws://") + "/ws"
        self.session = aiohttp.ClientSession()
        try:
            self.ws = await self.session.ws_connect(ws_url, max_msg_size=0, heartbeat=20)
            challenge, _ = unpack(await self.ws.receive_bytes(timeout=10))
            mac = self.auth.sign(f"{challenge['nonce']}:{self.node_id}".encode())
            await self.ws.send_bytes(pack({"node_id": self.node_id, "mac": mac, "info": self.info}))
            welcome, _ = unpack(await self.ws.receive_bytes(timeout=10))
        except Exception:
            await self.session.close()
            self.udp.close()
            raise
        if welcome.get("type") != "welcome":
            await self.session.close()
            self.udp.close()
            raise PermissionError(welcome.get("error", "hub refused login"))
        host = urlsplit(self.hub_url).hostname
        self.hub_udp = (socket.gethostbyname(host), welcome["udp_port"])

        self._tasks.append(asyncio.ensure_future(self._server_loop()))
        self._tasks.append(asyncio.ensure_future(self._keepalive_loop()))
        if self.allow_direct:
            for _ in range(10):
                self._observe()
                try:
                    await asyncio.wait_for(self._observed.wait(), 0.3)
                    break
                except asyncio.TimeoutError:
                    pass
        log.info("%s online, public udp %s", self.node_id, self.public_addr)

    async def close(self) -> None:
        for t in self._tasks:
            t.cancel()
        for p in self.peers.values():
            if p.channel:
                p.channel.close()
            if p.punch_task:
                p.punch_task.cancel()
        await self.ws.close()
        await self.session.close()
        self.udp.close()
        self.closed.set()

    # ---------- public API ----------

    async def list_peers(self) -> list[dict]:
        req = os.urandom(4).hex()
        fut = asyncio.get_running_loop().create_future()
        self._server_waiters[req] = fut
        await self._server_send({"type": "list", "req": req})
        return await asyncio.wait_for(fut, 10)

    async def send(self, peer_id: str, header: dict, payload: bytes = b"") -> str:
        """Deliver one message, direct if possible, else via the hub relay. Returns the path used."""
        blob = pack(header, payload)
        channel = await self._ensure_path(peer_id)
        if channel is not None:
            try:
                await channel.send(blob)
                self.stats["direct"] += 1
                return "direct"
            except (PathDown, asyncio.TimeoutError):
                log.info("direct send to %s failed, falling back to relay", peer_id)
        await self._server_send({"type": "relay", "to": peer_id}, self.cipher.seal(blob, b"relay"))
        self.stats["relay"] += 1
        return "relay"

    async def connect_direct(self, peer_id: str) -> bool:
        """Try hole punching to a peer now and wait for the outcome. True if a direct path is up."""
        p = self._peer(peer_id)
        if not (p.channel and p.channel.alive) and self.allow_direct:
            if p.connecting is None:
                p.connecting = asyncio.ensure_future(self._connect(peer_id, p))
            await asyncio.shield(p.connecting)
        return self.path_to(peer_id) == "direct"

    async def update_info(self, fields: dict) -> None:
        """Change what the hub knows about this device (e.g. its usage limit)."""
        self.info.update(fields)
        if getattr(self, "ws", None) is not None and not self.ws.closed:
            await self._server_send({"type": "info", "info": fields})

    def path_to(self, peer_id: str) -> str:
        p = self.peers.get(peer_id)
        return "direct" if p and p.channel and p.channel.alive else "relay"

    # ---------- path management ----------

    def _peer(self, peer_id: str) -> _PeerState:
        if peer_id not in self.peers:
            self.peers[peer_id] = _PeerState()
        return self.peers[peer_id]

    async def _ensure_path(self, peer_id: str) -> ReliableChannel | None:
        """Return a live direct channel, or None to use the relay. Never waits on hole punching:
        the attempt runs in the background and later messages switch over once it succeeds."""
        p = self._peer(peer_id)
        if p.channel and p.channel.alive:
            return p.channel
        if (self.allow_direct and peer_id != "hub" and p.connecting is None
                and time.monotonic() - p.last_attempt >= DIRECT_RETRY):
            p.connecting = asyncio.ensure_future(self._connect(peer_id, p))
        return None

    async def _connect(self, peer_id: str, p: _PeerState) -> None:
        p.last_attempt = time.monotonic()
        p.punched.clear()
        try:
            await self._server_send({"type": "connect", "peer": peer_id})
            await asyncio.wait_for(p.punched.wait(), PUNCH_TIMEOUT)
            if p.channel is None:
                raise asyncio.TimeoutError
            log.info("%s -> %s: direct UDP path via %s", self.node_id, peer_id, p.channel.addr)
        except asyncio.TimeoutError:
            log.info("%s -> %s: hole punching failed, using relay", self.node_id, peer_id)
        finally:
            p.connecting = None

    def _start_punch(self, peer_id: str, candidates: list) -> None:
        if not self.allow_direct or not candidates:
            return
        p = self._peer(peer_id)
        p.last_attempt = time.monotonic()
        if p.punch_task and not p.punch_task.done():
            p.punch_task.cancel()
        log.info("%s: punching %s at %s", self.node_id, peer_id, candidates)
        p.punch_task = asyncio.ensure_future(self._punch(peer_id, p, [tuple(c) for c in candidates]))

    async def _punch(self, peer_id: str, p: _PeerState, candidates: list) -> None:
        pkt = self._seal_peer({"t": "punch", "from": self.node_id, "to": peer_id})
        deadline = time.monotonic() + PUNCH_TIMEOUT
        while time.monotonic() < deadline and not (p.channel and p.channel.alive):
            for addr in candidates:
                self._sendto(pkt, addr)
            await asyncio.sleep(PUNCH_INTERVAL)

    def _path_open(self, peer_id: str, addr: tuple) -> None:
        p = self._peer(peer_id)
        if p.channel and p.channel.alive:
            return
        p.channel = ReliableChannel(self, peer_id, addr)
        p.punched.set()

    def _direct_failed(self, peer_id: str, channel: ReliableChannel) -> None:
        channel.close("peer stopped acknowledging")
        p = self._peer(peer_id)
        if p.channel is channel:
            p.channel = None
            p.last_attempt = time.monotonic()

    # ---------- UDP ----------

    def _seal_peer(self, header: dict, data: bytes = b"") -> bytes:
        return PEER_MAGIC + self.cipher.seal(pack(header, data), PEER_MAGIC)

    def _sendto(self, data: bytes, addr: tuple) -> None:
        try:
            self.udp.sendto(data, addr)
        except OSError as e:
            log.debug("sendto %s failed: %s", addr, e)

    def _observe(self) -> None:
        self._sendto(OBSERVE_REQ + self.cipher.seal(self.node_id.encode(), OBSERVE_REQ), self.hub_udp)

    def _on_datagram(self, data: bytes, addr: tuple) -> None:
        magic = data[:4]
        try:
            if magic == OBSERVE_RESP:
                self.public_addr = tuple(json.loads(self.cipher.open(data[4:], OBSERVE_RESP))["addr"])
                self._observed.set()
                return
            if magic != PEER_MAGIC:
                return
            header, body = unpack(self.cipher.open(data[4:], PEER_MAGIC))
        except (InvalidTag, ValueError, KeyError):
            return  # not ours, wrong secret, or garbage
        sender = header.get("from")
        if not sender or sender == self.node_id or not self.allow_direct:
            return
        kind = header.get("t")
        addr = (addr[0], addr[1])
        if kind == "punch":
            if header.get("to") != self.node_id:
                return
            log.info("%s: punch from %s via %s", self.node_id, sender, addr)
            self._sendto(self._seal_peer({"t": "punch_ack", "from": self.node_id}), addr)
            self._path_open(sender, addr)
            return
        if kind == "punch_ack":
            self._path_open(sender, addr)
            return
        p = self.peers.get(sender)
        ch = p.channel if p else None
        if ch is None or not ch.alive:
            if kind != "d":
                return
            self._path_open(sender, addr)  # the peer already has a working path to us
            ch = self.peers[sender].channel
        if ch.addr != addr:
            ch.addr = addr  # peer's NAT mapping changed (e.g. Wi-Fi -> cellular handover)
        if kind == "d":
            ch.on_data(header, body)
        elif kind == "a":
            ch.on_ack(header["s"])

    # ---------- hub connection ----------

    async def _server_send(self, header: dict, payload: bytes = b"") -> None:
        async with self._ws_lock:
            for part in split(pack(header, payload)):
                await self.ws.send_bytes(part)

    async def _server_loop(self) -> None:
        joiner = Joiner()
        async for msg in self.ws:
            if msg.type != aiohttp.WSMsgType.BINARY:
                continue
            data = joiner.feed(msg.data)
            if data is None:
                continue
            header, payload = unpack(data)
            kind = header.get("type")
            if kind == "relay":
                try:
                    blob = self.cipher.open(payload, b"relay")
                except InvalidTag:
                    continue
                self._deliver(header["from"], blob)
            elif kind == "punch":
                if "error" in header:
                    self._peer(header["peer"]).punched.set()  # hub says it can't punch: use relay now
                else:
                    self._start_punch(header["peer"], header.get("candidates", []))
            elif kind == "peers":
                fut = self._server_waiters.pop(header.get("req"), None)
                if fut and not fut.done():
                    fut.set_result(header["peers"])
        log.error("%s lost connection to the hub", self.node_id)
        self.closed.set()

    async def _keepalive_loop(self) -> None:
        while True:
            await asyncio.sleep(KEEPALIVE)
            if self.allow_direct:
                self._observe()  # keeps our NAT mapping toward the hub fresh
            for p in self.peers.values():
                if p.channel and p.channel.alive:
                    self._sendto(self._seal_peer({"t": "ka", "from": self.node_id}), p.channel.addr)


def _local_ips() -> list[str]:
    """IPv4 LAN addresses to offer as candidates (the UDP socket is IPv4-only in this MVP)."""
    ips = set()
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))  # sends nothing; just selects the outbound interface
            ips.add(s.getsockname()[0])
    except OSError:
        pass
    try:
        for res in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(res[4][0])
    except OSError:
        pass
    ips.add("127.0.0.1")
    return sorted(ips)
