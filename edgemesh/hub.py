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
import hashlib
import hmac
import json
import logging
import os
from importlib import resources
from pathlib import Path

from aiohttp import WSMsgType, web
from cryptography.exceptions import InvalidTag

from . import project
from .crypto import MeshCipher, contribute_key
from .hf import split_spec
from .messenger import Messenger
from .protocol import pack, unpack
from .wsparts import Joiner, split

log = logging.getLogger("edgemesh.hub")

OBSERVE_REQ = b"EMOB"
OBSERVE_RESP = b"EMOA"
HUB_ID = "hub"
INFO_FIELDS = {"limit", "limit_reason", "limits", "score", "mem_mb", "steps", "cores", "power", "uses", "gpu"}


class _Client:
    def __init__(self, node_id: str, ws: web.WebSocketResponse, info: dict, native: bool, contributor: bool = False):
        self.node_id, self.ws, self.info, self.native = node_id, ws, info, native
        self.contributor = contributor  # joined with the contribute-only key
        self.udp_addr: tuple[str, int] | None = None
        self.lock = asyncio.Lock()

    async def send(self, header: dict, payload: bytes = b"") -> None:
        async with self.lock:  # parts of one message must not interleave with another's
            for part in split(pack(header, payload)):
                await self.ws.send_bytes(part)


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
    def __init__(self, secret: str, host: str = "0.0.0.0", port: int = 8000, model=None,
                 models_dir: str | None = None, hf_token: str | None = None):
        """model: a DictSource/HFSource, a path to a .emm file, or None (no model yet)."""
        self.secret = secret
        self.contribute_key = contribute_key(secret)
        self.cipher = MeshCipher(self.contribute_key)  # device-to-device traffic (see crypto.py)
        self.admin_auth = MeshCipher(secret)
        self.host, self.port = host, port
        self.clients: dict[str, _Client] = {}
        self.endpoint = HubEndpoint(self)
        self.orchestrator = None
        self.model = model
        from .hf import Loader

        self.loader = Loader(Path(models_dir or Path.home() / ".edgemesh" / "models"), hf_token)
        self.lan_origin: str | None = None
        self.public_origin: str | None = None  # set by --public (Cloudflare tunnel) or --public-url
        self.removed: set[str] = set()  # device names taken off the mesh (refused until restart)
        self.chat: list[dict] = []  # the shared conversation every controller sees
        self.chat_version = 0
        self._runner = None
        self._udp = None

    # ---------- lifecycle ----------

    async def start(self) -> None:
        if self.model is not None:
            from .orchestrator import Orchestrator

            source = self.model
            if isinstance(source, str):
                from .model import DictSource, load_model

                source = DictSource(*load_model(source))
            self.orchestrator = Orchestrator(self.endpoint, source, self.workers)
        app = web.Application(client_max_size=1 << 30)
        app.add_routes([
            web.get("/", self._index),
            web.get("/api/hello", self._hello),
            web.get("/contribute", self._contribute_page),
            web.get("/install.ps1", self._install),
            web.get("/install.sh", self._install),
            web.get("/termux.sh", self._termux),
            web.get("/download/edgemesh.zip", self._package),
            web.get("/api/models", self._models),
            web.delete("/api/models", self._delete_model),
            web.post("/api/models/upload", self._upload_model),
            web.get("/api/chat", self._chat_get),
            web.post("/api/chat/clear", self._chat_clear),
            web.get("/ws", self._ws),
            web.get("/api/status", self._status),
            web.post("/api/generate", self._generate),
            web.get("/api/model", self._model_status),
            web.post("/api/model", self._load_model),
            web.get("/static/{name}", self._static),
            web.get("/api/qr.svg", self._qr),
            web.post("/api/devices/{node}/limits", self._set_limits),
            web.post("/api/devices/{node}/remove", self._remove_device),
            web.get("/api/estimate", self._estimate),
        ])
        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()
        try:
            site = web.TCPSite(self._runner, self.host, self.port)
            await site.start()
            self.port = site._server.sockets[0].getsockname()[1]
            loop = asyncio.get_running_loop()
            self._udp, _ = await loop.create_datagram_endpoint(lambda: _Reflector(self), local_addr=(self.host, self.port))
        except OSError:  # port taken: free what we opened so the caller can try another port
            await self._runner.cleanup()
            self._runner = None
            raise
        log.info("hub listening on tcp+udp port %d", self.port)
        self._pinger = asyncio.ensure_future(self._ping_loop())

    async def close(self) -> None:
        if getattr(self, "_pinger", None):
            self._pinger.cancel()
        if self._udp:
            self._udp.close()
        for c in list(self.clients.values()):
            await c.ws.close()
        if self._runner:
            await self._runner.cleanup()

    async def _ping_loop(self) -> None:
        """Measure each device's round trip to the hub, for speed estimates."""
        while True:
            for c in list(self.clients.values()):
                t0 = asyncio.get_running_loop().time()
                try:
                    await self.endpoint.request(c.node_id, {"op": "ping"}, timeout=5)
                    c.info["rtt_ms"] = round((asyncio.get_running_loop().time() - t0) * 1000, 1)
                except Exception:
                    pass
            await asyncio.sleep(10)

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
                msg = f"{nonce}:{node_id}".encode()
                admin = self.admin_auth.verify(msg, str(hello["mac"]))
                ok = admin or self.cipher.verify(msg, str(hello["mac"]))
                native = True
            else:  # browser: presents the key from its join link
                given = str(hello.get("key", ""))
                admin = hmac.compare_digest(given, self.secret)
                ok = admin or hmac.compare_digest(given, self.contribute_key)
                native = False
            if node_id in self.removed and not admin:
                await ws.send_bytes(pack({"type": "error", "error": "removed from this mesh by the main computer"}))
                return ws
            if not node_id or node_id == HUB_ID or not ok:
                log.warning("%s (%s) tried to join with a key that isn't this mesh's", node_id or "a device", request.remote)
                await ws.send_bytes(pack({"type": "error", "error": "wrong mesh key"}))
                return ws
            old = self.clients.get(node_id)
            if old is not None and not old.contributor and not admin:  # contributors can't take over a device
                await ws.send_bytes(pack({"type": "error", "error": "that device name is taken"}))
                return ws
            if old is not None:
                await old.ws.close()
            client = _Client(node_id, ws, dict(hello.get("info", {})), native, contributor=not admin)
            client.info["contributor"] = not admin
            self.clients[node_id] = client
            await client.send({"type": "welcome", "udp_port": self.port, "observed_ip": request.remote})
            log.info("%s joined: %s %s", node_id, client.info.get("kind"), client.info.get("platform", ""))
            joiner = Joiner()
            async for msg in ws:
                if msg.type == WSMsgType.BINARY:
                    data = joiner.feed(msg.data)
                    if data is not None:
                        header, payload = unpack(data)
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
        elif kind == "info":  # a device's limits or status changed
            fields = header.get("info") or {}
            client.info.update({k: v for k, v in fields.items() if k in INFO_FIELDS})
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

    def _may_contribute(self, request: web.Request) -> bool:
        key = request.query.get("c") or request.query.get("k", "")
        return hmac.compare_digest(key, self.contribute_key) or hmac.compare_digest(key, self.secret)

    async def _index(self, request: web.Request) -> web.Response:
        return web.Response(text=_web_file("index.html"), content_type="text/html")

    async def _hello(self, request: web.Request) -> web.Response:
        """No key needed: lets a second `edgemesh host` see that EdgeMesh already runs on a port."""
        return web.json_response({"edgemesh": True, "mesh": hashlib.sha256(self.secret.encode()).hexdigest()[:12],
                                  "repo": project.REPO, "raw": project.RAW, "pages": project.PAGES})

    async def _contribute_page(self, request: web.Request) -> web.Response:
        return web.Response(text=_web_file("contribute.html"), content_type="text/html")

    def _origin(self, request: web.Request) -> str:
        proto = request.headers.get("X-Forwarded-Proto") or request.scheme
        return f"{proto}://{request.host}"

    async def _install(self, request: web.Request) -> web.Response:
        """One-line installers: they set up EdgeMesh and start contributing in the background."""
        if not self._may_contribute(request):
            raise web.HTTPUnauthorized(text="wrong key")
        name = "install.ps1" if request.path.endswith(".ps1") else "install.sh"
        text = (resources.files("edgemesh").joinpath("installers", name).read_text(encoding="utf-8")
                .replace("@ORIGIN@", self._origin(request)).replace("@KEY@", self.contribute_key))
        return web.Response(text=text, content_type="text/plain")

    async def _termux(self, request: web.Request) -> web.Response:
        """Android phones in Termux: ?k= (full key) moves hosting there with the same mesh key;
        ?c= (contribute key) makes the phone a native worker, which computes faster than a browser."""
        if self._authorized(request):
            key, mode = self.secret, "host"
        elif hmac.compare_digest(request.query.get("c", ""), self.contribute_key):
            key, mode = self.contribute_key, "contribute"
        else:
            raise web.HTTPUnauthorized(text="wrong key")
        text = (resources.files("edgemesh").joinpath("installers", "termux-host.sh").read_text(encoding="utf-8")
                .replace("@ORIGIN@", self._origin(request)).replace("@KEY@", key).replace("@MODE@", mode))
        return web.Response(text=text, content_type="text/plain")

    async def _package(self, request: web.Request) -> web.Response:
        """This EdgeMesh version as a pip-installable zip, for the installers."""
        if not self._may_contribute(request):
            raise web.HTTPUnauthorized(text="wrong key")
        return web.Response(body=await asyncio.to_thread(_package_zip), content_type="application/zip")

    async def _models(self, request: web.Request) -> web.Response:
        if not self._authorized(request):
            raise web.HTTPUnauthorized(text="wrong mesh key")
        from .hf import downloaded

        current = self.orchestrator.cfg.get("name") if self.orchestrator else None
        models = await asyncio.to_thread(downloaded, self.loader.cache)
        return web.json_response({"current": current, "models": [{**m, "loaded": m["repo"] == current} for m in models]})

    async def _upload_model(self, request: web.Request) -> web.Response:
        """POST ?name=x.gguf&offset=N[&done=1], body = the next piece of the file. Pieces keep each
        request small enough for tunnels (Cloudflare allows 100 MB per request); a failed upload
        resumes from the size the hub reports."""
        if not self._authorized(request):
            raise web.HTTPUnauthorized(text="wrong mesh key")
        name = Path(request.query.get("name", "")).name
        if not name.lower().endswith(".gguf") or name.startswith("."):
            raise web.HTTPBadRequest(text="only .gguf files can be uploaded")
        folder = self.loader.cache / "gguf"
        folder.mkdir(parents=True, exist_ok=True)
        part = folder / (name + ".part")
        have = part.stat().st_size if part.exists() else 0
        offset = int(request.query.get("offset", "0"))
        if offset == 0 and have:
            part.unlink()
            have = 0
        if offset != have:
            return web.json_response({"have": have}, status=409)
        with open(part, "ab") as f:
            async for chunk in request.content.iter_chunked(1 << 20):
                f.write(chunk)
        size = part.stat().st_size
        if request.query.get("done") == "1":
            target = folder / name
            os.replace(part, target)
            return web.json_response({"have": size, "repo": f"gguf:{target.resolve()}"})
        return web.json_response({"have": size})

    async def _delete_model(self, request: web.Request) -> web.Response:
        if not self._authorized(request):
            raise web.HTTPUnauthorized(text="wrong mesh key")
        from .hf import remove

        repo = request.query.get("repo", "")
        if self.orchestrator is not None and self.orchestrator.cfg.get("name") == repo:
            raise web.HTTPConflict(text="that model is in use; load another one first")
        d = self.loader.current
        if d is not None and d.repo == repo and d.state in ("starting", "downloading", "loading"):
            raise web.HTTPConflict(text="that model is still downloading")
        try:
            freed = await asyncio.to_thread(remove, self.loader.cache, repo)
        except FileNotFoundError as e:
            raise web.HTTPNotFound(text=str(e))
        log.info("removed %s (%.1f GB)", repo, freed / 2**30)
        return web.json_response({"freed": freed})

    async def _chat_get(self, request: web.Request) -> web.Response:
        if not self._authorized(request):
            raise web.HTTPUnauthorized(text="wrong mesh key")
        return web.json_response({"version": self.chat_version, "messages": self.chat})

    async def _chat_clear(self, request: web.Request) -> web.Response:
        if not self._authorized(request):
            raise web.HTTPUnauthorized(text="wrong mesh key")
        self.chat = []
        self.chat_version += 1
        return web.json_response({"version": self.chat_version})

    async def _static(self, request: web.Request) -> web.Response:
        name = request.match_info["name"]
        types = {"app.js": "application/javascript", "worker.js": "application/javascript",
                 "common.js": "application/javascript", "contribute.js": "application/javascript",
                 "gpu.js": "application/javascript", "beacon.js": "application/javascript",
                 "model.js": "application/javascript", "pool.js": "application/javascript", "style.css": "text/css",
                 "icon.svg": "image/svg+xml", "manifest.webmanifest": "application/manifest+json"}
        if name not in types:
            raise web.HTTPNotFound()
        return web.Response(text=_web_file(name), content_type=types[name])

    async def _qr(self, request: web.Request) -> web.Response:
        """QR code (SVG) for the join link, so phones can scan it from the hub's screen."""
        if not self._authorized(request):
            raise web.HTTPUnauthorized(text="wrong mesh key")
        import io

        import segno

        buf = io.BytesIO()
        segno.make(request.query.get("url", ""), error="m").save(buf, kind="svg", scale=6, border=2,
                                                               dark="#111", light="#fff")
        return web.Response(body=buf.getvalue(), content_type="image/svg+xml")

    async def _estimate(self, request: web.Request) -> web.Response:
        """GET ?repo=org/name (or nothing for the current model) -> speed / memory / disk estimate."""
        if not self._authorized(request):
            raise web.HTTPUnauthorized(text="wrong mesh key")
        from .estimate import estimate

        repo = request.query.get("repo", "").strip().removeprefix("https://huggingface.co/").strip("/")
        if repo.startswith("gguf:"):
            from .gguf_source import local_gguf_config

            try:
                cfg, disk = await asyncio.to_thread(local_gguf_config, repo[5:])
            except (OSError, ValueError) as e:
                return web.json_response({"problem": str(e)})
        elif repo:
            from .hf import remote_config

            try:
                cfg, disk = await asyncio.to_thread(remote_config, repo, request.query.get("token") or self.loader.token)
            except (ValueError, PermissionError, FileNotFoundError) as e:
                return web.json_response({"problem": str(e)})
            except Exception as e:
                return web.json_response({"problem": f"could not reach Hugging Face: {e}"})
        elif self.orchestrator is not None:
            cfg = self.orchestrator.cfg
            folder = getattr(self.orchestrator.source, "folder", None)
            disk = sum(f.stat().st_size for f in folder.iterdir() if f.is_file()) if folder else None
        else:
            raise web.HTTPNotFound(text="no model loaded")
        workers = [{"node_id": c.node_id, **c.info, "direct": c.info.get("direct", c.native)} for c in self.clients.values()]
        return web.json_response(estimate(cfg, workers, disk))

    async def _set_limits(self, request: web.Request) -> web.Response:
        """POST {"cpu", "schedule"} -> forwarded to the device, which stores and applies it."""
        if not self._authorized(request):
            raise web.HTTPUnauthorized(text="wrong mesh key")
        from .limits import validate

        node = request.match_info["node"]
        if node not in self.clients:
            raise web.HTTPNotFound(text="device not connected")
        try:
            limits = validate(await request.json())
            rh, _ = await self.endpoint.request(node, {"op": "set_limits", "limits": limits}, timeout=15)
        except (ValueError, KeyError) as e:
            raise web.HTTPBadRequest(text=str(e))
        except (RuntimeError, asyncio.TimeoutError) as e:
            raise web.HTTPBadGateway(text=f"device did not accept the limits: {e}")
        return web.json_response(rh.get("limits", limits))

    async def _remove_device(self, request: web.Request) -> web.Response:
        """Take a device off the mesh: it stops, removes its autostart and, if the one-line
        installer set it up, uninstalls EdgeMesh. Browsers forget the link."""
        if not self._authorized(request):
            raise web.HTTPUnauthorized(text="wrong mesh key")
        node = request.match_info["node"]
        client = self.clients.get(node)
        if client is None:
            raise web.HTTPNotFound(text="device not connected")
        if client.info.get("hub_machine"):
            raise web.HTTPBadRequest(text="that's the main computer itself; stop it there")
        self.removed.add(node)
        try:
            rh, _ = await self.endpoint.request(node, {"op": "uninstall"}, timeout=10)
        except (RuntimeError, asyncio.TimeoutError) as e:
            rh = {"ok": False, "error": str(e)}
        await client.ws.close()
        log.info("removed %s from the mesh: %s", node, rh)
        return web.json_response(rh)

    async def _status(self, request: web.Request) -> web.Response:
        if not self._authorized(request):
            raise web.HTTPUnauthorized(text="wrong mesh key")
        o = self.orchestrator
        return web.json_response({
            "model": o.cfg if o else None,
            "lan_origin": self.lan_origin,
            "public_origin": self.public_origin,
            "contribute_key": self.contribute_key,
            "chat_version": self.chat_version,
            "capacity": _capacity([c.info for c in self.clients.values()]),
            "plan": [list(p) for p in o.plan] if o else [],
            "busy": bool(o and o._lock.locked()),
            "last_run": o.last_run if o else None,
            "devices": [{"node_id": c.node_id, "kind": c.info.get("kind"), "role": c.info.get("role"),
                         "platform": c.info.get("platform"), "score": c.info.get("score"),
                         "mem_mb": c.info.get("mem_mb"), "public_udp": c.udp_addr, "direct": c.info.get("direct"),
                         "limit": c.info.get("limit", 100), "limit_reason": c.info.get("limit_reason"),
                         "limits": c.info.get("limits"), "is_hub_machine": c.info.get("hub_machine", False),
                         "contributor": c.contributor, "rtt_ms": c.info.get("rtt_ms"), "cores": c.info.get("cores"),
                         "cores_total": c.info.get("cores_total"),
                         "power": c.info.get("power"), "uses": c.info.get("uses"), "gpu": c.info.get("gpu"),
                         "helped": o.helped.get(c.node_id) if o else None}
                        for c in self.clients.values()],
        })

    async def _generate(self, request: web.Request) -> web.StreamResponse:
        """POST {"prompt" or "messages", "max_new", "temperature", "chat"} -> newline-delimited JSON events.
        Closing the request (the Stop button) ends the generation."""
        if not self._authorized(request):
            raise web.HTTPUnauthorized(text="wrong mesh key")
        if self.orchestrator is None:
            raise web.HTTPServiceUnavailable(text="this hub was started without a model")
        body = await request.json()
        resp = web.StreamResponse(headers={"Content-Type": "application/x-ndjson", "Cache-Control": "no-cache"})
        await resp.prepare(request)
        record = body.get("record")  # {"user": text, "by": device}: add to the shared conversation
        reply = None
        if isinstance(record, dict):
            self.chat.append({"role": "user", "content": str(record.get("user", "")), "by": str(record.get("by", ""))[:64]})
            reply = {"role": "assistant", "content": "", "model": self.orchestrator.cfg.get("name"), "live": True}
            self.chat.append(reply)
            self.chat = self.chat[-200:]
            self.chat_version += 1
        try:
            messages = body.get("messages")
            prompt = messages if isinstance(messages, list) else str(body.get("prompt", ""))
            gen = self.orchestrator.generate(prompt, int(body.get("max_new", 32)), float(body.get("temperature", 0.0)),
                                             body.get("seed"), bool(body.get("chat", False)), body.get("context"))
            try:
                async for ev in gen:
                    if reply is not None and ev["type"] in ("token", "done"):
                        reply["content"] = ev["text"]
                        self.chat_version += 1
                        if ev["type"] == "done":
                            reply.pop("live", None)
                            reply["stats"] = ev["stats"]
                    await resp.write((json.dumps(ev) + "\n").encode())
            finally:
                await gen.aclose()
                if reply is not None and reply.pop("live", None):
                    reply["stopped"] = True
                    self.chat_version += 1
        except ConnectionResetError:
            log.info("client stopped the generation")
            return resp
        except Exception as e:
            log.warning("generation failed: %s", e)
            await resp.write((json.dumps({"type": "error", "error": str(e)}) + "\n").encode())
        await resp.write_eof()
        return resp


    async def _model_status(self, request: web.Request) -> web.Response:
        if not self._authorized(request):
            raise web.HTTPUnauthorized(text="wrong mesh key")
        o = self.orchestrator
        d = self.loader.current
        return web.json_response({
            "model": o.cfg if o else None,
            "chat_template": bool(o and o.tokenizer.has_chat_template),
            "download": d.as_dict() if d else None,
        })

    async def _load_model(self, request: web.Request) -> web.Response:
        """POST {"repo": "org/name", "token": optional HF token} -> starts download + switch."""
        if not self._authorized(request):
            raise web.HTTPUnauthorized(text="wrong mesh key")
        body = await request.json()
        repo = str(body.get("repo", "")).strip().removeprefix("https://huggingface.co/").strip("/")
        if repo.startswith("gguf:") or repo.lower().endswith(".gguf"):  # a GGUF file on this computer
            path = Path(repo.removeprefix("gguf:").strip('"')).expanduser()
            if path.suffix.lower() != ".gguf" or not path.is_file():
                raise web.HTTPBadRequest(text=f"no .gguf file at {path} on the main computer")
            repo = f"gguf:{path.resolve()}"
        elif split_spec(repo)[0].count("/") != 1:
            raise web.HTTPBadRequest(text="use the Hugging Face repo id, like Qwen/Qwen2.5-0.5B-Instruct, "
                                          "or the path of a .gguf file on the main computer")
        loop = asyncio.get_running_loop()

        def ready(source):
            done = asyncio.run_coroutine_threadsafe(self._switch(source), loop)
            done.result(timeout=600)

        try:
            d = self.loader.start(repo, body.get("token") or None, ready)
        except RuntimeError as e:
            raise web.HTTPConflict(text=str(e))
        return web.json_response(d.as_dict())

    async def _switch(self, source) -> None:
        from .orchestrator import Orchestrator

        if self.orchestrator is None:
            self.orchestrator = Orchestrator(self.endpoint, source, self.workers)
        else:
            async with self.orchestrator._lock:  # wait for a running generation to finish
                self.orchestrator.set_model(source)
        log.info("now serving %s", source.cfg["name"])


def _candidates(c: _Client) -> list:
    """Public (server-reflexive) mapping first, then the device's own LAN addresses."""
    out = []
    if c.udp_addr:
        out.append(list(c.udp_addr))
    for addr in c.info.get("local_candidates", []):
        if list(addr) not in out:
            out.append(list(addr))
    return out


def _capacity(infos: list[dict]) -> dict:
    """The biggest model the connected devices can hold, and how big a model they can run at a
    given speed (generation reads every weight once per token: ~2 FLOPs per parameter)."""
    workers = [i for i in infos if i.get("role") == "worker" and i.get("limit", 100) > 0]
    mem = sum(i.get("mem_mb", 0) * 2**20 / (2 if i.get("uses") == "gpu" else 4) for i in workers)
    flops = sum(i.get("score", 0) * i.get("limit", 100) / 100 for i in workers) * 1e9
    speed = lambda tps: int(min(mem, flops / (2 * tps))) if flops else 0
    return {"max_params": int(mem), "params_at_10": speed(10), "params_at_5": speed(5), "params_at_1": speed(1),
            "gflops": round(flops / 1e9, 1), "devices": len(workers)}


def _package_zip() -> bytes:
    """Zip of the installed edgemesh package plus a pyproject.toml, installable with pip."""
    import io
    import zipfile
    from importlib import metadata

    root = Path(__file__).resolve().parent
    try:
        reqs = [r for r in metadata.requires("edgemesh") or [] if "extra ==" not in r]
        version = metadata.version("edgemesh")
    except metadata.PackageNotFoundError:
        reqs, version = ["numpy", "cryptography", "aiohttp", "tokenizers", "jinja2", "segno", "threadpoolctl"], "0.0.0"
    pyproject = (f'[project]\nname = "edgemesh"\nversion = "{version}"\nrequires-python = ">=3.10"\n'
                 f'dependencies = {json.dumps(reqs)}\n\n[project.scripts]\nedgemesh = "edgemesh.cli:main"\n\n'
                 '[build-system]\nrequires = ["setuptools>=61"]\nbuild-backend = "setuptools.build_meta"\n\n'
                 '[tool.setuptools]\npackages = ["edgemesh"]\n\n[tool.setuptools.package-data]\n'
                 'edgemesh = ["web/*", "installers/*"]\n')
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("edgemesh-src/pyproject.toml", pyproject)
        for f in root.rglob("*"):
            if f.is_file() and "__pycache__" not in f.parts and f.suffix != ".pyc":
                z.write(f, "edgemesh-src/edgemesh/" + f.relative_to(root).as_posix())
    return buf.getvalue()


def _web_file(name: str) -> str:
    return resources.files("edgemesh").joinpath("web", name).read_text(encoding="utf-8")
