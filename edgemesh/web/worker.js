// Browser worker (runs as a Web Worker): joins the mesh over the hub's WebSocket,
// holds a slice of the model, and passes activations along the device chain.
// Same ops as edgemesh/worker.py: load, forward, drop, info.

"use strict";
importScripts("/static/model.js");

let ws = null, cfgStart = null, shards = {}, stats = { tokens: 0, layers: null };

const status = (state, extra = {}) => postMessage({ state, stats, ...extra });

function sendTo(peer, header, payload) {
  ws.send(packMsg({ type: "relay", to: peer }, packMsg(header, payload)));
}

async function handle(from, header, payload) {
  const op = header.op;
  if (op === "load") {
    const t = decodeTensors(payload);
    shards = { [header.model]: new Shard(header.cfg, t, header.start, header.end) };
    stats.layers = [header.start, header.end];
    status("ready", { note: `holding layers ${header.start}-${header.end} of ${header.cfg.name}` });
    return [{ ok: true }, null];
  }
  if (op === "forward") {
    try {
      const hop = header.route[header.hop], shard = shards[header.model];
      if (!shard || shard.start !== hop.start || shard.end !== hop.end) throw new Error("shard not loaded here");
      const y = shard.forward(header.job, decodeTensors(payload)["x"], header.pos);
      stats.tokens++;
      if (header.hop + 1 < header.route.length) {
        sendTo(header.route[header.hop + 1].node, { ...header, hop: header.hop + 1, kind: "msg" }, encodeTensors({ x: y }));
      } else {
        sendTo(header.reply_to, { op: "result", req: header.req, kind: "msg" }, encodeTensors({ logits: y }));
      }
      if (stats.tokens % 5 === 0) status("ready");
    } catch (e) {
      sendTo(header.reply_to, { op: "error", req: header.req, error: String(e.message || e), node: cfgStart.name, kind: "msg" });
    }
    return null;
  }
  if (op === "preconnect") return null; // browsers always use the relay in this version
  if (op === "drop") { const s = shards[header.model]; if (s) s.drop(header.job); return null; }
  if (op === "info") return [{ shards: Object.fromEntries(Object.entries(shards).map(([m, s]) => [m, [s.start, s.end]])) }, null];
  throw new Error(`unknown op ${op}`);
}

function connect() {
  const c = cfgStart;
  status("connecting");
  ws = new WebSocket(c.wsUrl);
  ws.binaryType = "arraybuffer";
  let welcomed = false;
  ws.onmessage = async (ev) => {
    const [h, payload] = unpackMsg(new Uint8Array(ev.data));
    if (h.type === "challenge") {
      ws.send(JSON.stringify({ node_id: c.name, key: c.key, info: c.info }));
    } else if (h.type === "welcome") {
      welcomed = true;
      status("ready", { note: "connected, waiting for the hub to assign layers" });
    } else if (h.type === "error") {
      status("error", { note: h.error });
      c.stop = true;
    } else if (h.type === "relay") {
      const [mh, mp] = unpackMsg(payload);
      let res;
      try { res = await handle(h.from, mh, mp); } catch (e) { res = [{ error: String(e.message || e) }, null]; }
      if (mh.kind === "req" && res) sendTo(h.from, { ...res[0], rpc: mh.rpc, kind: "resp" }, res[1]);
    }
  };
  ws.onclose = () => {
    shards = {}; stats.layers = null;
    if (c.stop) { status(welcomed ? "stopped" : "error", { note: welcomed ? "" : "could not connect" }); return; }
    status("connecting", { note: "connection lost, retrying..." });
    setTimeout(connect, 3000);
  };
}

onmessage = (ev) => {
  if (ev.data.cmd === "start") {
    const score = benchmark();
    cfgStart = { ...ev.data, info: { ...ev.data.info, score } };
    connect();
  } else if (ev.data.cmd === "stop" && cfgStart) {
    cfgStart.stop = true;
    if (ws) ws.close();
  }
};
