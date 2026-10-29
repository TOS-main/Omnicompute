// Browser worker (runs as a Web Worker): joins the mesh over the hub's WebSocket,
// holds a slice of the model, and passes activations along the device chain.
// Same ops as omnicompute/worker.py: load, forward, drop, info.

"use strict";
importScripts("/static/model.js", "/static/gpu.js");

let ws = null, cfgStart = null, shards = {}, stats = { tokens: 0, layers: null, cores: 1, activeCores: 1, gpu: null, power: null };
let power = null; // "charging" | "battery" | null (reported by the page)

// ---- multi-core pool: each core holds a slice of every big weight matrix ----

class Pool {
  constructor(n) {
    this.workers = Array.from({ length: n }, () => new Worker("/static/pool.js"));
    this.pending = new Map();
    this.seq = 0;
    this.slices = new Map(); // weight name -> [{worker, c0, c1}] column (or row) ranges
    this.active = n; // cores allowed to work at once (usage limits can lower it)
    for (const w of this.workers) w.onmessage = (ev) => { const p = this.pending.get(ev.data.seq); this.pending.delete(ev.data.seq); p(ev.data); };
  }

  call(w, msg, transfer) {
    const seq = ++this.seq;
    return new Promise((resolve) => { this.pending.set(seq, resolve); w.postMessage({ ...msg, seq }, transfer || []); });
  }

  has(name) { return this.slices.has(name); }

  async bench() {
    const r = await Promise.all(this.workers.map((w) => this.call(w, { op: "bench" })));
    return r.reduce((a, b) => a + b.gflops, 0);
  }

  async clear() { this.slices.clear(); await Promise.all(this.workers.map((w) => this.call(w, { op: "clear" }))); }

  // split W[K,N] by columns across the cores
  async putMatrix(name, t) {
    const [K, N] = t.shape, n = this.workers.length, parts = [], jobs = [];
    for (let j = 0; j < n; j++) {
      const c0 = Math.floor((N * j) / n), c1 = Math.floor((N * (j + 1)) / n), w = c1 - c0;
      if (w <= 0) continue;
      const data = new Float32Array(K * w);
      for (let k = 0; k < K; k++) data.set(t.data.subarray(k * N + c0, k * N + c1), k * w);
      parts.push({ worker: this.workers[j], c0, c1 });
      jobs.push(this.call(this.workers[j], { op: "put", id: name, layout: "cols", K, N: w, data }, [data.buffer]));
    }
    await Promise.all(jobs);
    this.slices.set(name, { parts, N, layout: "cols" });
  }

  // split a tied embedding E[V,D] by rows, for logits = x . E^T
  async putRows(name, t) {
    const [V, D] = t.shape, n = this.workers.length, parts = [], jobs = [];
    for (let j = 0; j < n; j++) {
      const r0 = Math.floor((V * j) / n), r1 = Math.floor((V * (j + 1)) / n);
      if (r1 <= r0) continue;
      const data = t.data.slice(r0 * D, r1 * D);
      parts.push({ worker: this.workers[j], c0: r0, c1: r1 });
      jobs.push(this.call(this.workers[j], { op: "put", id: name, layout: "rows", K: D, R: r1 - r0, data }, [data.buffer]));
    }
    await Promise.all(jobs);
    this.slices.set(name, { parts, N: V, layout: "rows" });
  }

  async mm(name, x, T) {
    const { parts, N, layout } = this.slices.get(name);
    const outs = [];
    for (let i = 0; i < parts.length; i += this.active) { // at most `active` cores busy at once
      const batch = parts.slice(i, i + this.active);
      outs.push(...await Promise.all(batch.map((p) => this.call(p.worker, { op: "mm", id: name, x, T }))));
    }
    const y = new Float32Array(T * N);
    parts.forEach((p, j) => {
      const o = outs[j].out, w = p.c1 - p.c0;
      if (layout === "rows") y.set(o, p.c0);
      else for (let t = 0; t < T; t++) y.set(o.subarray(t * w, (t + 1) * w), t * N + p.c0);
    });
    return y;
  }
}

let pool = null, cpuPool = null, gpuPool = null, cpuScore = 0, gpuScore = 0;

// GPU on, allowed by the limits, and faster than the CPU cores (or the CPU is switched off)
const wantGpu = () => !!gpuPool && !gpuPool.lost && !gpuBroken && limits.gpu !== false
  && (limits.use_cpu === false || gpuScore > cpuScore || (cfgStart && cfgStart.forceGpu));

// Move a shard's big matrices onto the pool (GPU or CPU cores); the main copy is dropped to save memory.
// GPUs and drivers differ; after uploading a matrix, multiply a test vector on both the GPU and
// the CPU and make sure they agree. Any mismatch => this device stays on the CPU.
let gpuBroken = false;
async function gpuAgrees(name, t, rows) {
  const K = rows ? t.shape[1] : t.shape[0], N = rows ? t.shape[0] : t.shape[1];
  const x = new Float32Array(K);
  for (let i = 0; i < K; i++) x[i] = Math.sin(i * 12.9898) * 0.5;
  const y = await gpuPool.mm(name, x, 1);
  const n = Math.min(N, rows ? 64 : 256);
  let worst = 0, scale = 1e-3;
  for (let j = 0; j < n; j++) {
    let ref = 0;
    if (rows) for (let k = 0; k < K; k++) ref += x[k] * t.data[j * K + k];
    else for (let k = 0; k < K; k++) ref += x[k] * t.data[k * N + j];
    worst = Math.max(worst, Math.abs(ref - y[j]));
    scale = Math.max(scale, Math.abs(ref));
  }
  return Number.isFinite(worst) && worst <= (gpuPool.bits === 8 ? 0.03 : 0.01) * scale + 1e-3; // 8-bit weights round a little
}

async function distribute(shard) {
  if (gpuPool) await gpuPool.clear();
  if (cpuPool) await cpuPool.clear();
  pool = wantGpu() && !gpuBroken ? gpuPool : cpuPool;
  stats.gpu = pool && pool === gpuPool ? gpuPool.name : null;
  if (!pool) return;
  const big = Object.entries(shard.w).filter(([name, t]) => t.shape.length === 2 && /\.w$|^lm_head$/.test(name));
  for (const [name, t] of big) {
    await pool.putMatrix(name, t);
    if (pool === gpuPool && !(await gpuAgrees(name, t, false))) return gpuFailed(shard, name);
  }
  const tied = shard.last && !shard.w["lm_head"] ? (shard.llama || shard.gemma ? "embed" : "wte") : null;
  if (tied) {
    await pool.putRows(tied + "^T", shard.w[tied]);
    if (pool === gpuPool && !(await gpuAgrees(tied + "^T", shard.w[tied], true))) return gpuFailed(shard, tied);
  }
  for (const [name] of big) delete shard.w[name].data; // the pool holds them now
  if (tied && !shard.first) shard.w[tied] = { shape: shard.w[tied].shape }; // only needed for the head
  shard.pool = pool;
}

async function gpuFailed(shard, name) {
  gpuBroken = true;
  status("ready", { note: `this GPU gave wrong results (${name}); using the CPU instead` });
  reportLimits(true);
  return distribute(shard);
}

// load / forward / drop run one at a time, in arrival order, so a new model never replaces
// weights while a step is still using them
let opQueue = Promise.resolve();
const SERIAL_OPS = new Set(["load", "unload", "forward", "drop"]);

// ---- usage limits (same rules as omnicompute/limits.py) ----

let limits = { cpu: 80, schedule: [] };
const minutes = (t) => { const [h, m] = t.split(":").map(Number); return h * 60 + m; };

function effectiveLimit() {
  const d = new Date(), now = d.getHours() * 60 + d.getMinutes();
  let cpu = limits.cpu ?? 80, why = "default", inWindow = false;
  for (const w of limits.schedule || []) {
    const a = minutes(w.from), b = minutes(w.to);
    if (a < b ? now >= a && now < b : now >= a || now < b) { cpu = w.cpu; why = `schedule ${w.from}-${w.to}`; inWindow = true; break; }
  }
  if (!inWindow && power === "charging" && limits.charging) { cpu = limits.charging.cpu; why = "charging"; }
  if (power === "battery" && limits.battery && limits.battery.cpu < cpu) { cpu = limits.battery.cpu; why = "on battery"; }
  const g = limits.gpu_limit;  // separate GPU limit (null = same as the CPU's); windows and battery still cap it
  if (g != null && wantGpu()) return why === "default" ? [g, "GPU limit"] : [Math.min(g, cpu), why];
  return [cpu, why];
}

function effectiveCores() {
  let cores = limits.cores || null;
  if (power === "charging" && limits.charging && limits.charging.cores) cores = limits.charging.cores;
  if (power === "battery" && limits.battery && limits.battery.cores) cores = Math.min(cores || limits.battery.cores, limits.battery.cores);
  return cores;
}

function engineInfo() {
  const gpu = wantGpu();
  if (cpuPool) cpuPool.active = Math.max(1, Math.min(cpuPool.workers.length, effectiveCores() || cpuPool.workers.length));
  stats.activeCores = cpuPool ? cpuPool.active : 1;
  stats.power = power;
  return { score: Math.round((gpu ? gpuScore : cpuScore * stats.activeCores / stats.cores) * 100) / 100,
           cpu_score: Math.round(cpuScore * 100) / 100, cores: stats.activeCores, cores_total: stats.cores, uses: gpu ? "gpu" : "cpu", power,
           gpu: gpuPool ? { name: gpuPool.name, score: Math.round(gpuScore * 100) / 100 } : null };
}

let reported = "";
function reportLimits(force) {
  const [cpu, why] = effectiveLimit(), info = { limit: cpu, limit_reason: why, limits, ...engineInfo() };
  const key = JSON.stringify(info);
  if ((force || key !== reported) && ws && ws.readyState === 1) {
    reported = key;
    wsSend(packMsg({ type: "info", info }));
  }
}
setInterval(() => reportLimits(false), 20000);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const status = (state, extra = {}) => postMessage({ state, stats, ...extra });

// big messages travel in 1 MB parts (see omnicompute/wsparts.py): proxies and tunnels cap message size
const PART = 1 << 20, MAGIC = [69, 77, 67, 72]; // "EMCH"
function wsSend(u8) {
  if (u8.length <= PART) { ws.send(u8); return; }
  const n = Math.ceil(u8.length / PART), id = crypto.getRandomValues(new Uint8Array(8));
  for (let i = 0; i < n; i++) {
    const chunk = u8.subarray(i * PART, (i + 1) * PART), m = new Uint8Array(20 + chunk.length), dv = new DataView(m.buffer);
    m.set(MAGIC); m.set(id, 4); dv.setUint32(12, i, false); dv.setUint32(16, n, false); m.set(chunk, 20);
    ws.send(m);
  }
}
const joining = new Map();
function joinParts(u8) {
  if (!(u8[0] === 69 && u8[1] === 77 && u8[2] === 67 && u8[3] === 72)) return u8;
  const dv = new DataView(u8.buffer, u8.byteOffset), id = Array.from(u8.subarray(4, 12)).join(",");
  const i = dv.getUint32(12, false), n = dv.getUint32(16, false);
  if (!joining.has(id)) joining.set(id, []);
  const parts = joining.get(id);
  parts[i] = u8.slice(20);
  if (parts.filter(Boolean).length < n) return null;
  joining.delete(id);
  const out = new Uint8Array(parts.reduce((a, p) => a + p.length, 0));
  let o = 0;
  for (const p of parts) { out.set(p, o); o += p.length; }
  return out;
}

function sendTo(peer, header, payload) {
  wsSend(packMsg({ type: "relay", to: peer }, packMsg(header, payload)));
}

const CONTROL_OPS = new Set(["load", "unload", "set_limits", "drop", "preconnect", "info", "credit", "live", "uninstall"]);

async function handle(from, header, payload) {
  const op = header.op;
  if (CONTROL_OPS.has(op) && from !== "hub") throw new Error(`${op} is only accepted from the hub`);
  if (op === "credit") { postMessage({ state: "credit", credit: header }); return null; }
  if (op === "uninstall") { // removed by the main computer: stop and forget the link
    cfgStart.stop = true;
    postMessage({ state: "removed" });
    setTimeout(() => ws && ws.close(), 200);
    return [{ ok: true, removed: ["browser"] }, null];
  }
  if (op === "unload") { shards = {}; stats.layers = null; status("ready", { note: "no model loaded" }); return null; }
  if (op === "live") { postMessage({ state: "live", text: header.text, done: header.done }); return null; }
  if (op === "load") {
    const t = decodeTensors(payload);
    shards = {};
    const shard = new Shard(header.cfg, t, header.start, header.end);
    await distribute(shard);
    shards = { [header.model]: shard };
    stats.layers = [header.start, header.end];
    status("ready", { note: `holding layers ${header.start}-${header.end} of ${header.cfg.name}` });
    reportLimits(true);
    return [{ ok: true }, null];
  }
  if (op === "forward") {
    try {
      const hop = header.route[header.hop], shard = shards[header.model];
      if (!shard || shard.start !== hop.start || shard.end !== hop.end) throw new Error("shard not loaded here");
      const [cpu] = effectiveLimit();
      if (cpu <= 0) throw new Error("paused by its usage limit");
      const t0 = performance.now();
      const out = await shard.forwardEx(header.job, decodeTensors(payload), header.pos);
      const busy = (performance.now() - t0) / 1000;
      stats.tokens++;
      const timings = [...(header.timings || []), [cfgStart.name, busy]];
      if (cpu < 100) await sleep(busy * (100 / cpu - 1) * 1000); // duty cycle down to the limit
      if (header.hop + 1 < header.route.length) {
        sendTo(header.route[header.hop + 1].node, { ...header, hop: header.hop + 1, kind: "msg", timings }, encodeTensors(out));
      } else {
        sendTo(header.reply_to, { op: "result", req: header.req, kind: "msg", timings }, encodeTensors({ logits: out.x }));
      }
      if (stats.tokens % 5 === 0) status("ready");
    } catch (e) {
      sendTo(header.reply_to, { op: "error", req: header.req, error: String(e.message || e), node: cfgStart.name, kind: "msg" });
    }
    return null;
  }
  if (op === "ping") return [{ pong: true }, null];
  if (op === "set_limits") {
    limits = header.limits;
    postMessage({ state: "limits", limits }); // the page stores them for next time
    reportLimits(true);
    return [{ limits }, null];
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
    if (typeof ev.data === "string") return;
    const whole = joinParts(new Uint8Array(ev.data));
    if (!whole) return;
    const [h, payload] = unpackMsg(whole);
    if (h.type === "challenge") {
      const [cpu, why] = effectiveLimit();
      ws.send(JSON.stringify({ node_id: c.name, key: c.key, info: { ...c.info, limit: cpu, limit_reason: why, limits, ...engineInfo() } }));
    } else if (h.type === "welcome") {
      welcomed = true;
      status("ready", { note: "connected, waiting for the hub to assign layers" });
    } else if (h.type === "error") {
      status("error", { note: h.error });
      c.stop = true;
    } else if (h.type === "relay") {
      const [mh, mp] = unpackMsg(payload);
      const run = async () => {
        let res;
        try { res = await handle(h.from, mh, mp); } catch (e) { res = [{ error: String(e.message || e) }, null]; }
        if (mh.kind === "req" && res) sendTo(h.from, { ...res[0], rpc: mh.rpc, kind: "resp" }, res[1]);
      };
      if (SERIAL_OPS.has(mh.op)) opQueue = opQueue.then(run, run); else run();
    }
  };
  ws.onclose = () => {
    shards = {}; stats.layers = null;
    if (c.stop) { status(welcomed ? "stopped" : "error", { note: welcomed ? "" : "could not connect" }); return; }
    status("connecting", { note: "connection lost, retrying..." });
    setTimeout(connect, 3000);
  };
}

onmessage = async (ev) => {
  if (ev.data.cmd === "start") {
    if (ev.data.limits) limits = ev.data.limits;
    status("connecting", { note: "measuring this device's speed" });
    const cores = typeof Worker === "undefined" ? 1 : Math.max(1, Math.min(16, (navigator.hardwareConcurrency || 2) - 1));
    if (cores > 1) { cpuPool = new Pool(cores); cpuScore = await cpuPool.bench(); } else cpuScore = benchmark();
    stats.cores = cores;
    gpuPool = await GpuPool.open();
    if (gpuPool) { try { gpuScore = await gpuPool.bench(); } catch { gpuPool = null; } }
    cfgStart = { ...ev.data };
    connect();
  } else if (ev.data.cmd === "power") {
    power = ev.data.power;
    reportLimits(false);
  } else if (ev.data.cmd === "limits") {
    limits = ev.data.limits;
    reportLimits(true);
  } else if (ev.data.cmd === "stop" && cfgStart) {
    cfgStart.stop = true;
    if (ws) ws.close();
  }
};
