// GPT-2-architecture layer shard in plain JavaScript: the same math as edgemesh/model.py,
// so a phone browser can host a slice of the model. Tensors are {data: TypedArray, shape: [...]}.

"use strict";

function layerNorm(x, T, D, w, b, out) {
  for (let i = 0; i < T; i++) {
    const o = i * D;
    let mu = 0;
    for (let k = 0; k < D; k++) mu += x[o + k];
    mu /= D;
    let v = 0;
    for (let k = 0; k < D; k++) { const d = x[o + k] - mu; v += d * d; }
    const inv = 1 / Math.sqrt(v / D + 1e-5);
    for (let k = 0; k < D; k++) out[o + k] = (x[o + k] - mu) * inv * w[k] + b[k];
  }
  return out;
}

// out[T,N] = x[T,K] @ W[K,N] + b[N]
function matmul(x, T, K, W, N, b) {
  const out = new Float32Array(T * N);
  const acc = new Float64Array(N);
  for (let i = 0; i < T; i++) {
    if (b) acc.set(b); else acc.fill(0);
    const xo = i * K;
    for (let k = 0; k < K; k++) {
      const xv = x[xo + k];
      if (xv === 0) continue;
      const wo = k * N;
      for (let j = 0; j < N; j++) acc[j] += xv * W[wo + j];
    }
    out.set(acc, i * N);
  }
  return out;
}

function gelu(a) {
  for (let i = 0; i < a.length; i++) {
    const v = a[i];
    a[i] = 0.5 * v * (1 + Math.tanh(0.7978845608028654 * (v + 0.044715 * v * v * v)));
  }
  return a;
}

function rmsNorm(x, T, D, w, eps, out) {
  for (let i = 0; i < T; i++) {
    const o = i * D;
    let ss = 0;
    for (let k = 0; k < D; k++) ss += x[o + k] * x[o + k];
    const inv = 1 / Math.sqrt(ss / D + eps);
    for (let k = 0; k < D; k++) out[o + k] = x[o + k] * inv * w[k];
  }
  return out;
}

function ropeInvFreq(cfg) {
  const hd = cfg.head_dim, theta = cfg.rope_theta || 10000;
  let inv = [];
  for (let i = 0; i < hd; i += 2) inv.push(1 / Math.pow(theta, i / hd));
  const rs = cfg.rope_scaling || {}, kind = rs.rope_type || rs.type;
  if (kind === "linear") inv = inv.map((f) => f / rs.factor);
  else if (kind === "llama3") {
    const f = rs.factor, lo = rs.low_freq_factor ?? 1, hi = rs.high_freq_factor ?? 4;
    const old = rs.original_max_position_embeddings ?? 8192;
    inv = inv.map((v) => {
      const wl = (2 * Math.PI) / v;
      const scaled = wl > old / lo ? v / f : v;
      if (wl <= old / lo && wl >= old / hi) {
        const s = (old / wl - lo) / (hi - lo);
        return ((1 - s) * scaled) / f + s * scaled;
      }
      return scaled;
    });
  }
  return Float64Array.from(inv);
}

// rotate one head vector (length hd) in place, HF "rotate_half" convention
function rope(vec, off, hd, p, inv) {
  const half = hd / 2;
  for (let j = 0; j < half; j++) {
    const a = p * inv[j], c = Math.fround(Math.cos(a)), s = Math.fround(Math.sin(a));
    const x1 = vec[off + j], x2 = vec[off + half + j];
    vec[off + j] = x1 * c - x2 * s;
    vec[off + half + j] = x2 * c + x1 * s;
  }
}

class Shard {
  constructor(cfg, tensors, start, end) {
    this.cfg = cfg; this.w = tensors; this.start = start; this.end = end;
    this.first = start === 0; this.last = end === cfg.n_layer;
    this.llama = cfg.arch === "llama";
    if (this.llama) this.inv = ropeInvFreq(cfg);
    this.caches = new Map(); // job -> {len, k: {layer: Float32Array}, v: {...}}
    this.pool = null; // optional multi-core backend (see worker.js); null = compute here
  }

  // y[T,N] = x[T,K] @ W[name] (+ b), on the pool when one is attached
  async mm(name, x, T, b) {
    const W = this.w[name];
    if (this.pool && this.pool.has(name)) {
      const y = await this.pool.mm(name, x, T);
      if (b) { const N = b.length; for (let i = 0; i < T; i++) for (let j = 0; j < N; j++) y[i * N + j] += b[j]; }
      return y;
    }
    return matmul(x, T, W.shape[0], W.data, W.shape[1], b);
  }

  // logits for one row with a tied head: dot products with every embedding row
  async tiedHead(name, n) {
    if (this.pool && this.pool.has(name + "^T")) return this.pool.mm(name + "^T", n, 1);
    const emb = this.w[name].data, V = this.cfg.vocab, D = this.cfg.n_embd, logits = new Float32Array(V);
    for (let t = 0; t < V; t++) {
      let s = 0;
      for (let k = 0; k < D; k++) s += n[k] * emb[t * D + k];
      logits[t] = s;
    }
    return logits;
  }

  drop(job) { this.caches.delete(job); }

  async forward(job, x, pos) {
    const D = this.cfg.n_embd, w = (n) => this.w[n].data;
    let cache = this.caches.get(job);
    if (!cache || cache.len !== pos) {
      if (pos !== 0) throw new Error(`no KV cache for job ${job} at position ${pos}`);
      cache = { len: 0, k: {}, v: {} };
    }
    this.caches.delete(job);
    this.caches.set(job, cache);
    while (this.caches.size > 8) this.caches.delete(this.caches.keys().next().value);

    let T, h;
    if (this.first) {
      T = x.data.length;
      h = new Float32Array(T * D);
      const emb = w(this.llama ? "embed" : "wte"), wpe = this.llama ? null : w("wpe");
      for (let i = 0; i < T; i++) {
        const id = x.data[i];
        for (let k = 0; k < D; k++) h[i * D + k] = emb[id * D + k] + (wpe ? wpe[(pos + i) * D + k] : 0);
      }
    } else {
      T = x.shape[0];
      h = Float32Array.from(x.data);
    }
    for (let l = this.start; l < this.end; l++) h = await (this.llama ? this.llamaBlock(l, h, T, pos, cache) : this.gpt2Block(l, h, T, pos, cache));
    cache.len = pos + T;
    if (!this.last) return { data: h, shape: [T, D] };

    const V = this.cfg.vocab, row = h.subarray((T - 1) * D, T * D), n = new Float32Array(D);
    if (this.llama) rmsNorm(row, 1, D, w("norm_f"), this.cfg.rms_eps, n);
    else layerNorm(row, 1, D, w("ln_f.w"), w("ln_f.b"), n);
    if (this.llama && this.w["lm_head"]) return { data: await this.mm("lm_head", n, 1, null), shape: [V] };
    return { data: await this.tiedHead(this.llama ? "embed" : "wte", n), shape: [V] };
  }

  // q: [T, H*hd] (row stride qs, offset qo), keys/values cached as [S, KV*hd]; writes y [T, H*hd]
  attend(q, qs, qo, K, Vc, T, pos, H, KV, hd) {
    const y = new Float32Array(T * H * hd), kvw = KV * hd, group = H / KV, scale = 1 / Math.sqrt(hd);
    const scores = new Float64Array(pos + T);
    for (let t = 0; t < T; t++) {
      const S = pos + t + 1;
      for (let hh = 0; hh < H; hh++) {
        const qb = t * qs + qo + hh * hd, kh = Math.floor(hh / group) * hd;
        let max = -Infinity;
        for (let s = 0; s < S; s++) {
          let d = 0;
          const kb = s * kvw + kh;
          for (let k = 0; k < hd; k++) d += q[qb + k] * K[kb + k];
          d = Math.fround(d) * scale;
          scores[s] = d;
          if (d > max) max = d;
        }
        let sum = 0;
        for (let s = 0; s < S; s++) { scores[s] = Math.exp(scores[s] - max); sum += scores[s]; }
        const yb = t * H * hd + hh * hd;
        for (let s = 0; s < S; s++) {
          const p = scores[s] / sum, vb = s * kvw + kh;
          for (let k = 0; k < hd; k++) y[yb + k] += p * Vc[vb + k];
        }
      }
    }
    return y;
  }

  // KV cache rows [S, width] for layer i, grown (doubling) to hold `need` positions
  kv(cache, i, width, need) {
    let k = cache.k[i], v = cache.v[i];
    if (!k || k.length < need * width) {
      const cap = Math.max(need, 64, k ? (2 * k.length) / width : 0) * width;
      const nk = new Float32Array(cap), nv = new Float32Array(cap);
      if (k) { nk.set(k); nv.set(v); }
      cache.k[i] = k = nk; cache.v[i] = v = nv;
    }
    return [k, v];
  }

  async gpt2Block(i, x, T, pos, cache) {
    const w = (n) => this.w[`h.${i}.${n}`].data;
    const { n_embd: D, n_head: H } = this.cfg, hd = D / H;
    const h = layerNorm(x, T, D, w("ln_1.w"), w("ln_1.b"), new Float32Array(T * D));
    const P = `h.${i}.`, qkv = await this.mm(P + "attn.c_attn.w", h, T, w("attn.c_attn.b"));
    const [K, Vc] = this.kv(cache, i, D, pos + T);
    for (let t = 0; t < T; t++) {
      K.set(qkv.subarray(t * 3 * D + D, t * 3 * D + 2 * D), (pos + t) * D);
      Vc.set(qkv.subarray(t * 3 * D + 2 * D, t * 3 * D + 3 * D), (pos + t) * D);
    }
    const y = this.attend(qkv, 3 * D, 0, K, Vc, T, pos, H, H, hd);
    const a = await this.mm(P + "attn.c_proj.w", y, T, w("attn.c_proj.b"));
    for (let k = 0; k < a.length; k++) a[k] += x[k];
    const h2 = layerNorm(a, T, D, w("ln_2.w"), w("ln_2.b"), new Float32Array(T * D));
    const m = await this.mm(P + "mlp.c_proj.w", gelu(await this.mm(P + "mlp.c_fc.w", h2, T, w("mlp.c_fc.b"))), T, w("mlp.c_proj.b"));
    for (let k = 0; k < m.length; k++) m[k] += a[k];
    return m;
  }

  async llamaBlock(i, x, T, pos, cache) {
    const opt = (n) => (this.w[`h.${i}.${n}`] ? this.w[`h.${i}.${n}`].data : null), w = (n) => this.w[`h.${i}.${n}`].data;
    const { n_embd: D, n_head: H, n_kv_head: KV, head_dim: hd, ffn: F, rms_eps: eps } = this.cfg;
    const h = rmsNorm(x, T, D, w("attn_norm"), eps, new Float32Array(T * D));
    const P = `h.${i}.`;
    const [q, k, v] = await Promise.all([this.mm(P + "q.w", h, T, opt("q.b")), this.mm(P + "k.w", h, T, opt("k.b")),
                                         this.mm(P + "v.w", h, T, opt("v.b"))]);
    const qn = opt("q_norm"), kn = opt("k_norm");
    for (let t = 0; t < T; t++) {
      for (let hh = 0; hh < H; hh++) {
        const o = t * H * hd + hh * hd;
        if (qn) rmsNorm(q.subarray(o, o + hd), 1, hd, qn, eps, q.subarray(o, o + hd));
        rope(q, o, hd, pos + t, this.inv);
      }
      for (let hh = 0; hh < KV; hh++) {
        const o = t * KV * hd + hh * hd;
        if (kn) rmsNorm(k.subarray(o, o + hd), 1, hd, kn, eps, k.subarray(o, o + hd));
        rope(k, o, hd, pos + t, this.inv);
      }
    }
    const [K, Vc] = this.kv(cache, i, KV * hd, pos + T);
    K.set(k, pos * KV * hd);
    Vc.set(v, pos * KV * hd);
    const y = this.attend(q, H * hd, 0, K, Vc, T, pos, H, KV, hd);
    const a = await this.mm(P + "o.w", y, T, null);
    for (let j = 0; j < a.length; j++) a[j] += x[j];
    const h2 = rmsNorm(a, T, D, w("mlp_norm"), eps, new Float32Array(T * D));
    const [g, u] = await Promise.all([this.mm(P + "gate.w", h2, T, null), this.mm(P + "up.w", h2, T, null)]);
    for (let j = 0; j < g.length; j++) g[j] = (g[j] / (1 + Math.exp(-g[j]))) * u[j];
    const m = await this.mm(P + "down.w", g, T, null);
    for (let j = 0; j < m.length; j++) m[j] += a[j];
    return m;
  }
}

// ---- wire format (see edgemesh/protocol.py) ----

const enc = new TextEncoder(), dec = new TextDecoder();

function packMsg(header, payload) {
  const h = enc.encode(JSON.stringify(header));
  const p = payload || new Uint8Array(0);
  const out = new Uint8Array(4 + h.length + p.length);
  new DataView(out.buffer).setUint32(0, h.length, false);
  out.set(h, 4); out.set(p, 4 + h.length);
  return out;
}

function unpackMsg(u8) {
  const n = new DataView(u8.buffer, u8.byteOffset, u8.byteLength).getUint32(0, false);
  return [JSON.parse(dec.decode(u8.subarray(4, 4 + n))), u8.subarray(4 + n)];
}

const DTYPES = { float32: Float32Array, int32: Int32Array, float16: Uint16Array };

// float16 bits -> Float32Array (weights arrive as float16 to halve the download)
const F16 = (() => {
  const t = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    t[h] = e === 0 ? s * m * 2 ** -24 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e - 15);
  }
  return t;
})();

function decodeTensors(u8) {
  const n = new DataView(u8.buffer, u8.byteOffset, u8.byteLength).getUint32(0, false);
  const meta = JSON.parse(dec.decode(u8.subarray(4, 4 + n)));
  let off = 4 + n;
  const out = {};
  for (const t of meta) {
    const Ctor = DTYPES[t.dtype];
    if (!Ctor) throw new Error(`browser workers do not support ${t.dtype}`);
    const count = t.shape.reduce((a, b) => a * b, 1), nbytes = count * Ctor.BYTES_PER_ELEMENT;
    const copy = u8.slice(off, off + nbytes); // copy => aligned buffer
    let data = new Ctor(copy.buffer);
    if (t.dtype === "float16") { const f = new Float32Array(data.length); for (let i = 0; i < data.length; i++) f[i] = F16[data[i]]; data = f; }
    out[t.name] = { data, shape: t.shape };
    off += nbytes + ((8 - (nbytes % 8)) % 8);
  }
  return out;
}

function encodeTensors(tensors) {
  const meta = [], parts = [];
  for (const [name, t] of Object.entries(tensors)) {
    const dtype = t.data instanceof Int32Array ? "int32" : "float32";
    meta.push({ name, dtype, shape: t.shape });
    parts.push(new Uint8Array(t.data.buffer, t.data.byteOffset, t.data.byteLength));
  }
  let m = JSON.stringify(meta);
  m += " ".repeat((8 - ((m.length + 4) % 8)) % 8);
  const mb = enc.encode(m);
  const pad = (n) => (8 - (n % 8)) % 8;
  const total = 4 + mb.length + parts.reduce((a, p) => a + p.length + pad(p.length), 0);
  const out = new Uint8Array(total);
  new DataView(out.buffer).setUint32(0, mb.length, false);
  out.set(mb, 4);
  let off = 4 + mb.length;
  for (const p of parts) { out.set(p, off); off += p.length + pad(p.length); }
  return out;
}

// Effective GFLOP/s for token-by-token generation on one core: vector x 1024x1024 matrix
// (4 MB, bigger than most caches), like edgemesh/worker.py's benchmark.
function benchmark() {
  const N = 1024, w = new Float32Array(N * N), x = new Float32Array(N);
  for (let i = 0; i < w.length; i++) w[i] = ((i * 2654435761) % 1000) / 1000 - 0.5;
  x.fill(0.5);
  matmul(x, 1, N, w, N, null);
  let n = 0;
  const t0 = performance.now();
  while (performance.now() - t0 < 300) { matmul(x, 1, N, w, N, null); n++; }
  return (n * 2 * N * N) / ((performance.now() - t0) / 1000) / 1e9;
}

if (typeof module !== "undefined") module.exports = { Shard, packMsg, unpackMsg, decodeTensors, encodeTensors, benchmark };
