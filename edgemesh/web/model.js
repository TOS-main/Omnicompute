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
    acc.set(b);
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

class Shard {
  constructor(cfg, tensors, start, end) {
    this.cfg = cfg; this.w = tensors; this.start = start; this.end = end;
    this.first = start === 0; this.last = end === cfg.n_layer;
    this.caches = new Map(); // job -> {len, k: {layer: Float32Array(n_ctx*D)}, v: {...}}
  }

  drop(job) { this.caches.delete(job); }

  forward(job, x, pos) {
    const { n_embd: D, n_ctx: C } = this.cfg;
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
      const wte = this.w["wte"].data, wpe = this.w["wpe"].data;
      h = new Float32Array(T * D);
      for (let i = 0; i < T; i++) {
        const id = x.data[i];
        for (let k = 0; k < D; k++) h[i * D + k] = wte[id * D + k] + wpe[(pos + i) * D + k];
      }
    } else {
      T = x.shape[0];
      h = Float32Array.from(x.data);
    }
    for (let l = this.start; l < this.end; l++) h = this.block(l, h, T, pos, cache);
    cache.len = pos + T;

    if (this.last) {
      const V = this.cfg.vocab;
      const lastRow = layerNorm(h.subarray((T - 1) * D, T * D), 1, D, this.w["ln_f.w"].data, this.w["ln_f.b"].data, new Float32Array(D));
      const wte = this.w["wte"].data, logits = new Float32Array(V);
      for (let t = 0; t < V; t++) {
        let s = 0;
        for (let k = 0; k < D; k++) s += lastRow[k] * wte[t * D + k];
        logits[t] = s;
      }
      return { data: logits, shape: [V] };
    }
    return { data: h, shape: [T, D] };
  }

  block(i, x, T, pos, cache) {
    const w = (n) => this.w[`h.${i}.${n}`].data;
    const { n_embd: D, n_head: H, n_ctx: C } = this.cfg;
    const hd = D / H, scale = 1 / Math.sqrt(hd);
    const h = layerNorm(x, T, D, w("ln_1.w"), w("ln_1.b"), new Float32Array(T * D));
    const qkv = matmul(h, T, D, w("attn.c_attn.w"), 3 * D, w("attn.c_attn.b"));
    if (!cache.k[i]) { cache.k[i] = new Float32Array(C * D); cache.v[i] = new Float32Array(C * D); }
    const K = cache.k[i], Vc = cache.v[i];
    for (let t = 0; t < T; t++) {
      K.set(qkv.subarray(t * 3 * D + D, t * 3 * D + 2 * D), (pos + t) * D);
      Vc.set(qkv.subarray(t * 3 * D + 2 * D, t * 3 * D + 3 * D), (pos + t) * D);
    }
    const y = new Float32Array(T * D);
    const scores = new Float64Array(pos + T);
    for (let t = 0; t < T; t++) {
      const S = pos + t + 1; // causal: attend to positions 0..pos+t
      for (let hh = 0; hh < H; hh++) {
        const qo = t * 3 * D + hh * hd;
        let max = -Infinity;
        for (let s = 0; s < S; s++) {
          let d = 0;
          const ko = s * D + hh * hd;
          for (let k = 0; k < hd; k++) d += qkv[qo + k] * K[ko + k];
          d = Math.fround(d) * scale;
          scores[s] = d;
          if (d > max) max = d;
        }
        let sum = 0;
        for (let s = 0; s < S; s++) { scores[s] = Math.exp(scores[s] - max); sum += scores[s]; }
        const yo = t * D + hh * hd;
        for (let s = 0; s < S; s++) {
          const p = scores[s] / sum, vo = s * D + hh * hd;
          for (let k = 0; k < hd; k++) y[yo + k] += p * Vc[vo + k];
        }
      }
    }
    const a = matmul(y, T, D, w("attn.c_proj.w"), D, w("attn.c_proj.b"));
    for (let k = 0; k < a.length; k++) a[k] += x[k];
    const h2 = layerNorm(a, T, D, w("ln_2.w"), w("ln_2.b"), new Float32Array(T * D));
    const m = matmul(gelu(matmul(h2, T, D, w("mlp.c_fc.w"), 4 * D, w("mlp.c_fc.b"))), T, 4 * D, w("mlp.c_proj.w"), D, w("mlp.c_proj.b"));
    for (let k = 0; k < m.length; k++) m[k] += a[k];
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

const DTYPES = { float32: Float32Array, int32: Int32Array };

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
    out[t.name] = { data: new Ctor(copy.buffer), shape: t.shape };
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

function benchmark() {
  const N = 96, a = new Float32Array(N * N).fill(0.5), b = new Float32Array(N);
  let n = 0;
  const t0 = performance.now();
  while (performance.now() - t0 < 300) { matmul(a, N, N, a, N, b); n++; }
  return Math.round((n * 2 * N * N * N) / ((performance.now() - t0) / 1000) / 1e7) / 100;
}

if (typeof module !== "undefined") module.exports = { Shard, packMsg, unpackMsg, decodeTensors, encodeTensors, benchmark };
