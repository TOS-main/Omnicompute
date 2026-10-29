// Layer shard in plain JavaScript (GPT-2, Llama-style and Gemma 4): the same math as
// omnicompute/model.py, so a phone or Chromebook browser can host a slice of the model. Tensors are {data: TypedArray, shape: [...]}.

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
    this.gemma = cfg.arch === "gemma4";
    if (this.llama) this.inv = ropeInvFreq(cfg);
    if (this.gemma) this.gemmaSetup();
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

  jobCache(job, pos) {
    let cache = this.caches.get(job);
    if (!cache || cache.len !== pos) {
      if (pos !== 0) throw new Error(`no KV cache for job ${job} at position ${pos}`);
      cache = { len: 0, k: {}, v: {} };
    }
    this.caches.delete(job);
    this.caches.set(job, cache);
    while (this.caches.size > 8) this.caches.delete(this.caches.keys().next().value);
    return cache;
  }

  // a step's tensors {x, extras} -> what goes to the next device ({x: hidden or logits, extras});
  // only Gemma 4 has extras (see gemmaForward)
  async forwardEx(job, t, pos) {
    if (this.gemma) return this.gemmaForward(job, t, pos);
    return { x: await this.forward(job, t.x, pos) };
  }

  async forward(job, x, pos) {
    const D = this.cfg.n_embd, w = (n) => this.w[n].data;
    const cache = this.jobCache(job, pos);

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
  // scale defaults to 1/sqrt(hd); window > 0: each token sees only that many most recent positions
  attend(q, qs, qo, K, Vc, T, pos, H, KV, hd, scale = 1 / Math.sqrt(hd), window = 0) {
    const y = new Float32Array(T * H * hd), kvw = KV * hd, group = H / KV;
    const scores = new Float64Array(pos + T);
    for (let t = 0; t < T; t++) {
      const S = pos + t + 1, s0 = window > 0 ? Math.max(0, S - window) : 0;
      for (let hh = 0; hh < H; hh++) {
        const qb = t * qs + qo + hh * hd, kh = Math.floor(hh / group) * hd;
        let max = -Infinity;
        for (let s = s0; s < S; s++) {
          let d = 0;
          const kb = s * kvw + kh;
          for (let k = 0; k < hd; k++) d += q[qb + k] * K[kb + k];
          d = Math.fround(d) * scale;
          scores[s] = d;
          if (d > max) max = d;
        }
        let sum = 0;
        for (let s = s0; s < S; s++) { scores[s] = Math.exp(scores[s] - max); sum += scores[s]; }
        const yb = t * H * hd + hh * hd;
        for (let s = s0; s < S; s++) {
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

  // ---- Gemma 4 (see omnicompute/model.py): sliding-window and full layers with their own head
  // sizes, q/k/v norms, unscaled attention, norms after attention and the MLP, per-layer
  // embeddings (PLE) and a learned output scale. The last n_kv_shared layers reuse the keys and
  // values of the last earlier layer of the same kind; between devices travel the PLE inputs of
  // the layers still to come ("ple") and the source keys/values a later device needs ("kvk.j").

  gemmaSetup() {
    const c = this.cfg, hd = c.head_dim, hds = c.head_dim_swa, f = c.rope_factors || [];
    this.invFull = Float64Array.from({ length: hd / 2 }, (_, j) => 1 / Math.pow(c.rope_theta, (2 * j) / hd) / (f.length === hd / 2 ? f[j] : 1));
    this.invSwa = Float64Array.from({ length: hds / 2 }, (_, j) => 1 / Math.pow(c.rope_theta_swa, (2 * j) / hds));
    const L = c.n_layer, firstShared = L - (c.n_kv_shared || 0);
    this.kvSrc = {};
    for (let i = firstShared; i < L; i++) {
      let j = -1;
      for (let k = 0; k < firstShared; k++) if (c.swa[k] === c.swa[i]) j = k;
      this.kvSrc[i] = j;
    }
    this.sendKv = new Set(Object.keys(this.kvSrc).map(Number).filter((i) => i >= this.end).map((i) => this.kvSrc[i]));
  }

  async gemmaForward(job, t, pos) {
    const c = this.cfg, D = c.n_embd, cache = this.jobCache(job, pos);
    let T, h, ple = null, pleL = 0;
    if (this.first) {
      if (c.ple_dim) throw new Error("the first layers of this model run on a computer or the app");
      T = t.x.data.length;
      h = new Float32Array(T * D);
      const emb = this.w.embed.data, sc = Math.fround(Math.sqrt(D));
      for (let i = 0; i < T; i++) for (let k = 0; k < D; k++) h[i * D + k] = emb[t.x.data[i] * D + k] * sc;
    } else {
      T = t.x.shape[0];
      h = Float32Array.from(t.x.data);
      if (t.ple) { ple = t.ple.data; pleL = t.ple.shape[1]; }
      for (const name of Object.keys(t)) { // an earlier device's source keys/values, [KV, T, hd]
        if (!name.startsWith("kvk.")) continue;
        const j = Number(name.slice(4)), k = t[name], [KV, n, hd] = k.shape;
        const [K, Vc] = this.kv(cache, j, KV * hd, pos + n);
        this.fromHeads(k.data, K, pos, KV, n, hd);
        this.fromHeads(t[`kvv.${j}`].data, Vc, pos, KV, n, hd);
      }
    }
    this.stepKv = {};
    const P = c.ple_dim || 0;
    for (let l = this.start; l < this.end; l++) {
      let pi = null;
      if (ple) {
        pi = new Float32Array(T * P);
        for (let i = 0; i < T; i++) pi.set(ple.subarray((i * pleL + l - this.start) * P, (i * pleL + l - this.start + 1) * P), i * P);
      }
      h = await this.gemmaBlock(l, h, T, pos, cache, pi);
    }
    cache.len = pos + T;
    if (this.last) {
      const n = rmsNorm(h.subarray((T - 1) * D, T * D), 1, D, this.w.norm_f.data, c.rms_eps, new Float32Array(D));
      const logits = this.w.lm_head ? await this.mm("lm_head", n, 1, null) : await this.tiedHead("embed", n);
      if (c.softcap) for (let i = 0; i < logits.length; i++) logits[i] = Math.tanh(logits[i] / c.softcap) * c.softcap;
      return { x: { data: logits, shape: [logits.length] } };
    }
    const out = { x: { data: h, shape: [T, D] } };
    if (ple) {
      const left = pleL - (this.end - this.start), o = new Float32Array(T * left * P);
      for (let i = 0; i < T; i++) o.set(ple.subarray((i * pleL + this.end - this.start) * P, (i + 1) * pleL * P), i * left * P);
      out.ple = { data: o, shape: [T, left, P] };
    }
    for (const j of this.sendKv) {
      if (this.stepKv[j]) { out[`kvk.${j}`] = this.stepKv[j][0]; out[`kvv.${j}`] = this.stepKv[j][1]; }
      else if (t[`kvk.${j}`]) { out[`kvk.${j}`] = t[`kvk.${j}`]; out[`kvv.${j}`] = t[`kvv.${j}`]; }
    }
    return out;
  }

  // rows [T, KV*hd] at position pos of the cache <- [KV, T, hd], and back
  fromHeads(src, dst, pos, KV, T, hd) {
    for (let g = 0; g < KV; g++) for (let i = 0; i < T; i++) dst.set(src.subarray((g * T + i) * hd, (g * T + i + 1) * hd), (pos + i) * KV * hd + g * hd);
  }

  toHeads(rows, KV, T, hd) {
    const out = new Float32Array(KV * T * hd);
    for (let g = 0; g < KV; g++) for (let i = 0; i < T; i++) out.set(rows.subarray(i * KV * hd + g * hd, i * KV * hd + (g + 1) * hd), (g * T + i) * hd);
    return { data: out, shape: [KV, T, hd] };
  }

  async gemmaBlock(i, x, T, pos, cache, pleI) {
    const c = this.cfg, P = `h.${i}.`, w = (n) => this.w[P + n].data;
    const swa = c.swa[i], eps = c.rms_eps, H = c.n_head, D = c.n_embd, KV = c.n_kv_heads[i];
    const hd = swa ? c.head_dim_swa : c.head_dim, inv = swa ? this.invSwa : this.invFull;
    const h = rmsNorm(x, T, D, w("attn_norm"), eps, new Float32Array(T * D));
    const src = this.kvSrc[i], shared = src !== undefined;
    const names = [P + "q.w"].concat(shared ? [] : [P + "k.w"].concat(c.k_eq_v[i] ? [] : [P + "v.w"]));
    const outs = await Promise.all(names.map((n) => this.mm(n, h, T, null)));
    const q = outs[0], qn = w("q_norm");
    for (let t = 0; t < T; t++) for (let hh = 0; hh < H; hh++) {
      const o = t * H * hd + hh * hd, s = q.subarray(o, o + hd);
      rmsNorm(s, 1, hd, qn, eps, s);
      rope(q, o, hd, pos + t, inv);
    }
    let K, Vc, kvh = KV;
    if (shared) {
      K = cache.k[src]; Vc = cache.v[src]; kvh = c.n_kv_heads[src];
      if (!K) throw new Error(`no keys/values from layer ${src}`);
    } else {
      const k = outs[1], v = c.k_eq_v[i] ? Float32Array.from(k) : outs[2], kn = w("k_norm"), one = new Float32Array(hd).fill(1);
      for (let t = 0; t < T; t++) for (let hh = 0; hh < KV; hh++) {
        const o = t * KV * hd + hh * hd, ks = k.subarray(o, o + hd), vs = v.subarray(o, o + hd);
        rmsNorm(ks, 1, hd, kn, eps, ks);
        rope(k, o, hd, pos + t, inv);
        rmsNorm(vs, 1, hd, one, eps, vs);
      }
      if (this.sendKv.has(i)) this.stepKv[i] = [this.toHeads(k, KV, T, hd), this.toHeads(v, KV, T, hd)];
      [K, Vc] = this.kv(cache, i, KV * hd, pos + T);
      K.set(k, pos * KV * hd);
      Vc.set(v, pos * KV * hd);
    }
    const y = this.attend(q, H * hd, 0, K, Vc, T, pos, H, kvh, hd, 1, swa ? c.window : 0);
    const a = rmsNorm(await this.mm(P + "o.w", y, T, null), T, D, w("post_attn_norm"), eps, new Float32Array(T * D));
    for (let j = 0; j < a.length; j++) a[j] += x[j];
    const h2 = rmsNorm(a, T, D, w("mlp_norm"), eps, new Float32Array(T * D));
    const [g, u] = await Promise.all([this.mm(P + "gate.w", h2, T, null), this.mm(P + "up.w", h2, T, null)]);
    gelu(g);
    for (let j = 0; j < g.length; j++) g[j] *= u[j];
    const m = rmsNorm(await this.mm(P + "down.w", g, T, null), T, D, w("post_mlp_norm"), eps, new Float32Array(T * D));
    for (let j = 0; j < m.length; j++) m[j] += a[j];
    if (pleI) {
      const pg = gelu(await this.mm(P + "ple_gate.w", m, T, null));
      for (let j = 0; j < pg.length; j++) pg[j] *= pleI[j];
      const po = rmsNorm(await this.mm(P + "ple_out.w", pg, T, null), T, D, w("ple_post_norm"), eps, new Float32Array(T * D));
      for (let j = 0; j < m.length; j++) m[j] += po[j];
    }
    const sc = w("out_scale")[0];
    for (let j = 0; j < m.length; j++) m[j] *= sc;
    return m;
  }
}

// ---- wire format (see omnicompute/protocol.py) ----

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
// (4 MB, bigger than most caches), like omnicompute/worker.py's benchmark.
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
