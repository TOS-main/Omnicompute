// WebGPU backend for the browser worker: the same shader as edgemesh/gpu.py.
// Weight matrices live on the GPU as packed float16 pairs; activations stay float32.
// WebGPU needs a secure page (https, or localhost): use the --public link on phones and Chromebooks.

"use strict";

const GPU_SHADER = `
struct P { K: u32, N2: u32, T: u32, pad: u32 };
@group(0) @binding(0) var<storage, read> W: array<u32>;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y: array<vec2<f32>>;
@group(0) @binding(3) var<uniform> p: P;
var<workgroup> part: array<vec2<f32>, 256>;

// 32 column pairs per workgroup, each summed by 8 threads over every 8th row, so a GPU's
// thousands of threads all stay busy even on small matrices.
@compute @workgroup_size(32, 8)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
  let n = wg.x * 32u + l.x;
  let t = wg.y;
  var acc = vec2<f32>(0.0, 0.0);
  if (n < p.N2) {
    let xo = t * p.K;
    for (var k = l.y; k < p.K; k = k + 8u) {
      acc = acc + X[xo + k] * unpack2x16float(W[k * p.N2 + n]);
    }
  }
  part[l.y * 32u + l.x] = acc;
  workgroupBarrier();
  if (l.y == 0u && n < p.N2) {
    var s = part[l.x];
    for (var i = 1u; i < 8u; i = i + 1u) { s = s + part[i * 32u + l.x]; }
    Y[t * p.N2 + n] = s;
  }
}`;

// float32 -> float16 bits (round to nearest even)
const _f32 = new Float32Array(1), _u32 = new Uint32Array(_f32.buffer);
function toHalf(v) {
  _f32[0] = v;
  const x = _u32[0], sign = (x >>> 16) & 0x8000, exp = (x >>> 23) & 0xff;
  let m = x & 0x7fffff;
  if (exp === 255) return sign | 0x7c00 | (m ? 0x200 : 0);
  const e = exp - 127 + 15;
  if (e >= 31) return sign | 0x7c00;
  if (e <= 0) {
    if (e < -10) return sign;
    m |= 0x800000;
    const shift = 14 - e, half = m >>> shift, rem = m & ((1 << shift) - 1), mid = 1 << (shift - 1);
    return sign | (half + (rem > mid || (rem === mid && half & 1) ? 1 : 0));
  }
  const half = (e << 10) | (m >>> 13), rem = m & 0x1fff;
  return sign | (half + (rem > 0x1000 || (rem === 0x1000 && half & 1) ? 1 : 0));
}

class GpuPool {
  static async open() {
    if (typeof navigator === "undefined" || !navigator.gpu) return null;
    try {
      const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
      if (!adapter) return null;
      const lim = adapter.limits;
      const device = await adapter.requestDevice({ requiredLimits: {
        maxStorageBufferBindingSize: lim.maxStorageBufferBindingSize, maxBufferSize: lim.maxBufferSize } });
      const info = adapter.info || {};
      return new GpuPool(device, [info.vendor, info.architecture || info.description].filter(Boolean).join(" ") || "GPU",
        Math.min(lim.maxStorageBufferBindingSize, 120 << 20));
    } catch { return null; }
  }

  constructor(device, name, maxBind) {
    this.device = device; this.name = name; this.maxBind = maxBind;
    this.pipeline = device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code: GPU_SHADER }), entryPoint: "main" } });
    this.mats = new Map();
    this.lost = false;
    device.lost.then(() => { this.lost = true; });
  }

  has(name) { return this.mats.has(name); }

  async clear() { for (const name of [...this.mats.keys()]) this.drop(name); }

  // W[K,N] row-major Float32Array -> column chunks of packed half pairs
  async putMatrix(name, t) {
    const [K, N] = t.shape, N2full = Math.ceil(N / 2);
    const per = Math.max(1, Math.floor(this.maxBind / (K * 4))); // pairs per chunk
    const chunks = [];
    for (let p0 = 0; p0 < N2full; p0 += per) {
      const n2 = Math.min(per, N2full - p0), packed = new Uint32Array(K * n2);
      for (let k = 0; k < K; k++) {
        const row = k * N;
        for (let j = 0; j < n2; j++) {
          const c = 2 * (p0 + j);
          const lo = toHalf(t.data[row + c]), hi = c + 1 < N ? toHalf(t.data[row + c + 1]) : 0;
          packed[k * n2 + j] = lo | (hi << 16);
        }
      }
      const buf = this.device.createBuffer({ size: packed.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      this.device.queue.writeBuffer(buf, 0, packed);
      chunks.push({ buf, p0, n2 });
    }
    this.mats.set(name, { K, N, chunks });
  }

  // tied output head E[V,D]: store E^T so logits = x @ E^T
  async putRows(name, t) {
    const [V, D] = t.shape, tr = new Float32Array(D * V);
    for (let v = 0; v < V; v++) for (let d = 0; d < D; d++) tr[d * V + v] = t.data[v * D + d];
    await this.putMatrix(name, { data: tr, shape: [D, V] });
  }

  // Input, output and readback buffers for one matrix; kept for one-token steps (same shapes every token).
  buffers(name, T) {
    const m = this.mats.get(name);
    if (T === 1 && m.step) return m.step;
    const dev = this.device;
    const xbuf = dev.createBuffer({ size: Math.max(16, T * m.K * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const all = [xbuf], chunks = [];
    for (const c of m.chunks) {
      const size = T * c.n2 * 8;
      const out = dev.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
      const read = dev.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      const uni = dev.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      dev.queue.writeBuffer(uni, 0, new Uint32Array([m.K, c.n2, T, 0]));
      const bind = dev.createBindGroup({ layout: this.pipeline.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: c.buf } }, { binding: 1, resource: { buffer: xbuf } },
        { binding: 2, resource: { buffer: out } }, { binding: 3, resource: { buffer: uni } }] });
      all.push(out, read, uni);
      chunks.push({ c, out, read, bind, size });
    }
    const set = { xbuf, chunks, all };
    if (T === 1) m.step = set;
    return set;
  }

  async mm(name, x, T) {
    const dev = this.device, m = this.mats.get(name), set = this.buffers(name, T);
    dev.queue.writeBuffer(set.xbuf, 0, x.buffer, x.byteOffset, x.byteLength);
    const enc = dev.createCommandEncoder();
    for (const { c, out, read, bind, size } of set.chunks) {
      const pass = enc.beginComputePass();
      pass.setPipeline(this.pipeline);
      pass.setBindGroup(0, bind);
      pass.dispatchWorkgroups(Math.ceil(c.n2 / 32), T, 1);
      pass.end();
      enc.copyBufferToBuffer(out, 0, read, 0, size);
    }
    dev.queue.submit([enc.finish()]);
    const y = new Float32Array(T * m.N);
    await Promise.all(set.chunks.map(async ({ c, read }) => {
      await read.mapAsync(GPUMapMode.READ);
      const part = new Float32Array(read.getMappedRange()), w = c.n2 * 2, c0 = c.p0 * 2, keep = Math.min(w, m.N - c0);
      for (let t = 0; t < T; t++) y.set(part.subarray(t * w, t * w + keep), t * m.N + c0);
      read.unmap();
    }));
    if (T !== 1) for (const b of set.all) b.destroy();
    return y;
  }

  drop(name) {
    const m = this.mats.get(name);
    if (!m) return;
    for (const c of m.chunks) c.buf.destroy();
    if (m.step) for (const b of m.step.all) b.destroy();
    this.mats.delete(name);
  }

  // effective GFLOP/s for one generation step through a 2048x2048 matrix, round trip included
  async bench() {
    const N = 2048, w = new Float32Array(N * N), x = new Float32Array(N).fill(0.5);
    for (let i = 0; i < w.length; i++) w[i] = ((i * 2654435761) % 1000) / 1000 - 0.5;
    await this.putMatrix("_bench", { data: w, shape: [N, N] });
    await this.mm("_bench", x, 1);
    let n = 0;
    const t0 = performance.now();
    while (performance.now() - t0 < 300) { await this.mm("_bench", x, 1); n++; }
    const s = (performance.now() - t0) / 1000;
    this.drop("_bench");
    return (n * 2 * N * N) / s / 1e9;
  }
}

if (typeof module !== "undefined") module.exports = { toHalf };
