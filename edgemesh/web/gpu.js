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

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let n = g.x;
  let t = g.y;
  if (n >= p.N2) { return; }
  var acc = vec2<f32>(0.0, 0.0);
  let xo = t * p.K;
  for (var k = 0u; k < p.K; k = k + 1u) {
    acc = acc + X[xo + k] * unpack2x16float(W[k * p.N2 + n]);
  }
  Y[t * p.N2 + n] = acc;
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

  async clear() { for (const m of this.mats.values()) for (const c of m.chunks) c.buf.destroy(); this.mats.clear(); }

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

  async mm(name, x, T) {
    const dev = this.device, m = this.mats.get(name);
    const xbuf = dev.createBuffer({ size: Math.max(16, x.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    dev.queue.writeBuffer(xbuf, 0, x.buffer, x.byteOffset, x.byteLength);
    const enc = dev.createCommandEncoder(), reads = [], temps = [xbuf];
    for (const c of m.chunks) {
      const size = T * c.n2 * 8;
      const out = dev.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
      const read = dev.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      const uni = dev.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      dev.queue.writeBuffer(uni, 0, new Uint32Array([m.K, c.n2, T, 0]));
      const bind = dev.createBindGroup({ layout: this.pipeline.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: c.buf } }, { binding: 1, resource: { buffer: xbuf } },
        { binding: 2, resource: { buffer: out } }, { binding: 3, resource: { buffer: uni } }] });
      const pass = enc.beginComputePass();
      pass.setPipeline(this.pipeline);
      pass.setBindGroup(0, bind);
      pass.dispatchWorkgroups(Math.ceil(c.n2 / 64), T, 1);
      pass.end();
      enc.copyBufferToBuffer(out, 0, read, 0, size);
      reads.push([c, read]);
      temps.push(out, uni);
    }
    dev.queue.submit([enc.finish()]);
    const y = new Float32Array(T * m.N);
    await Promise.all(reads.map(async ([c, read]) => {
      await read.mapAsync(GPUMapMode.READ);
      const part = new Float32Array(read.getMappedRange()), w = c.n2 * 2, c0 = c.p0 * 2, keep = Math.min(w, m.N - c0);
      for (let t = 0; t < T; t++) y.set(part.subarray(t * w, t * w + keep), t * m.N + c0);
      read.unmap();
      read.destroy();
    }));
    for (const b of temps) b.destroy();
    return y;
  }

  // effective GFLOP/s for vector x 1024x1024 matrix, including the round trip
  async bench() {
    const N = 1024, w = new Float32Array(N * N), x = new Float32Array(N).fill(0.5);
    for (let i = 0; i < w.length; i++) w[i] = ((i * 2654435761) % 1000) / 1000 - 0.5;
    await this.putMatrix("_bench", { data: w, shape: [N, N] });
    await this.mm("_bench", x, 1);
    let n = 0;
    const t0 = performance.now();
    while (performance.now() - t0 < 300) { await this.mm("_bench", x, 1); n++; }
    for (const c of this.mats.get("_bench").chunks) c.buf.destroy();
    this.mats.delete("_bench");
    return (n * 2 * N * N) / ((performance.now() - t0) / 1000) / 1e9;
  }
}

if (typeof module !== "undefined") module.exports = { toHalf };
