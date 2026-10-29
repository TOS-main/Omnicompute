"""Optional GPU backend for native workers, through WebGPU (the `wgpu` package).

WebGPU runs on Direct3D 12 (Windows), Metal (macOS) and Vulkan (Linux, Android), so one
code path covers NVIDIA, AMD, Intel and Apple GPUs with no CUDA install. The browser worker
(web/worker.js) uses the same shader through the browser's WebGPU.

Weight matrices live on the GPU as packed float16 pairs (half the memory and bandwidth of
float32); activations stay float32 on the CPU. Generation multiplies one vector by each
matrix, so the gain comes from the GPU's much faster memory. Several matrices that share an
input (q/k/v, gate/up) run in one submission to save round trips.

Install with `pip install wgpu`. Without it, or without a usable GPU, workers use the CPU.
EDGEMESH_GPU=0 turns the GPU off; EDGEMESH_GPU=force uses it even when the CPU is faster
(and accepts software-emulated GPUs, for testing).
"""

from __future__ import annotations

import logging
import os
import threading
import time

import numpy as np

log = logging.getLogger("edgemesh.gpu")

SHADER = """
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
}
"""

CHUNK_BYTES = 120 << 20  # stay under the default 128 MB storage binding limit


class GPUMismatch(RuntimeError):
    pass


class GPU:
    """Holds weight matrices on one GPU and multiplies activations by them."""

    def __init__(self, adapter):
        import wgpu

        self.wgpu = wgpu
        info = adapter.info
        self.name = info.get("device") or info.get("description") or "GPU"
        self.software = info.get("adapter_type") == "CPU"
        lim = adapter.limits
        self.device = adapter.request_device_sync(required_limits={
            "max-storage-buffer-binding-size": lim["max-storage-buffer-binding-size"],
            "max-buffer-size": lim["max-buffer-size"]})
        self.max_bind = min(lim["max-storage-buffer-binding-size"], CHUNK_BYTES)
        self.pipeline = self.device.create_compute_pipeline(
            layout="auto", compute={"module": self.device.create_shader_module(code=SHADER), "entry_point": "main"})
        self.mats: dict[str, dict] = {}
        self.lock = threading.Lock()  # steps run in worker threads

    # ---------- weights

    def put(self, name: str, w: np.ndarray) -> None:
        """w: [K, N] (float16 or float32). Stored as column chunks of packed float16 pairs."""
        K, N = w.shape
        h = np.ascontiguousarray(w, dtype=np.float16)
        if N % 2:
            h = np.concatenate([h, np.zeros((K, 1), np.float16)], 1)
        per_chunk = max(2, (self.max_bind // (K * 2)) // 2 * 2)  # columns per chunk, even
        chunks = []
        for c0 in range(0, h.shape[1], per_chunk):
            part = np.ascontiguousarray(h[:, c0:c0 + per_chunk])
            buf = self.device.create_buffer_with_data(data=part.view(np.uint32).tobytes(),
                                                      usage=self.wgpu.BufferUsage.STORAGE)
            chunks.append({"buf": buf, "c0": c0, "n2": part.shape[1] // 2})
        self.mats[name] = {"K": K, "N": N, "chunks": chunks}
        # drivers differ: check one product against numpy before trusting this GPU with the matrix
        x = np.sin(np.arange(K, dtype=np.float32) * 12.9898)[None] * 0.5
        y, ref = self.mm_many(x, [name])[0][0], (x @ h[:, :N].astype(np.float32))[0]
        if not np.all(np.isfinite(y)) or np.abs(y - ref).max() > 0.01 * max(np.abs(ref).max(), 1e-3) + 1e-3:
            raise GPUMismatch(f"{self.name} gave wrong results for {name}")

    def has(self, name: str) -> bool:
        return name in self.mats

    def clear(self) -> None:
        for m in self.mats.values():
            for c in m["chunks"]:
                c["buf"].destroy()
        self.mats.clear()

    # ---------- compute

    def mm_many(self, x: np.ndarray, names: list[str]) -> list[np.ndarray]:
        """[x @ W[name] for name in names], x: [T, K] float32, in one GPU submission."""
        with self.lock:
            return self._mm_many(x, names)

    def _mm_many(self, x: np.ndarray, names: list[str]) -> list[np.ndarray]:
        wgpu, dev = self.wgpu, self.device
        x = np.ascontiguousarray(x, dtype=np.float32)
        T = x.shape[0]
        xbuf = dev.create_buffer_with_data(data=x.tobytes(), usage=wgpu.BufferUsage.STORAGE)
        enc = dev.create_command_encoder()
        jobs = []
        for name in names:
            m = self.mats[name]
            for c in m["chunks"]:
                size = T * c["n2"] * 8
                out = dev.create_buffer(size=size, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
                read = dev.create_buffer(size=size, usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST)
                uni = dev.create_buffer_with_data(data=np.array([m["K"], c["n2"], T, 0], np.uint32).tobytes(),
                                                  usage=wgpu.BufferUsage.UNIFORM)
                bind = dev.create_bind_group(layout=self.pipeline.get_bind_group_layout(0), entries=[
                    {"binding": 0, "resource": {"buffer": c["buf"]}},
                    {"binding": 1, "resource": {"buffer": xbuf}},
                    {"binding": 2, "resource": {"buffer": out}},
                    {"binding": 3, "resource": {"buffer": uni}}])
                pas = enc.begin_compute_pass()
                pas.set_pipeline(self.pipeline)
                pas.set_bind_group(0, bind)
                pas.dispatch_workgroups((c["n2"] + 63) // 64, T, 1)
                pas.end()
                enc.copy_buffer_to_buffer(out, 0, read, 0, size)
                jobs.append((name, c, read, size))
        dev.queue.submit([enc.finish()])
        results = {n: np.empty((T, self.mats[n]["N"] + self.mats[n]["N"] % 2), np.float32) for n in names}
        for name, c, read, size in jobs:
            read.map_sync(wgpu.MapMode.READ)
            y = np.frombuffer(read.read_mapped(), np.float32).reshape(T, c["n2"] * 2)
            results[name][:, c["c0"]:c["c0"] + c["n2"] * 2] = y
            read.unmap()
        return [results[n][:, :self.mats[n]["N"]] for n in names]

    def benchmark(self) -> float:
        """Effective GFLOP/s for vector x 2048x2048 matrix, including the round trip."""
        self.put("_bench", np.random.default_rng(0).standard_normal((2048, 2048)).astype(np.float16))
        x = np.ones((1, 2048), np.float32)
        try:
            self.mm_many(x, ["_bench"])
            n, t0 = 0, time.perf_counter()
            while time.perf_counter() - t0 < 0.4:
                self.mm_many(x, ["_bench"])
                n += 1
            return round(n * 2 * 2048 * 2048 / (time.perf_counter() - t0) / 1e9, 2)
        finally:
            for c in self.mats.pop("_bench")["chunks"]:
                c["buf"].destroy()


def open_gpu(allow_software: bool | None = None) -> GPU | None:
    """The best GPU on this machine, or None (no wgpu package, no GPU, or EDGEMESH_GPU=0).
    Software (CPU-emulated) adapters are skipped unless allowed, since numpy is faster."""
    if os.environ.get("EDGEMESH_GPU", "1") == "0":
        return None
    if allow_software is None:
        allow_software = os.environ.get("EDGEMESH_GPU") == "force"
    try:
        import wgpu
    except ImportError:
        return None
    try:
        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        if adapter is None or (adapter.info.get("adapter_type") == "CPU" and not allow_software):
            return None
        return GPU(adapter)
    except Exception as e:  # driver problems: fall back to the CPU
        log.info("no usable GPU: %s", e)
        return None
