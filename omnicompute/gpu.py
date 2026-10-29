"""Optional GPU backends for native workers.

* CUDA, through PyTorch, on NVIDIA cards when a CUDA build of torch is installed (the
  installers offer it when they find an NVIDIA card). Weights are float16 and cuBLAS does the
  products, which runs close to the card's memory bandwidth.
* WebGPU (the `wgpu` package) everywhere else: Direct3D 12 on Windows, Metal on macOS and
  Vulkan on Linux, so one code path covers NVIDIA, AMD, Intel and Apple GPUs with no driver
  toolkit. The browser worker (web/worker.js) uses the same shaders through the browser.

Generation multiplies one vector by each weight matrix, so speed is set by how many bytes of
weights the GPU reads per token. On WebGPU the weights are stored as 8-bit integers with one
scale per 128 rows of each column (half the bytes of float16, with a small loss of accuracy);
OMNICOMPUTE_WEIGHTS=16 keeps float16. Activations stay float32 on the CPU. Several matrices that
share an input (q/k/v, gate/up) run in one submission to save round trips.

OMNICOMPUTE_GPU=0 turns the GPU off, =webgpu skips CUDA, =force uses the GPU even when the CPU is
faster (and accepts software-emulated GPUs, for testing).
"""

from __future__ import annotations

import logging
import os
import threading
import time

import numpy as np

log = logging.getLogger("omnicompute.gpu")

# Both shaders come in two forms: one token (each new token while answering) and four tokens
# per workgroup (reading a prompt), which reads every weight once per four tokens. The same
# generator lives in web/gpu.js. Accumulators are separate variables, not an indexed array,
# so they stay in registers.

def _shader_f16(tt: int) -> str:
    """float16 weights: each u32 holds two neighbouring columns of one row. 32 column pairs per
    workgroup, each summed by 8 threads over every 8th row, so a GPU's thousands of threads
    all stay busy even on small matrices."""
    r = range(tt)
    return "\n".join([
        "struct P { K: u32, N2: u32, T: u32, pad: u32 };",
        "@group(0) @binding(0) var<storage, read> W: array<u32>;",
        "@group(0) @binding(1) var<storage, read> X: array<f32>;",
        "@group(0) @binding(2) var<storage, read_write> Y: array<vec2<f32>>;",
        "@group(0) @binding(3) var<uniform> p: P;",
        *[f"var<workgroup> part{i}: array<vec2<f32>, 256>;" for i in r],
        "@compute @workgroup_size(32, 8)",
        "fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {",
        "  let n = wg.x * 32u + l.x;",
        f"  let t0 = wg.y * {tt}u;",
        *[f"  let x{i} = min(t0 + {i}u, p.T - 1u) * p.K;" for i in r],
        *[f"  var a{i} = vec2<f32>(0.0, 0.0);" for i in r],
        "  if (n < p.N2) {",
        "    for (var k = l.y; k < p.K; k = k + 8u) {",
        "      let w = unpack2x16float(W[k * p.N2 + n]);",
        *[f"      a{i} = a{i} + X[x{i} + k] * w;" for i in r],
        "    }",
        "  }",
        *[f"  part{i}[l.y * 32u + l.x] = a{i};" for i in r],
        "  workgroupBarrier();",
        "  if (l.y == 0u && n < p.N2) {",
        *[f"    if (t0 + {i}u < p.T) {{ var s = part{i}[l.x]; for (var j = 1u; j < 8u; j = j + 1u) {{ s = s + part{i}[j * 32u + l.x]; }} "
          f"Y[(t0 + {i}u) * p.N2 + n] = s; }}" for i in r],
        "  }",
        "}"])


def _shader_q8(tt: int) -> str:
    """8-bit weights: each u32 holds four consecutive rows of one column (signed, /127), and S
    holds one scale per group of G4 u32s (4 * G4 rows) per column. Each of a column's 8 threads
    takes whole groups, so it reads each scale once per group rather than once per row."""
    r = range(tt)
    return "\n".join([
        "struct P { K4: u32, N: u32, T: u32, G4: u32 };",
        "@group(0) @binding(0) var<storage, read> W: array<u32>;",
        "@group(0) @binding(1) var<storage, read> X: array<vec4<f32>>;",
        "@group(0) @binding(2) var<storage, read_write> Y: array<f32>;",
        "@group(0) @binding(3) var<uniform> p: P;",
        "@group(0) @binding(4) var<storage, read> S: array<f32>;",
        *[f"var<workgroup> part{i}: array<f32, 256>;" for i in r],
        "@compute @workgroup_size(32, 8)",
        "fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {",
        "  let n = wg.x * 32u + l.x;",
        f"  let t0 = wg.y * {tt}u;",
        *[f"  let x{i} = min(t0 + {i}u, p.T - 1u) * p.K4;" for i in r],
        *[f"  var a{i} = 0.0;" for i in r],
        "  if (n < p.N) {",
        "    let groups = p.K4 / p.G4;",
        "    for (var g = l.y; g < groups; g = g + 8u) {",
        *[f"      var s{i} = 0.0;" for i in r],
        "      let k0 = g * p.G4;",
        "      for (var k = k0; k < k0 + p.G4; k = k + 1u) {",
        "        let w = unpack4x8snorm(W[k * p.N + n]);",
        *[f"        s{i} = s{i} + dot(X[x{i} + k], w);" for i in r],
        "      }",
        "      let sc = S[g * p.N + n];",
        *[f"      a{i} = a{i} + s{i} * sc;" for i in r],
        "    }",
        "  }",
        *[f"  part{i}[l.y * 32u + l.x] = a{i};" for i in r],
        "  workgroupBarrier();",
        "  if (l.y == 0u && n < p.N) {",
        *[f"    if (t0 + {i}u < p.T) {{ var s = part{i}[l.x]; for (var j = 1u; j < 8u; j = j + 1u) {{ s = s + part{i}[j * 32u + l.x]; }} "
          f"Y[(t0 + {i}u) * p.N + n] = s; }}" for i in r],
        "  }",
        "}"])


SHADER_F16, SHADER_Q8 = _shader_f16(1), _shader_q8(1)
TOKENS = 4  # tokens per workgroup when reading a prompt


def _pick(kind: str, T: int) -> tuple[str, int]:
    return (kind, 1 if T == 1 else TOKENS)

CHUNK_BYTES = 120 << 20  # stay under the default 128 MB storage binding limit
GROUP = 128  # rows per 8-bit scale


class GPUMismatch(RuntimeError):
    pass


def quantize_q8(w: np.ndarray, group: int = GROUP) -> tuple[np.ndarray, np.ndarray]:
    """w [K, N] -> (q [ceil(K/4), N] uint32 of 4 signed bytes, scales [ceil(K/group), N] float32).
    Row k of column n is q_byte / 127 * scale[k // group, n]."""
    K, N = w.shape
    Kp = -(-K // group) * group
    wf = np.zeros((Kp, N), np.float32)
    wf[:K] = w
    g = wf.reshape(Kp // group, group, N)
    scale = np.abs(g).max(axis=1)
    scale[scale == 0] = 1.0
    q = np.clip(np.rint(g / scale[:, None, :] * 127), -127, 127).astype(np.int8).reshape(Kp, N)
    packed = np.ascontiguousarray(q.reshape(Kp // 4, 4, N).transpose(0, 2, 1)).view(np.uint32).reshape(Kp // 4, N)
    return packed, scale.astype(np.float32)


def dequantize_q8(packed: np.ndarray, scale: np.ndarray, K: int, group: int = GROUP) -> np.ndarray:
    K4, N = packed.shape
    q = packed.reshape(K4, N, 1).view(np.int8).reshape(K4, N, 4).transpose(0, 2, 1).reshape(K4 * 4, N)
    return (q.astype(np.float32) / 127 * np.repeat(scale, group, axis=0)[:K4 * 4])[:K]


def _check(gpu, name: str, K: int, ref_w) -> None:
    """Drivers differ: check one product against numpy before trusting a GPU with a matrix.
    ref_w(cols) gives what the GPU should hold for the first `cols` columns, as float32."""
    x = np.sin(np.arange(K, dtype=np.float32) * 12.9898)[None] * 0.5
    y = gpu.mm_many(x, [name])[0][0]
    cols = min(len(y), 512)
    ref = (x @ ref_w(cols))[0]
    y = y[:cols]
    if not np.all(np.isfinite(y)) or np.abs(y - ref).max() > 0.01 * max(np.abs(ref).max(), 1e-3) + 1e-3:
        raise GPUMismatch(f"{gpu.name} gave wrong results for {name}")


def _bench(gpu) -> float:
    """Effective GFLOP/s for one generation step through two 4096x4096 matrices (the size of
    a 7B model's attention weights), round trip included."""
    rng = np.random.default_rng(0)
    names = ["_bench0", "_bench1"]
    for nm in names:
        gpu.put(nm, rng.standard_normal((4096, 4096), dtype=np.float32).astype(np.float16))
    x = np.ones((1, 4096), np.float32)
    try:
        gpu.mm_many(x, names)
        n, t0 = 0, time.perf_counter()
        while time.perf_counter() - t0 < 0.5:
            gpu.mm_many(x, names)
            n += 1
        return round(n * 2 * 2 * 4096 * 4096 / (time.perf_counter() - t0) / 1e9, 2)
    finally:
        for nm in names:
            gpu.drop(nm)


class GPU:
    """WebGPU: holds weight matrices on one GPU and multiplies activations by them."""

    backend = "webgpu"

    def __init__(self, adapter, bits: int | None = None):
        import wgpu

        self.wgpu = wgpu
        info = adapter.info
        self.name = info.get("device") or info.get("description") or "GPU"
        self.software = info.get("adapter_type") == "CPU"
        self.bits = bits or (16 if os.environ.get("OMNICOMPUTE_WEIGHTS") == "16" else 8)
        lim = adapter.limits
        self.device = adapter.request_device_sync(required_limits={
            "max-storage-buffer-binding-size": lim["max-storage-buffer-binding-size"],
            "max-buffer-size": lim["max-buffer-size"]})
        self.max_bind = min(lim["max-storage-buffer-binding-size"], CHUNK_BYTES)
        # (kind, tokens per workgroup) -> pipeline
        self.pipelines = {(kind, tt): self.device.create_compute_pipeline(layout="auto", compute={
            "module": self.device.create_shader_module(code=make(tt)), "entry_point": "main"})
            for kind, make in (("f16", _shader_f16), ("q8", _shader_q8)) for tt in (1, TOKENS)}
        self.mats: dict[str, dict] = {}
        self._steps: dict[str, dict] = {}  # name -> reusable buffers for one-token steps
        self.lock = threading.Lock()  # steps run in worker threads

    # ---------- weights

    def _storage(self, a: np.ndarray):
        return self.device.create_buffer_with_data(data=np.ascontiguousarray(a).tobytes(),
                                                   usage=self.wgpu.BufferUsage.STORAGE)

    def put(self, name: str, w: np.ndarray) -> None:
        """w: [K, N] (float16 or float32), stored as column chunks."""
        K, N = w.shape
        chunks = []
        if self.bits == 8:
            q, s = quantize_q8(np.asarray(w, np.float32))
            per = max(1, self.max_bind // (q.shape[0] * 4))
            for c0 in range(0, N, per):
                cols = min(per, N - c0)
                chunks.append({"bufs": [self._storage(q[:, c0:c0 + cols]), self._storage(s[:, c0:c0 + cols])],
                               "c0": c0, "cols": cols, "units": cols,
                               "uniform": [q.shape[0], cols, 0, GROUP // 4]})
            self.mats[name] = {"K": K, "N": N, "kind": "q8", "chunks": chunks}
            _check(self, name, K, lambda cols: dequantize_q8(q[:, :cols], s[:, :cols], K))
            return
        h = np.ascontiguousarray(w, dtype=np.float16)
        if N % 2:
            h = np.concatenate([h, np.zeros((K, 1), np.float16)], 1)
        per = max(2, (self.max_bind // (K * 2)) // 2 * 2)  # columns per chunk, even
        for c0 in range(0, h.shape[1], per):
            part = h[:, c0:c0 + per]
            chunks.append({"bufs": [self._storage(part.view(np.uint32))], "c0": c0, "cols": part.shape[1],
                           "units": part.shape[1] // 2, "uniform": [K, part.shape[1] // 2, 0, 0]})
        self.mats[name] = {"K": K, "N": N, "kind": "f16", "chunks": chunks}
        _check(self, name, K, lambda cols: h[:, :cols].astype(np.float32))

    def has(self, name: str) -> bool:
        return name in self.mats

    def clear(self) -> None:
        for name in list(self.mats):
            self.drop(name)

    def drop(self, name: str) -> None:
        for c in self.mats.pop(name)["chunks"]:
            for b in c["bufs"]:
                b.destroy()
        for b in self._steps.pop(name, {}).get("all", []):
            b.destroy()

    # ---------- compute

    def mm_many(self, x: np.ndarray, names: list[str]) -> list[np.ndarray]:
        """[x @ W[name] for name in names], x: [T, K] float32, in one GPU submission."""
        with self.lock:
            return self._mm_many(x, names)

    def _buffers(self, name: str, T: int) -> tuple[object, list]:
        """Input buffer and per-chunk (output, readback, bind group, size) for one matrix."""
        if T == 1 and name in self._steps:  # generation: the same shapes every token
            return self._steps[name]["x"], self._steps[name]["chunks"]
        wgpu, dev, m = self.wgpu, self.device, self.mats[name]
        pipe = self.pipelines[_pick(m["kind"], T)]
        xbuf = dev.create_buffer(size=max(16, T * (-(-m["K"] // GROUP) * GROUP) * 4),
                                 usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
        made, chunks = [xbuf], []
        for c in m["chunks"]:
            size = T * c["cols"] * 4
            out = dev.create_buffer(size=size, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
            read = dev.create_buffer(size=size, usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST)
            u = list(c["uniform"])
            u[2] = T
            uni = dev.create_buffer_with_data(data=np.array(u, np.uint32).tobytes(), usage=wgpu.BufferUsage.UNIFORM)
            entries = [{"binding": 0, "resource": {"buffer": c["bufs"][0]}},
                       {"binding": 1, "resource": {"buffer": xbuf}},
                       {"binding": 2, "resource": {"buffer": out}},
                       {"binding": 3, "resource": {"buffer": uni}}]
            if m["kind"] == "q8":
                entries.append({"binding": 4, "resource": {"buffer": c["bufs"][1]}})
            bind = dev.create_bind_group(layout=pipe.get_bind_group_layout(0), entries=entries)
            made += [out, read, uni]
            chunks.append((c, out, read, bind, size))
        if T == 1:
            self._steps[name] = {"x": xbuf, "chunks": chunks, "all": made}
        else:
            self._temp += made
        return xbuf, chunks

    def _mm_many(self, x: np.ndarray, names: list[str]) -> list[np.ndarray]:
        wgpu, dev = self.wgpu, self.device
        x = np.ascontiguousarray(x, dtype=np.float32)
        T = x.shape[0]
        self._temp = []
        enc = dev.create_command_encoder()
        jobs = []
        for name in names:
            m = self.mats[name]
            xbuf, chunks = self._buffers(name, T)
            if m["kind"] == "q8":  # rows come in fours: pad the input to match
                K4 = m["chunks"][0]["uniform"][0]
                xp = np.zeros((T, K4 * 4), np.float32)
                xp[:, :m["K"]] = x
                dev.queue.write_buffer(xbuf, 0, xp.tobytes())
            else:
                dev.queue.write_buffer(xbuf, 0, x.tobytes())
            for c, out, read, bind, size in chunks:
                pas = enc.begin_compute_pass()
                kind = _pick(m["kind"], T)
                pas.set_pipeline(self.pipelines[kind])
                pas.set_bind_group(0, bind)
                pas.dispatch_workgroups((c["units"] + 31) // 32, -(-T // kind[1]), 1)
                pas.end()
                enc.copy_buffer_to_buffer(out, 0, read, 0, size)
                jobs.append((name, c, read))
        dev.queue.submit([enc.finish()])
        results = {n: np.empty((T, self.mats[n]["N"] + 1), np.float32) for n in names}
        for name, c, read in jobs:
            read.map_sync(wgpu.MapMode.READ)
            y = np.frombuffer(read.read_mapped(), np.float32).reshape(T, c["cols"])
            results[name][:, c["c0"]:c["c0"] + c["cols"]] = y
            read.unmap()
        for b in self._temp:
            b.destroy()
        return [results[n][:, :self.mats[n]["N"]] for n in names]

    def benchmark(self) -> float:
        return _bench(self)


class CudaGPU:
    """NVIDIA GPUs through PyTorch's CUDA build: float16 weights, cuBLAS products."""

    backend = "cuda"
    software = False
    bits = 16

    def __init__(self, torch, device: str = "cuda"):
        self.torch = torch
        self.dev = torch.device(device)
        self.name = torch.cuda.get_device_name(self.dev) if self.dev.type == "cuda" else f"torch {device}"
        self.dtype = torch.float16 if self.dev.type == "cuda" else torch.float32  # CPU torch: for tests
        self.mats: dict[str, object] = {}
        self.lock = threading.Lock()

    def put(self, name: str, w: np.ndarray) -> None:
        K, N = w.shape
        h = np.ascontiguousarray(w, dtype=np.float16)
        self.mats[name] = self.torch.from_numpy(h).to(self.dev).to(self.dtype)
        _check(self, name, K, lambda cols: h[:, :cols].astype(np.float32))

    def has(self, name: str) -> bool:
        return name in self.mats

    def drop(self, name: str) -> None:
        self.mats.pop(name, None)

    def clear(self) -> None:
        self.mats.clear()
        if self.dev.type == "cuda":
            self.torch.cuda.empty_cache()

    def mm_many(self, x: np.ndarray, names: list[str]) -> list[np.ndarray]:
        torch = self.torch
        with self.lock, torch.inference_mode():
            xt = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(self.dev).to(self.dtype)
            ys = [xt @ self.mats[n] for n in names]
            flat = torch.cat(ys, dim=1).float().cpu().numpy()  # one copy back for all of them
        out, c0 = [], 0
        for y in ys:
            out.append(flat[:, c0:c0 + y.shape[1]])
            c0 += y.shape[1]
        return out

    def benchmark(self) -> float:
        return _bench(self)


def _open_cuda():
    mode = os.environ.get("OMNICOMPUTE_GPU", "1")
    if mode == "webgpu":
        return None
    try:
        import torch
    except Exception:  # not installed (or broken): WebGPU instead
        return None
    try:
        if mode == "torch-cpu":  # tests: the CUDA code path, run by CPU torch
            return CudaGPU(torch, "cpu")
        if torch.cuda.is_available():
            return CudaGPU(torch)
    except Exception as e:
        log.info("CUDA is installed but not usable: %s", e)
    return None


def open_gpu(allow_software: bool | None = None):
    """The best GPU on this machine (CUDA first, then WebGPU), or None (no GPU, no backend
    installed, or OMNICOMPUTE_GPU=0). Software (CPU-emulated) adapters are skipped unless allowed,
    since numpy is faster."""
    if os.environ.get("OMNICOMPUTE_GPU", "1") == "0":
        return None
    cuda = _open_cuda()
    if cuda is not None:
        return cuda
    if allow_software is None:
        allow_software = os.environ.get("OMNICOMPUTE_GPU") == "force"
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
