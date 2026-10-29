"""Wire format shared by every component (Python and the browser worker).

A message is a small JSON header followed by an opaque binary payload:

    [4-byte big-endian header length][JSON header][payload bytes]

Tensors use a deliberately simple layout so a browser can decode them with
a DataView and typed arrays, no numpy-format parser needed:

    [4-byte meta length][JSON meta: [{name, dtype, shape}, ...]][raw data...]

Each tensor's raw little-endian bytes follow in order, padded to 8 bytes.
"""

from __future__ import annotations

import json
import struct

import numpy as np

DTYPES = {"float32": np.float32, "int32": np.int32, "float16": np.float16}


def pack(header: dict, payload: bytes = b"") -> bytes:
    h = json.dumps(header, separators=(",", ":")).encode()
    return struct.pack("!I", len(h)) + h + payload


def unpack(data: bytes) -> tuple[dict, bytes]:
    (n,) = struct.unpack_from("!I", data)
    return json.loads(data[4 : 4 + n]), bytes(data[4 + n :])


def encode_tensors(arrays: dict) -> bytes:
    meta, chunks = [], []
    for name, a in arrays.items():
        a = np.ascontiguousarray(a)
        if a.dtype.name not in DTYPES:
            a = a.astype(np.float32)
        a = a.astype(a.dtype.newbyteorder("<"), copy=False)
        meta.append({"name": name, "dtype": a.dtype.name, "shape": list(a.shape)})
        raw = a.tobytes()
        chunks.append(raw + b"\0" * (-len(raw) % 8))
    m = json.dumps(meta, separators=(",", ":")).encode()
    m += b" " * (-(len(m) + 4) % 8)  # keep the first tensor 8-byte aligned
    return struct.pack("!I", len(m)) + m + b"".join(chunks)


def decode_tensors(data: bytes) -> dict:
    (n,) = struct.unpack_from("!I", data)
    meta = json.loads(data[4 : 4 + n])
    off, out = 4 + n, {}
    for t in meta:
        dt = np.dtype(DTYPES[t["dtype"]]).newbyteorder("<")
        count = int(np.prod(t["shape"], dtype=np.int64))
        out[t["name"]] = np.frombuffer(data, dt, count, off).reshape(t["shape"]).astype(dt.newbyteorder("="))
        off += count * dt.itemsize
        off += -off % 8
    return out
