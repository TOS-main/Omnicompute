"""Split big WebSocket messages into 1 MB parts.

Tunnels and proxies (Cloudflare, nginx, school filters) often cap the size of a single
WebSocket message, and model shards can be hundreds of MB. Parts look like
b"EMCH" + 8-byte message id + 4-byte index + 4-byte count + data; anything else is a
whole message. web/worker.js speaks the same format.
"""

from __future__ import annotations

import os
import struct

MAGIC = b"EMCH"
PART = 1 << 20


def split(data: bytes) -> list[bytes]:
    if len(data) <= PART:
        return [data]
    mid, n = os.urandom(8), -(-len(data) // PART)
    return [MAGIC + mid + struct.pack("!II", i, n) + data[i * PART:(i + 1) * PART] for i in range(n)]


class Joiner:
    def __init__(self):
        self.parts: dict[bytes, dict] = {}

    def feed(self, data: bytes) -> bytes | None:
        """Returns a complete message, or None while parts are still arriving."""
        if not data.startswith(MAGIC):
            return data
        mid = data[4:12]
        i, n = struct.unpack("!II", data[12:20])
        p = self.parts.setdefault(mid, {})
        p[i] = data[20:]
        if len(p) < n:
            return None
        del self.parts[mid]
        return b"".join(p[j] for j in range(n))
