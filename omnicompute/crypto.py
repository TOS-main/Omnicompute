"""Mesh-wide crypto derived from one shared secret.

Every device in a mesh knows the mesh secret. From it we derive:
  * an AES-256-GCM key that seals every UDP packet and every relayed message,
    so the relay server and the networks in between see only ciphertext
    (the server can decrypt, since it knows the secret; see README);
  * an HMAC key used for the challenge/response login to the server.

Two keys exist per mesh. The full mesh key (the "k=" link) gives control: chat, models and
every device's limits. The contribute key (the "c=" link) is derived from it and only lets a
device join as a worker. Device-to-device traffic is sealed with the contribute key, so
contributors can take part in the chain; logging in with it gets no admin rights.
"""

import hashlib
import hmac
import os

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF


def _derive(secret: str, info: bytes) -> bytes:
    return HKDF(hashes.SHA256(), 32, salt=b"edgemesh-v1", info=info).derive(secret.encode())


def contribute_key(secret: str) -> str:
    """The contribute-only key for a mesh key (a contribute key maps to itself)."""
    if is_contribute_key(secret):
        return secret
    return "c" + _derive(secret, b"contribute")[:12].hex()


def is_contribute_key(key: str) -> bool:
    return len(key) == 25 and key[0] == "c" and all(ch in "0123456789abcdef" for ch in key[1:])


class MeshCipher:
    def __init__(self, secret: str):
        self._aead = AESGCM(_derive(secret, b"payload"))
        self._auth_key = _derive(secret, b"auth")

    def seal(self, data: bytes, aad: bytes = b"") -> bytes:
        nonce = os.urandom(12)
        return nonce + self._aead.encrypt(nonce, data, aad)

    def open(self, blob: bytes, aad: bytes = b"") -> bytes:
        """Raises cryptography.exceptions.InvalidTag on tampering or a wrong secret."""
        return self._aead.decrypt(blob[:12], blob[12:], aad)

    def sign(self, msg: bytes) -> str:
        return hmac.new(self._auth_key, msg, hashlib.sha256).hexdigest()

    def verify(self, msg: bytes, mac: str) -> bool:
        return hmac.compare_digest(self.sign(msg), mac)
