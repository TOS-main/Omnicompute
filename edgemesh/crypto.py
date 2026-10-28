"""Mesh-wide crypto derived from one shared secret.

Every device in a mesh knows the mesh secret. From it we derive:
  * an AES-256-GCM key that seals every UDP packet and every relayed message,
    so the relay server and the networks in between see only ciphertext
    (the server can decrypt, since it knows the secret; see README);
  * an HMAC key used for the challenge/response login to the server.
"""

import hashlib
import hmac
import os

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF


def _derive(secret: str, info: bytes) -> bytes:
    return HKDF(hashes.SHA256(), 32, salt=b"edgemesh-v1", info=info).derive(secret.encode())


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
