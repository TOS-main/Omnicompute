"""Tokenizers: raw bytes (for random test models) and GPT-2 byte-level BPE."""

from __future__ import annotations

import json
import os
import re


class ByteTokenizer:
    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: list[int]) -> str:
        return bytes(i % 256 for i in ids).decode("utf-8", errors="replace")


def _bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


class GPT2Tokenizer:
    # Same splitting as GPT-2's, using \w classes instead of \p{L}/\p{N} so only the stdlib is needed.
    PAT = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+""")

    def __init__(self, vocab_path: str, merges_path: str):
        with open(vocab_path, encoding="utf-8") as f:
            self.encoder = json.load(f)
        self.decoder = {v: k for k, v in self.encoder.items()}
        with open(merges_path, encoding="utf-8") as f:
            merges = [tuple(line.split()) for line in f.read().split("\n")[1:] if line.strip()]
        self.ranks = {m: i for i, m in enumerate(merges)}
        self.b2u = _bytes_to_unicode()
        self.u2b = {v: k for k, v in self.b2u.items()}
        self.cache: dict[str, list[str]] = {}

    def _bpe(self, token: str) -> list[str]:
        if token in self.cache:
            return self.cache[token]
        word = list(token)
        while len(word) > 1:
            pairs = [(self.ranks.get((a, b), 1 << 30), i) for i, (a, b) in enumerate(zip(word, word[1:]))]
            rank, i = min(pairs)
            if rank == 1 << 30:
                break
            first, second = word[i], word[i + 1]
            merged, j = [], 0
            while j < len(word):
                if j < len(word) - 1 and word[j] == first and word[j + 1] == second:
                    merged.append(first + second)
                    j += 2
                else:
                    merged.append(word[j])
                    j += 1
            word = merged
        self.cache[token] = word
        return word

    def encode(self, text: str) -> list[int]:
        ids = []
        for tok in self.PAT.findall(text):
            u = "".join(self.b2u[b] for b in tok.encode("utf-8"))
            ids.extend(self.encoder[p] for p in self._bpe(u))
        return ids

    def decode(self, ids: list[int]) -> str:
        text = "".join(self.decoder.get(i, "") for i in ids)
        return bytes(self.u2b[c] for c in text).decode("utf-8", errors="replace")


def load_tokenizer(cfg: dict, model_dir: str):
    if cfg.get("tokenizer") == "gpt2":
        return GPT2Tokenizer(os.path.join(model_dir, cfg["vocab_file"]), os.path.join(model_dir, cfg["merges_file"]))
    return ByteTokenizer()
