"""A pure-Python BPE tokenizer, used where the compiled `tokenizers` package won't install
(Termux on Android) and for GGUF files, which carry their vocabulary in the file itself.

Covers the two kinds of BPE that nearly all chat models use:
* byte-level BPE (GPT-2, Llama 3, Qwen 2/3, Phi-4, ...): text is split with a regex, each
  piece's bytes are mapped to printable characters, and merges are applied by rank;
* SentencePiece-style BPE (Llama 2, Mistral, Phi-3, TinyLlama, ...): spaces become "▁",
  pairs merge by rank (or by score, in GGUF files), and unknown characters fall back to bytes.
"""

from __future__ import annotations

import re

try:  # exact Unicode classes when the (optional) regex module is there
    import regex as _re

    _HAS_REGEX = True
except ImportError:
    _re = re
    _HAS_REGEX = False

GPT2_SPLIT = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
LLAMA3_SPLIT = (r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*"""
                r"""|\s*[\r\n]+|\s+(?!\S)|\s+""")
QWEN2_SPLIT = (r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*"""
               r"""|\s*[\r\n]+|\s+(?!\S)|\s+""")
# GGUF names the pre-tokenizer ("tokenizer.ggml.pre"); most byte-level models use one of these
GGUF_SPLITS = {"gpt-2": GPT2_SPLIT, "gpt2": GPT2_SPLIT, "qwen2": QWEN2_SPLIT, "deepseek-r1-qwen": QWEN2_SPLIT,
               "llama-bpe": LLAMA3_SPLIT, "llama3": LLAMA3_SPLIT, "smollm": GPT2_SPLIT}


def _compile(pattern: str):
    if _HAS_REGEX:
        return _re.compile(pattern)
    # Python's re has no \p{..}: letters are [^\W\d_], numbers \d
    p = pattern
    p = p.replace(r"[^\s\p{L}\p{N}]", r"(?:[^\s\w]|_)")
    p = p.replace(r"[^\r\n\p{L}\p{N}]", r"(?:[^\r\n\w]|_)")
    p = p.replace(r"\p{L}", r"[^\W\d_]").replace(r"\p{N}", r"\d")
    return re.compile(p)


def _byte_map() -> dict[int, str]:
    """GPT-2's reversible map from bytes to printable unicode characters."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


BYTE_TO_CHAR = _byte_map()
CHAR_TO_BYTE = {c: b for b, c in BYTE_TO_CHAR.items()}


class BPE:
    def __init__(self, vocab: dict[str, int], merges: list[tuple[str, str]] | None = None,
                 scores: list[float] | None = None, kind: str = "byte", split: str | None = GPT2_SPLIT,
                 special: dict[str, int] | None = None, prefix_space: bool = True, byte_fallback: bool = True,
                 ignore_merges: bool = False, unk_id: int | None = None, whole_lines: bool = False):
        self.vocab = vocab
        self.inv = {i: t for t, i in vocab.items()}
        for t, i in (special or {}).items():
            self.inv.setdefault(i, t)
        self.kind = kind  # "byte" or "spm"
        self.ranks = {tuple(m): r for r, m in enumerate(merges or [])}
        self.scores = scores
        self.special = dict(special or {})
        self.special_ids = set(self.special.values())
        self.split = _compile(split) if split and kind == "byte" else None
        self.prefix_space = prefix_space
        self.byte_fallback = byte_fallback
        self.ignore_merges = ignore_merges
        self.unk_id = unk_id
        self.whole_lines = whole_lines  # Gemma 4: merges run across words, only newlines split
        self._special_re = (re.compile("|".join(re.escape(t) for t in sorted(self.special, key=len, reverse=True)))
                            if self.special else None)
        self._cache: dict[str, list[int]] = {}

    # ---------- encoding

    def token_to_id(self, tok: str) -> int | None:
        return self.special.get(tok, self.vocab.get(tok))

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        pos, first = 0, True
        for m in self._special_re.finditer(text) if self._special_re else []:
            ids += self._encode_plain(text[pos:m.start()], first)
            ids.append(self.special[m.group()])
            pos, first = m.end(), False
        ids += self._encode_plain(text[pos:], first)
        return ids

    def _encode_plain(self, text: str, first: bool) -> list[int]:
        if not text:
            return []
        if self.kind == "byte":
            out = []
            for piece in self.split.findall(text) if self.split else [text]:
                if isinstance(piece, tuple):
                    piece = next(p for p in piece if p)
                out += self._word("".join(BYTE_TO_CHAR[b] for b in piece.encode("utf-8")))
            return out
        text = text.replace(" ", "▁")
        if first and self.prefix_space:
            text = "▁" + text
        if self.whole_lines:
            return [i for w in re.findall(r"[^\n]+|\n+", text)
                    for i in ([self.vocab[w]] if w[0] == "\n" and w in self.vocab else self._word(w))]
        # SentencePiece never merges across a word start, so each word is merged on its own
        return [i for w in re.findall(r"▁*[^▁]+|▁+", text) for i in self._word(w)]

    def _rank(self, a: str, b: str):
        if self.ranks:
            return self.ranks.get((a, b))
        if self.scores is not None:  # GGUF SentencePiece: merge the pair whose result scores highest
            i = self.vocab.get(a + b)
            return None if i is None else -self.scores[i]
        return None

    def _word(self, word: str) -> list[int]:
        hit = self._cache.get(word)
        if hit is not None:
            return hit
        if self.ignore_merges and word in self.vocab:
            return [self.vocab[word]]
        parts = list(word)
        while len(parts) > 1:
            best, at = None, -1
            for i in range(len(parts) - 1):
                r = self._rank(parts[i], parts[i + 1])
                if r is not None and (best is None or r < best):
                    best, at = r, i
            if best is None:
                break
            a, b = parts[at], parts[at + 1]
            merged, i = [], 0
            while i < len(parts):  # merge every occurrence of this pair, left to right
                if i < len(parts) - 1 and parts[i] == a and parts[i + 1] == b and self._rank(a, b) == best:
                    merged.append(a + b)
                    i += 2
                else:
                    merged.append(parts[i])
                    i += 1
            parts = merged
        ids = []
        for p in parts:
            i = self.vocab.get(p)
            if i is not None:
                ids.append(i)
            elif self.byte_fallback and self.kind == "spm":
                ids += [self.vocab.get(f"<0x{b:02X}>", self.unk_id or 0) for b in p.encode("utf-8")]
            elif self.unk_id is not None:
                ids.append(self.unk_id)
        if len(self._cache) < 50000:
            self._cache[word] = ids
        return ids

    # ---------- decoding

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        out = bytearray()
        for i in ids:
            if skip_special and i in self.special_ids:
                continue
            t = self.inv.get(int(i))
            if t is None:
                continue
            if i in self.special_ids:
                out += t.encode("utf-8")
            elif self.kind == "byte":
                out += bytes(CHAR_TO_BYTE.get(c, 32) for c in t) if all(c in CHAR_TO_BYTE for c in t) else t.encode("utf-8")
            elif len(t) == 6 and t.startswith("<0x") and t.endswith(">"):
                out.append(int(t[3:5], 16))
            else:
                out += t.replace("▁", " ").encode("utf-8")
        s = out.decode("utf-8", errors="replace")
        if self.kind == "spm" and self.prefix_space and s.startswith(" "):
            s = s[1:]
        return s


# ---------------------------------------------------------------- from tokenizer.json


def _walk(node, kind: str):
    """All parts of a (possibly Sequence) normalizer / pre-tokenizer / decoder of this type."""
    if not node:
        return []
    if node.get("type") == "Sequence":
        items = node.get("pretokenizers") or node.get("normalizers") or node.get("decoders") or []
        return [x for n in items for x in _walk(n, kind)]
    return [node] if node.get("type") == kind else []


def from_tokenizer_json(tj: dict) -> BPE:
    model = tj.get("model") or {}
    if model.get("type") != "BPE":
        raise ValueError(f"tokenizer type {model.get('type')} needs the tokenizers package")
    merges = [tuple(m.split(" ", 1)) if isinstance(m, str) else tuple(m) for m in model.get("merges", [])]
    special = {t["content"]: t["id"] for t in tj.get("added_tokens", [])}
    pre, norm = tj.get("pre_tokenizer"), tj.get("normalizer")
    if _walk(pre, "ByteLevel"):
        splits = _walk(pre, "Split")
        pattern = None
        if splits:
            pattern = splits[0]["pattern"].get("Regex") or re.escape(splits[0]["pattern"].get("String", ""))
        elif _walk(pre, "ByteLevel")[0].get("use_regex", True):
            pattern = GPT2_SPLIT
        return BPE(model["vocab"], merges, kind="byte", split=pattern, special=special,
                   ignore_merges=bool(model.get("ignore_merges")))
    meta = _walk(pre, "Metaspace")
    prepend = bool(_walk(norm, "Prepend")) or bool(meta and meta[0].get("prepend_scheme", "always") != "never")
    unk = model.get("unk_token")
    return BPE(model["vocab"], merges, kind="spm", special=special, prefix_space=prepend,
               byte_fallback=bool(model.get("byte_fallback", True)), unk_id=model["vocab"].get(unk) if unk else None)
