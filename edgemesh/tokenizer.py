"""Tokenizers: raw bytes (random demo models), Hugging Face tokenizer.json files, and the
vocabulary inside GGUF files.

Only the hub tokenizes. It uses the fast `tokenizers` package when it is installed and
EdgeMesh's own pure-Python BPE (bpe.py) when it isn't, e.g. on a phone in Termux.
`jinja2` applies a model's chat template.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


DEFAULT_SYSTEM = "You are a helpful AI assistant. Answer clearly and concisely."


class ByteTokenizer:
    eos_ids: set[int] = set()
    has_chat_template = False

    def encode(self, text, chat: bool = False) -> list[int]:
        return list(_flatten(text).encode("utf-8"))

    def decode(self, ids: list[int]) -> str:
        return bytes(i % 256 for i in ids).decode("utf-8", errors="replace")


def _token_str(t):
    return t.get("content") if isinstance(t, dict) else t


class _Encoding:
    def __init__(self, ids):
        self.ids = ids


class _PureTokenizer:
    """The slice of the `tokenizers` API that HFTokenizer uses, on top of bpe.BPE."""

    def __init__(self, bpe, bos_id: int | None = None):
        self.bpe, self.bos_id = bpe, bos_id

    def encode(self, text: str, add_special_tokens: bool = True) -> _Encoding:
        ids = self.bpe.encode(text)
        if add_special_tokens and self.bos_id is not None and (not ids or ids[0] != self.bos_id):
            ids = [self.bos_id] + ids
        return _Encoding(ids)

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return self.bpe.decode(ids, skip_special_tokens)

    def token_to_id(self, tok: str):
        return self.bpe.token_to_id(tok)


def _pure_from_json(path: Path) -> _PureTokenizer:
    from . import bpe

    tj = json.loads(path.read_text(encoding="utf-8"))
    tok = bpe.from_tokenizer_json(tj)
    bos = None
    pp = tj.get("post_processor") or {}
    procs = pp.get("processors", [pp]) if pp.get("type") == "Sequence" else [pp]
    for p in procs:  # a template like "<s> $A" means encode() adds a BOS token
        single = p.get("single") if p.get("type") == "TemplateProcessing" else None
        if single and "SpecialToken" in single[0]:
            bos = tok.token_to_id(single[0]["SpecialToken"]["id"])
    return _PureTokenizer(tok, bos)


class HFTokenizer:
    def __init__(self, folder: str | Path, hf_config: dict | None = None):
        folder = Path(folder)
        try:
            from tokenizers import Tokenizer

            self.tok = Tokenizer.from_file(str(folder / "tokenizer.json"))
        except ImportError:  # no compiled tokenizers here (Termux): EdgeMesh's own BPE
            self.tok = _pure_from_json(folder / "tokenizer.json")
        tc = _read(folder / "tokenizer_config.json")
        gen = _read(folder / "generation_config.json")
        self.bos = _token_str(tc.get("bos_token")) or ""
        self.eos = _token_str(tc.get("eos_token")) or ""
        self.template = tc.get("chat_template")
        if isinstance(self.template, list):  # some repos ship several named templates
            self.template = next((t["template"] for t in self.template if t.get("name") == "default"),
                                 self.template[0]["template"])
        ids = set()
        for src in (gen.get("eos_token_id"), (hf_config or {}).get("eos_token_id")):
            ids.update(src if isinstance(src, list) else [src] if src is not None else [])
        if self.eos and self.tok.token_to_id(self.eos) is not None:
            ids.add(self.tok.token_to_id(self.eos))
        self.eos_ids = ids
        self.has_chat_template = bool(self.template)

    def apply_chat_template(self, messages: list[dict]) -> str:
        import jinja2

        env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)

        def raise_exception(msg):
            raise jinja2.TemplateError(msg)

        env.globals.update(raise_exception=raise_exception,
                           strftime_now=lambda fmt: datetime.now().strftime(fmt))
        env.filters["tojson"] = lambda v, indent=None, **_: json.dumps(v, indent=indent, ensure_ascii=False)
        return env.from_string(self.template).render(messages=messages, add_generation_prompt=True,
                                                     bos_token=self.bos, eos_token=self.eos)

    def encode(self, text, chat: bool = False) -> list[int]:
        """text: a prompt string, or a list of {"role", "content"} messages."""
        if chat and self.template:
            messages = text if isinstance(text, list) else [{"role": "system", "content": DEFAULT_SYSTEM},
                                                            {"role": "user", "content": text}]
            return self.tok.encode(self.apply_chat_template(messages), add_special_tokens=False).ids
        return self.tok.encode(_flatten(text)).ids

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=True)


class GGUFTokenizer(HFTokenizer):
    """The vocabulary, merges and chat template stored inside a GGUF file."""

    END_TOKENS = ("<|im_end|>", "<|eot_id|>", "<|end_of_text|>", "<|endoftext|>", "<|end|>", "</s>", "<end_of_turn>")

    def __init__(self, source):
        from . import bpe

        m = source.meta
        tokens = [str(t) for t in m("tokenizer.ggml.tokens", [])]
        types = list(m("tokenizer.ggml.token_type", [])) or [1] * len(tokens)
        vocab = {t: i for i, t in enumerate(tokens)}
        special = {t: i for i, t in enumerate(tokens) if i < len(types) and types[i] in (3, 4)}
        model = m("tokenizer.ggml.model", "gpt2")
        if model == "gpt2":
            merges = [tuple(str(x).split(" ", 1)) for x in m("tokenizer.ggml.merges", [])]
            split = bpe.GGUF_SPLITS.get(m("tokenizer.ggml.pre", ""), bpe.LLAMA3_SPLIT)
            tok = bpe.BPE(vocab, merges, kind="byte", split=split, special=special,
                          ignore_merges=m("tokenizer.ggml.pre", "") == "llama-bpe")
        elif model == "llama":
            tok = bpe.BPE(vocab, scores=[float(s) for s in m("tokenizer.ggml.scores", [])], kind="spm",
                          special=special, prefix_space=bool(m("tokenizer.ggml.add_space_prefix", True)),
                          unk_id=m("tokenizer.ggml.unknown_token_id"))
        else:
            raise ValueError(f"GGUF tokenizer '{model}' is not supported yet")
        bos_id, eos_id = m("tokenizer.ggml.bos_token_id"), m("tokenizer.ggml.eos_token_id")
        add_bos = m("tokenizer.ggml.add_bos_token", model == "llama")
        self.tok = _PureTokenizer(tok, bos_id if add_bos and bos_id is not None else None)
        self.bos = tokens[bos_id] if bos_id is not None and bos_id < len(tokens) else ""
        self.eos = tokens[eos_id] if eos_id is not None and eos_id < len(tokens) else ""
        self.template = m("tokenizer.chat_template")
        ids = {i for i in (eos_id, m("tokenizer.ggml.eot_token_id"), m("tokenizer.ggml.eom_token_id")) if i is not None}
        ids |= {special[t] for t in self.END_TOKENS if t in special}
        self.eos_ids = ids
        self.has_chat_template = bool(self.template)


def _flatten(text) -> str:
    """Chat history as plain text, for models without a chat template."""
    if isinstance(text, str):
        return text
    lines = [f"{m.get('role', 'user').capitalize()}: {m.get('content', '')}" for m in text]
    return "\n".join(lines) + "\nAssistant:"


def _read(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def load_tokenizer(source):
    if hasattr(source, "make_tokenizer"):
        return source.make_tokenizer()
    if source.cfg.get("tokenizer") == "hf":
        return HFTokenizer(source.folder, getattr(source, "hf_config", None))
    return ByteTokenizer()
