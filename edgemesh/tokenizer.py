"""Tokenizers: raw bytes (random demo models) and Hugging Face tokenizer.json files.

Only the hub tokenizes, so only the hub needs the `tokenizers` package (and
`jinja2` to apply a model's chat template).
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


class HFTokenizer:
    def __init__(self, folder: str | Path, hf_config: dict | None = None):
        from tokenizers import Tokenizer

        folder = Path(folder)
        self.tok = Tokenizer.from_file(str(folder / "tokenizer.json"))
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
    if source.cfg.get("tokenizer") == "hf":
        return HFTokenizer(source.folder, getattr(source, "hf_config", None))
    return ByteTokenizer()
