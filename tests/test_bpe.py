"""EdgeMesh's pure-Python BPE gives the same tokens as the `tokenizers` package."""

import json
import re
import sys
from pathlib import Path

import pytest

tokenizers = pytest.importorskip("tokenizers")
from tokenizers import Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers, processors, trainers  # noqa: E402

from edgemesh.bpe import LLAMA3_SPLIT, QWEN2_SPLIT  # noqa: E402
from edgemesh.hf import pick_gguf  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ((ROOT / "README.md").read_text(encoding="utf-8") + (ROOT / "edgemesh" / "hub.py").read_text(encoding="utf-8")
          + " Héllo wörld, naïve café 東京 タワー Привет мир 12345 3.14159 don't I'll we've ✓✱ 🚀🔥\n\n\t tabs  double  spaces")
TESTS = ["Hello world!", "  leading spaces and trailing  ", "don't stop, I'LL go; we've 12345678 apples",
         "東京タワー Привет, мир! naïve café", "line1\nline2\r\n\n  indented\ttab", "emoji 🚀🔥 and ✓",
         "<|im_start|>user\nhi<|im_end|>\n", "def f(x):\n    return x**2  # comment", "a" * 50, CORPUS[:3000]]


def _train(kind, split, path):
    if kind == "byte":
        tok = Tokenizer(models.BPE(ignore_merges=split == LLAMA3_SPLIT))
        tok.pre_tokenizer = (pre_tokenizers.Sequence([pre_tokenizers.Split(Regex(split), "isolated"),
                                                      pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
                             if split else pre_tokenizers.ByteLevel(add_prefix_space=False))
        tok.decoder = decoders.ByteLevel()
        tr = trainers.BpeTrainer(vocab_size=1200, special_tokens=["<|endoftext|>", "<|im_start|>", "<|im_end|>"],
                                 initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    else:
        tok = Tokenizer(models.BPE(byte_fallback=True, unk_token="<unk>"))
        tok.normalizer = normalizers.Sequence([normalizers.Prepend("▁"), normalizers.Replace(" ", "▁")])
        tok.decoder = decoders.Sequence([decoders.Replace("▁", " "), decoders.ByteFallback(), decoders.Fuse(),
                                         decoders.Strip(" ", 1, 0)])
        tr = trainers.BpeTrainer(vocab_size=1200, special_tokens=["<unk>", "<s>", "</s>"] + [f"<0x{i:02X}>" for i in range(256)])
    tok.train_from_iterator([CORPUS], tr)
    if kind == "spm":
        tok.post_processor = processors.TemplateProcessing(single="<s> $A", special_tokens=[("<s>", tok.token_to_id("<s>"))])
    tok.save(str(path))
    if kind == "spm":  # like real SentencePiece vocabularies: no word start inside a token
        tj = json.loads(path.read_text(encoding="utf-8"))
        tj["model"]["merges"] = [m for m in tj["model"]["merges"]
                                 if re.fullmatch("▁*[^▁]*", "".join(m) if isinstance(m, list) else m.replace(" ", ""))]
        path.write_text(json.dumps(tj))
    return Tokenizer.from_file(str(path))


@pytest.mark.parametrize("no_regex", [False, True])
@pytest.mark.parametrize("kind,split", [("byte", None), ("byte", QWEN2_SPLIT), ("byte", LLAMA3_SPLIT), ("spm", None)])
def test_pure_bpe_matches_tokenizers(tmp_path, monkeypatch, kind, split, no_regex):
    if no_regex:  # Termux may not have the regex module either
        monkeypatch.setitem(sys.modules, "regex", None)
        monkeypatch.delitem(sys.modules, "edgemesh.bpe", raising=False)
    from edgemesh.tokenizer import _pure_from_json

    ref = _train(kind, split, tmp_path / "t.json")
    mine = _pure_from_json(tmp_path / "t.json")
    for t in TESTS:
        ids = ref.encode(t).ids
        assert mine.encode(t).ids == ids, t[:40]
        assert mine.decode(ids) == ref.decode(ids, skip_special_tokens=True)


def test_hf_tokenizer_works_without_the_tokenizers_package(tmp_path, monkeypatch):
    _train("byte", QWEN2_SPLIT, tmp_path / "tokenizer.json")
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"eos_token": "<|im_end|>"}))
    from edgemesh.tokenizer import HFTokenizer

    fast = HFTokenizer(tmp_path)
    monkeypatch.setitem(sys.modules, "tokenizers", None)
    slow = HFTokenizer(tmp_path)
    assert type(slow.tok).__name__ == "_PureTokenizer"
    assert slow.encode("Hello there, 東京!") == fast.encode("Hello there, 東京!")
    assert slow.eos_ids == fast.eos_ids


def test_gguf_file_choice():
    files = {"a/model-Q8_0.gguf": 8, "a/model-Q4_K_M.gguf": 4, "a/model-Q2_K.gguf": 2, "mmproj-f16.gguf": 1}
    assert pick_gguf(files) == "a/model-Q4_K_M.gguf"
    assert pick_gguf(files, "Q8_0") == "a/model-Q8_0.gguf"
    with pytest.raises(ValueError):
        pick_gguf(files, "Q3_K_S")
    assert pick_gguf({"x.safetensors": 1}) is None
