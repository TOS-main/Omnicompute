"""Load a model through the hub's "load from Hugging Face" API (against a local fake of the
Hugging Face file server), then generate across two devices and compare with transformers."""

import asyncio
import json

import numpy as np
import pytest
from aiohttp import ClientSession, web

import omnicompute.hf as hf
from test_mesh import KEY, model_file, start_mesh  # noqa: F401

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

WORDS = "the cat sat on a mat and dog ran to big red house hello world how are you".split()


def make_repo(folder):
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders

    specials = ["<unk>", "<s>", "</s>", "<|user|>", "<|assistant|>"]
    vocab = {t: i for i, t in enumerate(specials + WORDS)}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok.decoder = decoders.WordPiece(prefix="##")  # joins words with spaces
    tok.save(str(folder / "tokenizer.json"))
    (folder / "tokenizer_config.json").write_text(json.dumps({
        "bos_token": "<s>", "eos_token": "</s>",
        "chat_template": "{{ bos_token }}{% for m in messages %}<|{{ m['role'] }}|> {{ m['content'] }} {% endfor %}"
                         "{% if add_generation_prompt %}<|assistant|>{% endif %}"}))
    cfg = transformers.Qwen2Config(vocab_size=len(vocab), hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                                   num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
                                   tie_word_embeddings=True, eos_token_id=99)  # never hit: generation runs full length
    torch.manual_seed(1)
    model = transformers.AutoModelForCausalLM.from_config(cfg).eval()
    model.save_pretrained(folder, safe_serialization=True)
    return model, tok


async def fake_hub(folder):
    files = {p.name: p for p in folder.iterdir() if p.is_file()}

    async def tree(request):
        return web.json_response([{"type": "file", "path": n, "size": p.stat().st_size} for n, p in files.items()])

    async def resolve(request):
        p = files.get(request.match_info["name"])
        return web.FileResponse(p) if p else web.Response(status=404)

    app = web.Application()
    app.add_routes([web.get("/api/models/{org}/{repo}/tree/main", tree),
                    web.get("/{org}/{repo}/resolve/main/{name}", resolve)])
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    return runner, f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"


async def test_load_from_hugging_face_and_chat(tmp_path, model_file, monkeypatch):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    model, tok = make_repo(repo_dir)
    runner, url = await fake_hub(repo_dir)
    monkeypatch.setattr(hf, "HF", url)
    hub, nodes = await start_mesh(model_file[0], [("a", False, 2), ("b", False, 1)])
    hub.loader.cache = tmp_path / "cache"
    base = f"http://127.0.0.1:{hub.port}"
    try:
        async with ClientSession(headers={"X-Mesh-Key": KEY}) as s:
            async with s.post(base + "/api/model", json={"repo": "test-org/tiny-qwen"}) as r:
                assert r.status == 200
            for _ in range(200):
                async with s.get(base + "/api/model") as r:
                    st = await r.json()
                if st["download"]["state"] in ("ready", "error"):
                    break
                await asyncio.sleep(0.05)
            assert st["download"]["state"] == "ready", st
            assert st["model"]["name"] == "test-org/tiny-qwen" and st["chat_template"]

            async with s.post(base + "/api/generate", json={"prompt": "hello cat", "max_new": 6, "chat": True}) as r:
                events = [json.loads(line) async for line in r.content]
        tokens = [e["id"] for e in events if e["type"] == "token"]
        assert events[0]["type"] == "plan" and len(events[0]["plan"]) == 2

        prompt_ids = tok.encode("<s><|user|> hello cat <|assistant|>", add_special_tokens=False).ids
        with torch.no_grad():
            ref = model.generate(torch.tensor([prompt_ids]), max_new_tokens=6, do_sample=False,
                                 eos_token_id=None, pad_token_id=0)[0, len(prompt_ids):].tolist()
        assert tokens == ref
        assert events[-1]["type"] == "done" and events[-1]["text"] == tok.decode(ref)
    finally:
        for n in nodes:
            await n.close()
        await hub.close()
        await runner.cleanup()
