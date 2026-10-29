"""Stands in for llama.cpp's llama-server in tests: same flags, /health, and streaming
/v1/chat/completions and /completion answers (reasoning, then the answer, then timings)."""

import asyncio
import json
import sys
import time

from aiohttp import web

args = sys.argv[1:]
port = int(args[args.index("--port") + 1])
with open(args[args.index("-m") + 1] + ".args.json", "w") as f:
    json.dump(args, f)
started = time.monotonic()


async def health(request):
    return web.json_response({"status": "ok"} if time.monotonic() - started > 1 else {"error": "Loading model"},
                             status=200 if time.monotonic() - started > 1 else 503)


async def stream(request, chunks):
    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await resp.prepare(request)
    for c in chunks:
        await resp.write(f"data: {json.dumps(c)}\n\n".encode())
        await asyncio.sleep(0.01)
    await resp.write(b"data: [DONE]\n\n")
    return resp


async def chat(request):
    body = await request.json()
    n = min(body.get("max_tokens", 100), 3)
    pieces = [{"choices": [{"delta": {"reasoning_content": "two plus two "}}]},
              {"choices": [{"delta": {"reasoning_content": "is four"}}]}]
    pieces += [{"choices": [{"delta": {"content": w}}]} for w in ["2 + 2", " = ", "4"][:n]]
    pieces.append({"choices": [{"delta": {}, "finish_reason": "stop"}],
                   "timings": {"prompt_n": 9, "prompt_per_second": 640.0, "predicted_n": 5, "predicted_per_second": 64.0}})
    return await stream(request, pieces)


async def completion(request):
    return await stream(request, [{"content": "hello"}, {"content": " world", "stop": True,
                                                         "timings": {"predicted_n": 2, "predicted_per_second": 50.0}}])


app = web.Application()
app.add_routes([web.get("/health", health), web.post("/v1/chat/completions", chat), web.post("/completion", completion)])
web.run_app(app, host="127.0.0.1", port=port, print=None)
