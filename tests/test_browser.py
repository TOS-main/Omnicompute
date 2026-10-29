"""End to end: a desktop worker plus two browser workers (as phones would run them)."""

import asyncio
import os

import pytest

from test_mesh import KEY, collect, model_file, reference, start_mesh  # noqa: F401

playwright = pytest.importorskip("playwright.async_api")


async def test_browser_workers_join_and_generate(model_file, monkeypatch):
    path, cfg, w = model_file
    monkeypatch.setattr("edgemesh.wsparts.PART", 4096)  # make the hub send layers in many parts
    hub, nodes = await start_mesh(path, [("pc", False, 1000)])
    url = f"http://127.0.0.1:{hub.port}/?k={KEY}"
    async with playwright.async_playwright() as p:
        exe = os.environ.get("EDGEMESH_CHROMIUM") or ("/opt/pw-browsers/chromium" if os.path.exists("/opt/pw-browsers/chromium") else None)
        try:
            browser = await p.chromium.launch()
        except Exception:
            browser = await p.chromium.launch(executable_path=exe, args=["--no-sandbox"])
        try:
            pages = []
            for i in range(2):
                ctx = await browser.new_context()
                page = await ctx.new_page()
                await page.goto(url)
                await page.evaluate(f"localStorage.setItem('edgemesh.name', JSON.stringify('phone{i}'));"
                                    "localStorage.setItem('edgemesh.limits', JSON.stringify({cpu: 100, always: true}))")
                await page.reload()
                await page.click("#tabs [data-tab=devices]")
                await page.click("#mode [data-mode=worker]")
                pages.append(page)
            for _ in range(100):
                if sum(c.info.get("kind") == "browser" for c in hub.clients.values()) == 2:
                    break
                await asyncio.sleep(0.1)
            # force every device into the chain: phones first, the fastest (pc) last
            for c in hub.clients.values():
                c.info["score"] = {"pc": 3, "phone0": 2, "phone1": 1}[c.node_id]
                c.info["limits"], c.info["limit"] = {"always": True}, 100
            assert all(c.info.get("cores", 0) >= 1 for c in hub.clients.values() if c.info["kind"] == "browser")
            events, ids = await collect(hub, "hi phones", 10)
            assert [p[0] for p in events[0]["plan"]] == ["phone0", "phone1", "pc"]
            assert ids == reference(cfg, w, "hi phones", 10)
            await pages[0].wait_for_function("document.getElementById('wstatus').textContent.includes('holding')", timeout=5000)

            # the web page can chat too (the phone is controller + worker at once)
            page = pages[1]
            await page.click("#tabs [data-tab=chat]")
            await page.evaluate("""() => { const t = document.getElementById('temp'); t.value = 0; t.dispatchEvent(new Event('input'));
                                          const m = document.getElementById('maxnew'); m.value = 16; m.dispatchEvent(new Event('input')); }""")
            await page.fill("#input", "hello there")
            await page.press("#input", "Enter")
            await page.wait_for_function("document.querySelector('.msg.ai .meta') && document.querySelector('.msg.ai .meta').textContent.includes('tok/s')", timeout=20000)
            assert await page.inner_text(".msg.user .bubble") == "hello there"
            await page.click("#tabs [data-tab=devices]")
            await page.wait_for_function("document.getElementById('devices').textContent.includes('phone1')", timeout=5000)
            assert (await page.inner_text("#devices")).count("layer") >= 3
            # who did how much of the answer, under the answer
            assert await page.locator(".msg.ai .contrib-chip").count() == 3

            # the other phone sees the same conversation (the main computer keeps it)
            other = pages[0]
            await other.click("#tabs [data-tab=chat]")
            await other.wait_for_function("document.querySelector('.msg.user .bubble') && document.querySelector('.msg.ai .contrib-chip')", timeout=10000)
            assert await other.inner_text(".msg.user .bubble") == "hello there"
            assert "from phone1" in await other.inner_text("#messages")

            # change the pc's usage limit from the phone's controller page; the pc stores and reports it
            pc_card = page.locator(".dev", has_text="pc")
            await pc_card.get_by_role("button", name="50%").click()
            for _ in range(50):
                if hub.clients["pc"].info.get("limit") == 50:
                    break
                await asyncio.sleep(0.1)
            assert hub.clients["pc"].info["limit"] == 50
        finally:
            await browser.close()
            for n in nodes:
                await n.close()
            await hub.close()


async def test_contribute_page_with_webgpu(model_file):
    """A phone on the contribute-only link, computing on its GPU through WebGPU."""
    path, cfg, w = model_file
    hub, nodes = await start_mesh(path, [("pc", False, 1000)])
    url = f"http://127.0.0.1:{hub.port}/contribute?c={hub.contribute_key}"
    async with playwright.async_playwright() as p:
        exe = os.environ.get("EDGEMESH_CHROMIUM") or ("/opt/pw-browsers/chromium" if os.path.exists("/opt/pw-browsers/chromium") else None)
        args = ["--no-sandbox", "--enable-unsafe-webgpu", "--use-webgpu-adapter=swiftshader"]  # software GPU for CI
        browser = await p.chromium.launch(executable_path=exe, args=args) if exe else await p.chromium.launch(args=args)
        try:
            page = await (await browser.new_context()).new_page()
            await page.goto(url)
            if not await page.evaluate("async () => !!navigator.gpu && !!(await navigator.gpu.requestAdapter())"):
                pytest.skip("this browser has no WebGPU adapter")
            await page.evaluate("localStorage.setItem('edgemesh.name', JSON.stringify('chromebook'));"
                                "localStorage.setItem('edgemesh.forcegpu', 'true');"
                                "localStorage.setItem('edgemesh.limits', JSON.stringify({cpu: 100, always: true}))")
            await page.reload()
            assert await page.locator("#tabs").count() == 0  # no chat, models or settings here
            await page.click("#toggle")
            for _ in range(150):
                if "chromebook" in hub.clients:
                    break
                await asyncio.sleep(0.1)
            assert hub.clients["chromebook"].contributor
            for _ in range(100):
                if hub.clients["chromebook"].info.get("uses") == "gpu":
                    break
                await asyncio.sleep(0.1)
            assert hub.clients["chromebook"].info["uses"] == "gpu"
            events, ids = await collect(hub, "webgpu", 8)
            assert [p[0] for p in events[0]["plan"]] == ["chromebook", "pc"]
            assert ids == reference(cfg, w, "webgpu", 8)
            await page.wait_for_function("document.getElementById('wstatus').textContent.includes('GPU')", timeout=5000)
            await page.wait_for_function("document.getElementById('lastcredit').textContent.includes('% of the work')", timeout=5000)
            assert not await page.locator("#livecard").is_hidden()  # it shows the answer it helped with
        finally:
            await browser.close()
            for n in nodes:
                await n.close()
            await hub.close()
