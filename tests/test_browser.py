"""End to end: a desktop worker plus two browser workers (as phones would run them)."""

import asyncio
import os

import pytest

from test_mesh import KEY, collect, model_file, reference, start_mesh  # noqa: F401

playwright = pytest.importorskip("playwright.async_api")


async def test_browser_workers_join_and_generate(model_file):
    path, cfg, w = model_file
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
                await page.evaluate(f"localStorage.setItem('edgemesh.name', 'phone{i}')")
                await page.reload()
                await page.click("#wtoggle")
                pages.append(page)
            for _ in range(100):
                if sum(c.info.get("kind") == "browser" for c in hub.clients.values()) == 2:
                    break
                await asyncio.sleep(0.1)
            # force every device into the chain: native first, then both phones
            for c in hub.clients.values():
                c.info["score"] = {"pc": 3, "phone0": 2, "phone1": 1}[c.node_id]
            events, ids = await collect(hub, "hi phones", 10)
            assert [p[0] for p in events[0]["plan"]] == ["pc", "phone0", "phone1"]
            assert ids == reference(cfg, w, "hi phones", 10)
            await pages[0].wait_for_function("document.getElementById('wstatus').textContent.includes('holding')", timeout=5000)

            # the web page can run a prompt too
            await pages[1].fill("#temp", "0")
            await pages[1].fill("#maxnew", "5")
            await pages[1].click("#run")
            await pages[1].wait_for_function("document.getElementById('runstats').textContent.includes('tok/s')", timeout=20000)
            await pages[1].wait_for_function("document.getElementById('devices').textContent.includes('phone1')", timeout=5000)
            assert (await pages[1].inner_text("#devices")).count("layer") == 3
        finally:
            await browser.close()
            for n in nodes:
                await n.close()
            await hub.close()
