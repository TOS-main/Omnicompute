"use strict";

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
let key = params.get("k") || safeGet("edgemesh.key") || "";
let worker = null;

function safeGet(k) { try { return localStorage.getItem(k); } catch { return null; } }
function safeSet(k, v) { try { localStorage.setItem(k, v); } catch { /* private mode */ } }

function deviceName() {
  let n = safeGet("edgemesh.name");
  if (!n) {
    const ua = navigator.userAgent;
    const kind = /iPhone/.test(ua) ? "iphone" : /iPad/.test(ua) ? "ipad" : /Android/.test(ua) ? "android"
      : /Mac/.test(ua) ? "mac" : /Windows/.test(ua) ? "windows" : "browser";
    n = `${kind}-${Math.random().toString(16).slice(2, 6)}`;
    safeSet("edgemesh.name", n);
  }
  return n;
}

function platformLabel() {
  const ua = navigator.userAgent;
  for (const re of [/Android [\d.]+/, /iPhone OS [\d_]+/, /iPad/, /Windows NT [\d.]+/, /Mac OS X [\d_]+/, /Linux/]) {
    const m = ua.match(re);
    if (m) return `browser on ${m[0].replace(/_/g, ".").replace("Windows NT 10.0", "Windows")}`;
  }
  return "browser";
}

function init() {
  if (!key) { $("keybox").hidden = false; return; }
  safeSet("edgemesh.key", key);
  setJoinLink(location.origin);
  refresh();
  setInterval(refresh, 3000);
}

function setJoinLink(origin) {
  const link = `${origin}/?k=${encodeURIComponent(key)}`;
  $("joinlink").value = link;
  $("joincmd").textContent = `edgemesh join "${link}"`;
}

$("keyform").onsubmit = (e) => {
  e.preventDefault();
  key = $("keyinput").value.trim();
  $("keybox").hidden = true;
  init();
};

$("copy").onclick = async () => {
  try { await navigator.clipboard.writeText($("joinlink").value); $("copy").textContent = "Copied"; }
  catch { $("joinlink").select(); }
};

function layerLabel(a, b) { return b - a === 1 ? `layer ${a}` : `layers ${a}–${b - 1}`; }

async function refresh() {
  let s;
  try {
    const r = await fetch("/api/status", { headers: { "X-Mesh-Key": key } });
    if (r.status === 401) { $("devices").innerHTML = "<li class='warn'>Wrong mesh key.</li>"; return; }
    s = await r.json();
  } catch { $("devices").innerHTML = "<li class='warn'>Hub unreachable.</li>"; return; }
  if (s.lan_origin && /^(localhost|127\.)/.test(location.hostname)) setJoinLink(s.lan_origin);
  $("modelname").textContent = s.model ? `· ${s.model.name}, ${s.model.n_layer} layers` : "";
  const layers = Object.fromEntries(s.plan.map(([n, a, b]) => [n, [a, b]]));
  const ul = $("devices");
  ul.innerHTML = "";
  if (!s.devices.length) ul.innerHTML = "<li class='muted'>No devices yet.</li>";
  for (const d of s.devices) {
    const li = document.createElement("li");
    const left = document.createElement("div");
    left.innerHTML = `<div></div><div class="tag"></div>`;
    left.children[0].textContent = d.node_id + (d.node_id === deviceName() && worker ? " (this device)" : "");
    left.children[1].textContent = `${d.platform || d.kind} · ${d.score ?? "?"} GFLOP/s · ${d.mem_mb ?? "?"} MB`
      + (d.kind === "native" ? (d.public_udp ? " · direct P2P ready" : " · relay") : " · via relay");
    const right = document.createElement("div");
    right.className = "layers";
    right.textContent = layers[d.node_id] ? layerLabel(...layers[d.node_id]) : "";
    li.append(left, right);
    ul.append(li);
  }
}

$("wtoggle").onclick = () => {
  if (worker) {
    worker.postMessage({ cmd: "stop" });
    setTimeout(() => { if (worker) worker.terminate(); worker = null; }, 500);
    $("wtoggle").textContent = "Contribute compute";
    $("wstatus").textContent = "Not contributing.";
    return;
  }
  if (!key) return;
  worker = new Worker("/static/worker.js");
  const mem = Math.min(1024, Math.round((navigator.deviceMemory || 2) * 1024 * 0.25));
  worker.postMessage({
    cmd: "start", key, name: deviceName(),
    wsUrl: `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`,
    info: { role: "worker", kind: "browser", platform: platformLabel(), mem_mb: mem },
  });
  worker.onmessage = (ev) => {
    const { state, note, stats } = ev.data;
    const label = { connecting: "Connecting…", ready: "Contributing", stopped: "Stopped", error: "Error" }[state];
    $("wstatus").className = state === "ready" ? "ok" : state === "error" ? "warn" : "muted";
    const held = stats && stats.layers ? ` · holding ${layerLabel(...stats.layers)}` : "";
    const steps = stats && stats.tokens ? ` · ${stats.tokens} steps run` : "";
    $("wstatus").textContent = `${label}${note && !held ? ": " + note : ""}${held}${steps}`;
  };
  $("wtoggle").textContent = "Stop";
  try { navigator.wakeLock && navigator.wakeLock.request("screen"); } catch { /* needs https */ }
};

$("run").onclick = async () => {
  const btn = $("run"), out = $("output"), stats = $("runstats");
  btn.disabled = true;
  const prompt = $("prompt").value;
  out.textContent = prompt;
  stats.textContent = "Starting…";
  try {
    const r = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Mesh-Key": key },
      body: JSON.stringify({ prompt, max_new: +$("maxnew").value, temperature: +$("temp").value }),
    });
    if (!r.ok) throw new Error(await r.text());
    const reader = r.body.getReader(), td = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += td.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf("\n")) >= 0) {
        const ev = JSON.parse(buf.slice(0, i));
        buf = buf.slice(i + 1);
        if (ev.type === "plan") stats.textContent = `Running on ${ev.plan.length} device(s)…`;
        else if (ev.type === "token") out.textContent = prompt + ev.text;
        else if (ev.type === "replan") stats.textContent = `A device dropped out; continuing on ${ev.plan.length}…`;
        else if (ev.type === "done") stats.textContent = `${ev.stats.tokens} tokens in ${ev.stats.seconds}s (${ev.stats.tok_per_s} tok/s)`;
        else if (ev.type === "error") stats.textContent = `Error: ${ev.error}`;
      }
    }
  } catch (e) {
    stats.textContent = `Error: ${e.message}`;
  } finally {
    btn.disabled = false;
    refresh();
  }
};

init();
