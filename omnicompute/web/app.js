"use strict";

// Helpers, device identity, the limits editor and PageWorker live in common.js.

const params = new URLSearchParams(location.search);
let key = params.get("k") || store.get("key", "");
if (params.get("k")) history.replaceState(null, "", location.pathname + location.hash); // keep the key out of screenshots

async function api(path, opts = {}) {
  const r = await fetch(path, { ...opts, headers: { "X-Mesh-Key": key, ...(opts.body ? { "Content-Type": "application/json" } : {}), ...(opts.headers || {}) } });
  if (r.status === 401) throw new Error("wrong mesh key");
  return r;
}

// ================================================================ tabs

function showTab(name) {
  for (const b of document.querySelectorAll("#tabs button")) b.classList.toggle("active", b.dataset.tab === name);
  for (const p of document.querySelectorAll(".page")) p.classList.toggle("active", p.id === `page-${name}`);
  store.set("tab", name);
  if (name === "model") refreshEstimate();
}
for (const b of document.querySelectorAll("#tabs button")) b.onclick = () => showTab(b.dataset.tab);
document.addEventListener("click", (e) => {
  const g = e.target.closest("[data-goto]");
  if (g) { e.preventDefault(); showTab(g.dataset.goto); }
});

// ================================================================ state

const state = { status: null, model: null, liveLoad: {}, generating: null, ident: null };
let worker = null;
let myLimits = store.get("limits", { ...DEFAULT_LIMITS });
// the host moved (new tunnel or Wi-Fi address): find it through the permanent link's beacon
const follower = hostFollower(() => key, () => `/?k=${encodeURIComponent(key)}`);

// ================================================================ status polling

async function refreshStatus() {
  try {
    const r = await api("/api/status");
    state.status = await r.json();
  } catch (e) {
    if (e.message === "wrong mesh key") { askForKey(true); return false; }
    $("pill").textContent = "main computer unreachable";
    $("pill").className = "pill bad";
    follower.lost();
    return true;
  }
  follower.ok();
  const s = state.status, n = s.devices.filter((d) => d.role === "worker").length;
  $("pill").className = "pill ok";
  $("pill").textContent = `${s.model ? s.model.name.split("/").pop() : "no model"} · ${n} device${n === 1 ? "" : "s"}${s.busy ? " · generating" : ""}`;
  $("devcount").textContent = n || "";
  renderDevices();
  applyModel(s.model);
  state.ckey = s.contribute_key;
  if (s.public_origin) setJoin(s.public_origin); // works from any network
  else if (s.lan_origin && /^(localhost|127\.)/.test(location.hostname)) setJoin(s.lan_origin);
  else setJoin(location.origin);
  if (s.chat_version !== state.chatVersion && !state.generating) { loadChat(); refreshChats(); }
  drawCapacity(s.capacity);
  return true;
}

// ================================================================ devices page

const cards = new Map(); // node_id -> {card, fields}

function renderDevices() {
  const s = state.status;
  if (!s) return;
  const plan = Object.fromEntries(s.plan.map(([n, a, b]) => [n, [a, b]]));
  const grid = $("devices");
  const seen = new Set();
  const workers = s.devices.filter((d) => d.role === "worker");
  if (!workers.length && !grid.querySelector(".empty-dev")) grid.replaceChildren(el("p", { class: "muted empty-dev" }, "No devices are contributing yet. Use the Connect tab to add some."));
  if (workers.length) grid.querySelector(".empty-dev")?.remove();
  for (const d of workers) {
    seen.add(d.node_id);
    let c = cards.get(d.node_id);
    if (!c) c = makeCard(d);
    const layers = plan[d.node_id];
    fill(c.badges,
      el("span", { class: "badge" }, d.kind === "browser" ? "browser" : "app"),
      d.kind === "native" && d.direct !== false ? el("span", { class: "badge on", title: "can send directly to other app devices" }, "direct P2P")
        : el("span", { class: "badge relay", title: "traffic goes through the main computer" }, "relay"),
      d.uses === "gpu" || d.uses === "gpu+cpu" ? el("span", { class: "badge on", title: d.gpu && d.gpu.name },
        (d.gpu && d.gpu.via === "cuda" ? "GPU · CUDA" : "GPU") + (d.uses === "gpu+cpu" ? " + CPU" : "")) : null,
      d.power ? el("span", { class: "badge" }, d.power === "charging" ? "charging" : "on battery") : null,
      d.contributor ? el("span", { class: "badge", title: "joined with the contribute-only link" }, "contributor") : null,
      d.is_hub_machine ? el("span", { class: "badge" }, "main computer") : null,
      d.node_id === (state.ident && state.ident.name) && worker ? el("span", { class: "badge on" }, "this device") : null,
      layers ? el("span", { class: "badge on" }, layers[1] - layers[0] === 1 ? `layer ${layers[0]}` : `layers ${layers[0]}–${layers[1] - 1}`)
        : d.limit === 0 ? el("span", { class: "badge" }, "paused")
        : s.plan.length ? el("span", { class: "badge", title: "the mesh is faster without it right now" }, "standby")
        : el("span", { class: "badge", title: "gets layers when you send a prompt" }, "ready"));
    const load = state.liveLoad[d.node_id], h = d.helped;
    fill(c.stats,
      stat(d.score != null ? d.score.toFixed(1) : "?", d.uses === "gpu" ? "GFLOP/s (GPU)" : d.uses === "gpu+cpu" ? "GFLOP/s (GPU + CPU)" : "GFLOP/s"),
      stat(d.mem_mb ? gb(d.mem_mb * 2 ** 20) : "?", "memory offered"),
      stat(d.rtt_ms != null ? `${Math.round(d.rtt_ms)} ms` : "…", "round trip"),
      stat(d.limit === 0 ? "paused" : `${d.limit}%`, d.limit_reason && d.limit_reason !== "default" ? d.limit_reason : "usage limit"),
      stat(load != null ? `${Math.round(load)}%` : "—", "load (last run)"),
      stat(h ? h.tokens : 0, h ? `tokens · ${h.answers} answers` : "tokens helped"),
      d.cores ? stat(d.cores, "cores") : null,
      d.gpu && d.uses === "cpu" ? stat(d.gpu.score.toFixed(1), `GPU GFLOP/s (${d.gpu.name})`.slice(0, 40)) : null);
    c.platform.textContent = d.platform || "";
    if (!c.card.contains(document.activeElement) && JSON.stringify(d.limits) !== c.lastLimits) {
      c.lastLimits = JSON.stringify(d.limits);
      c.limitSlot.replaceChildren(limitEditor(d.limits, (l) => setRemoteLimits(d.node_id, l, c), { cores: d.cores_total, awake: d.kind === "native" }));
    }
  }
  for (const [id, c] of cards) if (!seen.has(id)) { c.card.remove(); cards.delete(id); }
  const est = s.plan.length ? `Split over ${s.plan.length} device${s.plan.length > 1 ? "s" : ""}` : "";
  $("planline").textContent = est + (s.last_run && s.last_run.tok_per_s ? ` · last run ${s.last_run.tok_per_s} tok/s` : "");
}

function makeCard(d) {
  const c = { badges: el("div", { class: "badges" }), stats: el("div", { class: "stats" }), platform: el("div", { class: "muted small" }),
    limitSlot: el("div"), msg: el("p", { class: "small muted" }) };
  const remove = d.is_hub_machine ? null : el("button", { class: "ghost small danger", type: "button", title: "take it off the mesh",
    onclick: () => removeDevice(d, c) }, "Remove");
  c.card = el("div", { class: "card dev" }, el("div", { class: "head row space" }, el("div", {}, el("div", { class: "name" }, d.node_id), c.platform), remove),
    c.badges, c.stats, c.limitSlot, c.msg);
  $("devices").append(c.card);
  cards.set(d.node_id, c);
  return c;
}

async function removeDevice(d, c) {
  const what = d.kind === "browser" ? "It stops contributing and forgets the link."
    : "It stops, won't start with the computer anymore, and if it was set up with the one-line installer, Omnicompute is uninstalled from it.";
  if (!confirm(`Remove ${d.node_id} from the mesh? ${what}`)) return;
  c.msg.textContent = "Removing…";
  try {
    const r = await api(`/api/devices/${encodeURIComponent(d.node_id)}/remove`, { method: "POST" });
    if (!r.ok) throw new Error(await r.text());
    const res = await r.json();
    c.msg.textContent = res.ok ? (res.removed && res.removed.length ? "Removed and uninstalled." : "Removed.") : `Disconnected, but it said: ${res.error}`;
  } catch (e) { c.msg.textContent = `Not removed: ${e.message}`; }
}

async function setRemoteLimits(node, limits, c) {
  c.msg.textContent = "Saving…";
  try {
    const r = await api(`/api/devices/${encodeURIComponent(node)}/limits`, { method: "POST", body: JSON.stringify(limits) });
    if (!r.ok) throw new Error(await r.text());
    c.lastLimits = JSON.stringify(await r.json());
    c.msg.textContent = "Saved on the device.";
    if (state.ident && node === state.ident.name) { myLimits = limits; store.set("limits", limits); drawMyLimits(); }
  } catch (e) { c.msg.textContent = `Not saved: ${e.message}`; }
  setTimeout(() => { c.msg.textContent = ""; }, 3000);
}

// ---------- this device: controller / worker

function drawMyLimits() {
  $("mylimits").replaceChildren(limitEditor(myLimits, (l) => {
    myLimits = l;
    store.set("limits", l);
    if (worker) worker.limits(l);
  }, { label: "When working", cores: browserCores() }));
}

function setMode(mode) {
  for (const b of document.querySelectorAll("#mode button")) b.classList.toggle("active", b.dataset.mode === mode);
  store.set("mode", mode);
  if (mode === "worker" && !worker) startWorker();
  if (mode === "controller" && worker) stopWorker();
}
for (const b of document.querySelectorAll("#mode button")) b.onclick = () => setMode(b.dataset.mode);

function startWorker() {
  worker = new PageWorker(key, state.ident, myLimits, {
    onState: (d) => { $("wstatus").className = d.state === "ready" ? "ok" : d.state === "error" ? "bad" : "muted"; $("wstatus").textContent = workerStatusText(d); },
    onLimits: (l) => { myLimits = l; store.set("limits", l); drawMyLimits(); },
    onRemoved: () => { worker = null; keepAwake.stop(); for (const b of document.querySelectorAll("#mode button")) b.classList.toggle("active", b.dataset.mode === "controller"); },
  });
  if (store.get("background", false)) keepAwake.start();
}

function stopWorker() {
  worker.stop();
  worker = null;
  keepAwake.stop();
  $("wstatus").className = "muted";
  $("wstatus").textContent = "Controller only: runs prompts and manages the mesh.";
}

$("bg").checked = store.get("background", false);
$("bg").onchange = () => { store.set("background", $("bg").checked); if (worker && $("bg").checked) keepAwake.start(); else keepAwake.stop(); };

// ================================================================ markdown (small, safe: escapes all HTML)

const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

function inline(s) {
  const codes = [];
  s = s.replace(/`([^`\n]+)`/g, (_, c) => `\u0000${codes.push(c) - 1}\u0000`);
  s = esc(s)
    .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*\w])\*([^*\n]+)\*(?!\w)/g, "$1<em>$2</em>")
    .replace(/\[([^\]\n]+)\]\((https?:[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
  return s.replace(/\u0000(\d+)\u0000/g, (_, i) => `<code>${esc(codes[+i])}</code>`);
}

function renderMarkdown(src, streaming) {
  const out = [], lines = src.split("\n");
  let i = 0, para = [];
  const flush = () => { if (para.length) { out.push(`<p>${inline(para.join("\n")).replace(/\n/g, "<br>")}</p>`); para = []; } };
  while (i < lines.length) {
    const line = lines[i];
    const fence = line.match(/^\s*```\s*([\w+#.-]*)/);
    if (fence) {
      flush();
      const lang = fence[1], body = [];
      i++;
      while (i < lines.length && !/^\s*```/.test(lines[i])) body.push(lines[i++]);
      const open = i >= lines.length && streaming;
      i++;
      out.push(`<div class="codeblock"><div class="bar"><span>${esc(lang || "code")}${open ? " …" : ""}</span><button type="button" data-copy>Copy</button></div><pre><code>${esc(body.join("\n"))}</code></pre></div>`);
      continue;
    }
    let m;
    if ((m = line.match(/^(#{1,4})\s+(.*)/))) { flush(); out.push(`<h${m[1].length + 1}>${inline(m[2])}</h${m[1].length + 1}>`); i++; continue; }
    if (/^\s*([-*+]|\d+[.)])\s+/.test(line)) {
      flush();
      const ordered = /^\s*\d/.test(line), items = [];
      while (i < lines.length && /^\s*([-*+]|\d+[.)])\s+/.test(lines[i])) items.push(lines[i++].replace(/^\s*([-*+]|\d+[.)])\s+/, ""));
      out.push(`<${ordered ? "ol" : "ul"}>${items.map((t) => `<li>${inline(t)}</li>`).join("")}</${ordered ? "ol" : "ul"}>`);
      continue;
    }
    if (/^>\s?/.test(line)) { flush(); const q = []; while (i < lines.length && /^>\s?/.test(lines[i])) q.push(lines[i++].replace(/^>\s?/, "")); out.push(`<blockquote>${inline(q.join(" "))}</blockquote>`); continue; }
    if (/^\s*\|.*\|\s*$/.test(line) && i + 1 < lines.length && /^\s*\|[\s:|-]+\|\s*$/.test(lines[i + 1])) {
      flush();
      const cells = (l) => l.trim().replace(/^\||\|$/g, "").split("|").map((c) => inline(c.trim()));
      const head = cells(line);
      i += 2;
      const rows = [];
      while (i < lines.length && /^\s*\|.*\|\s*$/.test(lines[i])) rows.push(cells(lines[i++]));
      out.push(`<table><tr>${head.map((h) => `<th>${h}</th>`).join("")}</tr>${rows.map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join("")}</tr>`).join("")}</table>`);
      continue;
    }
    if (!line.trim()) { flush(); i++; continue; }
    para.push(line);
    i++;
  }
  flush();
  return out.join("");
}

document.addEventListener("click", async (e) => {
  const b = e.target.closest("[data-copy]");
  if (!b) return;
  const code = b.closest(".codeblock").querySelector("code").textContent;
  try { await navigator.clipboard.writeText(code); b.textContent = "Copied"; } catch { b.textContent = "Select & copy"; }
  setTimeout(() => { b.textContent = "Copy"; }, 1500);
});

// ================================================================ chat

let chat = []; // the shared conversation, kept by the main computer so every controller sees it
const settings = { system: "You are a helpful AI assistant. Answer clearly and concisely.", maxnew: 4096, ctx: null, temp: 0.7, chatfmt: true, markdown: true, ...store.get("settings", {}) };

if ((settings.v || 1) < 2) { // older pages saved 512 new tokens; start again from the new defaults
  settings.v = 2; settings.defaultsFor = null; settings.ctx = null; settings.maxnew = Math.max(settings.maxnew || 0, 4096);
}
function saveSettings() { store.set("settings", settings); }

function bindSetting(id, keyName, fmt, parse = Number) {
  const input = $(id), out = $(`${id}-out`);
  const show = () => { if (out) out.textContent = fmt(settings[keyName]); };
  if (input.type === "checkbox") input.checked = settings[keyName];
  else if (settings[keyName] != null) input.value = settings[keyName];
  input.addEventListener(input.type === "checkbox" ? "change" : "input", () => {
    settings[keyName] = input.type === "checkbox" ? input.checked : parse(input.value);
    show(); saveSettings();
    if (keyName === "ctx") clampMaxNew();
    if (keyName === "markdown") renderChat();
    updateCtxMeter();
  });
  show();
}
bindSetting("system", "system", () => "", String);
bindSetting("maxnew", "maxnew", (v) => `${v} tokens`);
bindSetting("ctx", "ctx", (v) => `${v} tokens`);
bindSetting("temp", "temp", (v) => (v === 0 ? "0 (greedy)" : v.toFixed(2)));
bindSetting("chatfmt", "chatfmt", () => "");
bindSetting("markdown", "markdown", () => "");

const MIN_NEW = 4096;
function clampMaxNew() {
  const mx = $("maxnew");
  mx.max = Math.max(16, settings.ctx || 4096);
  if (settings.maxnew > +mx.max) { settings.maxnew = +mx.max; mx.value = mx.max; $("maxnew-out").textContent = `${mx.max} tokens`; }
}

function applyModel(cfg) {
  const name = cfg ? cfg.name : null;
  $("demo-banner").hidden = !(cfg && cfg.tokenizer === "bytes");
  if (!cfg) { $("modelline").textContent = "No model loaded."; return; }
  const limit = cfg.n_ctx, ctx = $("ctx");
  if (state.model !== name) {
    state.model = name;
    ctx.max = limit;
    ctx.step = limit >= 8192 ? 512 : limit >= 2048 ? 256 : 64;
    ctx.min = Math.min(256, limit);
    if (!settings.ctx || settings.ctx > limit) settings.ctx = Math.min(cfg.ctx_default || limit, limit);
    ctx.value = settings.ctx;
    $("ctx-out").textContent = `${settings.ctx} tokens`;
    clampMaxNew();
    if (settings.defaultsFor !== name) { // a new model: start from its own defaults
      const g = cfg.gen_defaults || {};
      settings.defaultsFor = name;
      settings.maxnew = Math.min(Math.max(g.max_new_tokens || 0, MIN_NEW), settings.ctx); // at least ~4000 when the context allows
      if (g.temperature != null) settings.temp = g.temperature;
      $("maxnew").value = settings.maxnew; $("maxnew-out").textContent = `${settings.maxnew} tokens`;
      $("temp").value = settings.temp; $("temp-out").textContent = settings.temp === 0 ? "0 (greedy)" : settings.temp.toFixed(2);
      saveSettings();
    }
    $("maxnew-hint").textContent = `Model default: ${(cfg.gen_defaults && cfg.gen_defaults.max_new_tokens) ? cfg.gen_defaults.max_new_tokens.toLocaleString() + " tokens" : "until it finishes or the context is full"}.`;
    refreshModelInfo();
  }
  $("ctx-hint").textContent = `Model limit: ${limit.toLocaleString()} tokens. Longer contexts use more memory on every device.`;
  $("modelline").textContent = `${cfg.name} · ${cfg.n_layer} layers · ${cfg.hf_type || cfg.arch || "gpt2"}`;
}

let lastCtxUse = store.get("ctxuse", 0);
function updateCtxMeter() {
  const max = settings.ctx || 4096, used = Math.min(lastCtxUse, max);
  $("ctxfill").style.width = `${(100 * used) / max}%`;
  $("ctxfill").style.background = used > max * 0.9 ? "var(--bad)" : "";
  $("ctxtext").textContent = `Context: ${used.toLocaleString()} / ${max.toLocaleString()} tokens${used > max * 0.9 ? " (oldest messages will be cut)" : ""}`;
}

async function loadChat() {
  try {
    const r = await (await api("/api/chat")).json();
    if (state.generating) return;
    state.chatVersion = r.version;
    chat = r.messages;
    renderChat();
    const live = chat.some((m) => m.live);
    clearTimeout(loadChat.t);
    if (live) loadChat.t = setTimeout(loadChat, 700); // someone else's answer is streaming in
  } catch { /* status polling shows the problem */ }
}

function renderChat() {
  const box = $("messages");
  box.replaceChildren();
  if (!chat.length) box.append(el("div", { class: "empty" }, el("img", { src: "/static/icon.svg", alt: "" }), el("h2", {}, "Ask anything"),
    el("p", { class: "muted" }, "Your prompt runs on every device in the mesh.")));
  const nearBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 80 || !box.dataset.drawn;
  chat.forEach((m, i) => box.append(messageEl(m, (i === chat.length - 1 && state.generating) || m.live)));
  if (nearBottom) box.scrollTop = box.scrollHeight;
  box.dataset.drawn = "1";
}

function statsLine(st) {
  return `${st.tokens} tokens · ${st.tok_per_s ?? "–"} tok/s · first token ${st.first_token_s ?? "–"}s · ${st.devices} device(s)`
    + (st.stopped === "length" ? " · hit the max-token limit" : st.stopped === "context" ? " · context full" : "");
}

// who did how much of this answer: share of the compute time, and the layers each device ran
function contributionEl(list) {
  if (!list || !list.length) return null;
  return el("div", { class: "contrib-row" }, list.map((c) => el("span", { class: "contrib-chip",
    title: `layers ${c.layers[0]}–${c.layers[1] - 1} · ${c.compute_s}s compute · ${c.ms_per_token} ms per token` },
    el("i", { style: `width:${Math.max(4, c.share)}%` }), `${c.node_id} ${Math.round(c.share)}%`)));
}

function messageEl(m, streaming) {
  const me = state.ident && state.ident.name;
  if (m.role === "user") return el("div", { class: "msg user" }, m.by && m.by !== me ? el("div", { class: "who" }, `from ${m.by}`) : null,
    el("div", { class: "bubble" }, m.content));
  const bubble = el("div", { class: `bubble${settings.markdown ? "" : " plain"}${streaming ? " cursor" : ""}` });
  if (settings.markdown) bubble.innerHTML = renderMarkdown(m.content || "", streaming);
  else bubble.textContent = m.content;
  if (streaming && !m.content) bubble.textContent = "";
  const meta = m.meta || (m.stats ? statsLine(m.stats) : m.stopped ? "stopped" : null);
  return el("div", { class: "msg ai" }, el("div", { class: "who" }, m.model ? m.model.split("/").pop() : "assistant"), bubble,
    meta ? el("div", { class: "meta" }, meta) : null, contributionEl(m.stats && m.stats.contribution));
}

function updateLastMessage(streaming) {
  const box = $("messages"), last = box.lastElementChild, m = chat[chat.length - 1];
  const fresh = messageEl(m, streaming);
  const nearBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 80;
  if (last && last.classList.contains("ai")) last.replaceWith(fresh); else box.append(fresh);
  if (nearBottom) box.scrollTop = box.scrollHeight;
}

function showLive(stats, load) {
  $("live").hidden = false;
  $("livestats").textContent = stats;
  const names = Object.keys(load || {});
  $("liveload").replaceChildren(...names.map((n) => el("div", { class: "loaditem" }, el("span", {}, n),
    el("div", { class: "meter" }, el("div", { style: `width:${Math.min(100, load[n])}%` })), `${Math.round(load[n])}%`)));
}

async function send(text) {
  chat.push({ role: "user", content: text, by: state.ident.name });
  const reply = { role: "assistant", content: "", model: state.model };
  chat.push(reply);
  const ctrl = new AbortController();
  state.generating = ctrl;
  $("send").textContent = "Stop";
  $("send").classList.add("stop");
  renderChat();
  const history = chat.slice(0, -1).filter((m) => m.content).map(({ role, content }) => ({ role, content }));
  const messages = settings.system.trim() ? [{ role: "system", content: settings.system.trim() }, ...history] : history;
  const t0 = performance.now();
  let tokens = 0, first = null, promptTokens = 0, est = null;
  showLive("Planning the split…", {});
  try {
    const r = await api("/api/generate", { method: "POST", signal: ctrl.signal, body: JSON.stringify({
      messages, chat: settings.chatfmt, max_new: settings.maxnew, temperature: settings.temp, context: settings.ctx,
      record: { user: text, by: state.ident.name } }) });
    if (!r.ok) throw new Error(await r.text());
    const reader = r.body.getReader(), td = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += td.decode(value, { stream: true });
      let nl;
      while ((nl = buf.indexOf("\n")) >= 0) {
        const ev = JSON.parse(buf.slice(0, nl));
        buf = buf.slice(nl + 1);
        if (ev.type === "plan") {
          promptTokens = ev.prompt_tokens; est = ev.estimate;
          showLive(`Prompt ${promptTokens} tokens · ${ev.plan.length} device(s)${est ? ` · est. ${est} tok/s` : ""} · loading layers / reading prompt…`, {});
        } else if (ev.type === "token") {
          tokens++;
          first = first || performance.now();
          reply.content = ev.text;
          state.liveLoad = ev.load || {};
          const rate = tokens > 1 ? ((tokens - 1) / ((performance.now() - first) / 1000)).toFixed(1) : "…";
          showLive(`${tokens} tokens · ${rate} tok/s · first token ${((first - t0) / 1000).toFixed(1)}s`, ev.load);
          lastCtxUse = promptTokens + tokens;
          updateCtxMeter();
          updateLastMessage(true);
        } else if (ev.type === "replan") {
          showLive(`A device dropped out, continuing on ${ev.plan.length}…`, {});
        } else if (ev.type === "done") {
          reply.content = ev.text;
          reply.stats = ev.stats;
          state.liveLoad = ev.stats.load || {};
        } else if (ev.type === "error") {
          throw new Error(ev.error);
        }
      }
    }
  } catch (e) {
    if (e.name === "AbortError") reply.meta = `stopped · ${tokens} tokens`;
    else reply.meta = `error: ${e.message}`;
  } finally {
    state.generating = null; refreshChats();
    $("send").textContent = "Send";
    $("send").classList.remove("stop");
    store.set("ctxuse", lastCtxUse);
    updateLastMessage(false);
    setTimeout(() => { $("live").hidden = true; }, 4000);
    refreshStatus();
    setTimeout(loadChat, 300);
  }
}

$("composer").onsubmit = (e) => {
  e.preventDefault();
  if (state.generating) { state.generating.abort(); return; }
  const text = $("input").value.trim();
  if (!text) return;
  $("input").value = "";
  autosize();
  send(text);
};
$("input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); $("composer").requestSubmit(); }
});
function autosize() { const t = $("input"); t.style.height = "auto"; t.style.height = `${Math.min(t.scrollHeight, 220)}px`; }
$("input").addEventListener("input", autosize);
$("newchat").onclick = async () => {
  if (state.generating) state.generating.abort();
  chat = []; lastCtxUse = 0; store.set("ctxuse", 0); renderChat(); updateCtxMeter();
  try { await api("/api/chat/clear", { method: "POST" }); } catch { /* offline */ }
  refreshChats();
};

// Saved chats on the main device: search, sort, reopen, rename, delete
$("chatsort").value = store.get("chatsort", "updated");
$("chatsort").onchange = () => { store.set("chatsort", $("chatsort").value); refreshChats(); };
$("chatsearch").oninput = () => { clearTimeout(refreshChats.t); refreshChats.t = setTimeout(refreshChats, 250); };
const whenText = (t) => {
  const d = new Date(t * 1000), now = new Date();
  return d.toDateString() === now.toDateString() ? d.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" }) : d.toLocaleDateString();
};
async function refreshChats() {
  let r;
  try { r = await (await api(`/api/chats?q=${encodeURIComponent($("chatsearch").value)}`)).json(); } catch { return; }
  const by = $("chatsort").value, list = r.chats;
  const cmp = { updated: (a, b) => b.updated - a.updated, oldest: (a, b) => a.created - b.created,
    title: (a, b) => a.title.localeCompare(b.title), count: (a, b) => b.count - a.count }[by];
  list.sort(cmp);
  fill($("chatlist"), list.length ? list.map((c) => el("div", { class: "chatitem" + (c.id === r.open ? " open" : "") },
    el("button", { class: "chatopen", type: "button", title: c.title, onclick: () => openChat(c.id) },
      el("span", { class: "chattitle" }, c.title), el("span", { class: "muted small" }, `${whenText(c.updated)} · ${c.count} messages`)),
    el("button", { class: "ghost small", type: "button", title: "Rename", "aria-label": "Rename", onclick: () => renameChat(c) }, "✎"),
    el("button", { class: "ghost small danger", type: "button", title: "Delete", "aria-label": "Delete", onclick: () => deleteChat(c) }, "×")))
    : el("p", { class: "muted small" }, $("chatsearch").value ? "No chats match." : "Your chats show up here."));
}
async function openChat(id) {
  if (state.generating) return;
  const r = await api("/api/chats/open", { method: "POST", body: JSON.stringify({ id }) });
  if (!r.ok) { alert(await r.text()); return; }
  await loadChat(); refreshChats();
}
async function renameChat(c) {
  const title = prompt("Rename chat", c.title);
  if (title == null) return;
  await api("/api/chats/rename", { method: "POST", body: JSON.stringify({ id: c.id, title }) });
  refreshChats();
}
async function deleteChat(c) {
  if (!confirm(`Delete "${c.title}"?`)) return;
  await api(`/api/chats?id=${encodeURIComponent(c.id)}`, { method: "DELETE" });
  await loadChat(); refreshChats();
}

// ================================================================ model page

const SUGGESTED = [
  ["HuggingFaceTB/SmolLM2-360M-Instruct", "0.7 GB"], ["Qwen/Qwen2.5-0.5B-Instruct", "1 GB"],
  ["Qwen/Qwen2.5-1.5B-Instruct", "3.1 GB"], ["HuggingFaceTB/SmolLM2-1.7B-Instruct", "3.4 GB"],
  ["Qwen/Qwen2.5-3B-Instruct", "6.2 GB"], ["Qwen/Qwen2.5-7B-Instruct", "15 GB"],
];
$("chips").replaceChildren(...SUGGESTED.map(([repo, size]) => el("button", { type: "button", class: "chip",
  onclick: () => { $("hfrepo").value = repo; refreshEstimate(repo); } }, repo.split("/")[1], " ", el("small", {}, size))));

function estimateView(e, title) {
  if (e.problem) return el("div", { class: "est" }, el("p", { class: "warn" }, e.problem),
    e.disk_bytes ? el("p", { class: "muted small" }, `Download size ${gb(e.disk_bytes)}, needs about ${gb(e.memory_bytes_total)} of memory across devices.`) : null);
  return el("div", { class: "est" },
    el("div", { class: "row space" }, el("div", {}, el("div", { class: "muted small" }, title), el("div", { class: "big-num" }, `≈ ${e.tok_per_s} tok/s`)),
      el("div", { class: "kv" },
        stat(e.disk_bytes ? gb(e.disk_bytes) : "—", "download, main computer disk"),
        stat(gb(e.memory_bytes_total), "memory, all devices"),
        stat(`${Math.round(e.compute_ms)} / ${Math.round(e.network_ms)} ms`, "compute / network per token"))),
    el("table", {}, el("tr", {}, el("th", {}, "Device"), el("th", {}, "Layers"), el("th", {}, "Memory"), el("th", {}, "Receives"), el("th", {}, "Compute")),
      e.devices.map((d) => el("tr", {}, el("td", {}, d.node_id), el("td", {}, `${d.layers[0]}–${d.layers[1] - 1}`),
        el("td", {}, gb(d.memory_bytes)), el("td", {}, gb(d.transfer_bytes)), el("td", {}, `${d.compute_ms} ms`))),
      (e.standby || []).map((n) => el("tr", {}, el("td", {}, n), el("td", { colspan: 4, class: "muted" }, "standby: faster without it (turn on Always use to include)")))),
    el("p", { class: "muted small" }, "Estimates use each device's measured speed, usage limit and network delay. Devices keep their layers in memory (RAM), not on disk; only the main computer stores the download."));
}

async function refreshEstimate(repo) {
  repo = repo === undefined ? $("hfrepo").value.trim() : repo;
  const target = $("estimate");
  if (!repo) { target.replaceChildren(); return; }
  target.replaceChildren(el("p", { class: "muted est" }, `Estimating ${repo}…`));
  try {
    const r = await api(`/api/estimate?repo=${encodeURIComponent(repo)}&token=${encodeURIComponent($("hftoken").value.trim())}`);
    target.replaceChildren(estimateView(await r.json(), `On your current devices, ${repo.split("/").pop()} would run at`));
  } catch (e) { target.replaceChildren(el("p", { class: "warn est" }, e.message)); }
}
$("hfest").onclick = () => refreshEstimate();

async function refreshModelInfo() {
  let m, e = null;
  try {
    m = await (await api("/api/model")).json();
    if (m.model) e = await (await api("/api/estimate")).json();
  } catch { return; }
  const cur = $("current");
  if (!m.model) { cur.replaceChildren(el("p", { class: "muted" }, "No model loaded.")); }
  else {
    const c = m.model;
    cur.replaceChildren(
      el("div", { class: "row space" }, el("div", {}, el("div", { class: "big-num" }, c.name.split("/").pop()), el("div", { class: "muted small" }, c.name)),
        c.tokenizer === "bytes" ? el("span", { class: "badge relay" }, "test model, random weights") : el("span", { class: "badge on" }, m.chat_template ? "chat model" : "text model")),
      el("div", { class: "kv" }, stat(c.n_layer, "layers"), stat(fmtNum(Math.round((e && e.memory_bytes_total || 0) / 4)), "parameters"),
        stat(c.n_ctx.toLocaleString(), "max context"), stat(c.hf_type || c.arch || "gpt2", "architecture")),
      e ? estimateView(e, "Expected speed on your current devices") : null);
  }
  const d = m.download, busy = d && ["starting", "downloading", "loading"].includes(d.state);
  // say it next to the model too: a failed Load keeps the old model running
  if (d && d.state === "error") cur.prepend(el("p", { class: "bad" }, `Could not load ${d.repo.replace(/^gguf:/, "").split(/[\\/]/).pop()}: ${d.error}.${m.model ? " Still using the model below." : ""}`));
  else if (busy) cur.prepend(el("p", { class: "muted" }, `Switching to ${d.repo.replace(/^gguf:/, "").split(/[\\/]/).pop()}…`));
  $("dl").hidden = !d || d.state === "ready";
  $("hfload").disabled = !!busy;
  if (d) {
    const pct = d.total ? Math.round((100 * d.done) / d.total) : 0;
    $("dlbar").style.width = `${d.state === "loading" ? 100 : pct}%`;
    $("dltext").className = d.state === "error" ? "bad" : "muted";
    $("dltext").textContent = d.state === "error" ? `Could not load ${d.repo}: ${d.error}`
      : d.state === "downloading" ? `Downloading ${d.repo}: ${gb(d.done)} / ${gb(d.total)} (${pct}%)${d.file ? ` · ${d.file}` : ""}`
      : d.state === "loading" ? `Preparing ${d.repo}…` : `Starting ${d.repo}…`;
  }
  clearTimeout(refreshModelInfo.t);
  refreshModelInfo.t = setTimeout(refreshModelInfo, busy ? 1000 : 10000);
  if (busy !== refreshModelInfo.wasBusy) refreshDownloaded();
  refreshModelInfo.wasBusy = busy;
}

// models already downloaded on the main computer: load or uninstall them
async function refreshDownloaded() {
  let r;
  try { r = await (await api("/api/models")).json(); } catch { return; }
  const box = $("downloaded");
  if (!r.models.length) { fill(box, el("p", { class: "muted small" }, "Nothing downloaded yet. Models you load from Hugging Face are kept here.")); return; }
  const total = r.models.reduce((a, m) => a + m.bytes, 0);
  fill(box, el("table", { class: "models" }, r.models.map((m) => el("tr", {},
    el("td", {}, el("b", {}, m.name || m.repo.split("/").pop()),
      el("div", { class: "muted small" }, m.where ? `GGUF · ${m.where}` : m.format === "gguf" ? `${m.repo} · GGUF` : m.repo)),
    el("td", { class: "muted" }, gb(m.bytes)),
    el("td", { class: "actions" },
      m.loaded ? el("span", { class: "badge on" }, "loaded")
        : m.complete ? el("button", { class: "small", type: "button", onclick: () => loadRepo(m.repo) }, "Load")
        : el("span", { class: "badge relay", title: "the download did not finish; Load finishes it" }, "partial"),
      !m.loaded && m.removable !== false ? el("button", { class: "small ghost danger", type: "button", onclick: () => uninstall(m.name || m.repo, m.repo, m.bytes) }, "Uninstall") : null)))),
  el("p", { class: "muted small" }, `${gb(total)} used on the main computer's disk.`));
}

async function loadRepo(repo) {
  const r = await api("/api/model", { method: "POST", body: JSON.stringify({ repo, token: $("hftoken").value.trim() }) });
  if (!r.ok) { $("dl").hidden = false; $("dltext").className = "bad"; $("dltext").textContent = await r.text(); return; }
  refreshModelInfo();
}

async function uninstall(label, repo, bytes) {
  if (!confirm(`Delete ${label} from the main computer? This frees ${gb(bytes)}. You can download it again later.`)) return;
  const r = await api(`/api/models?repo=${encodeURIComponent(repo)}`, { method: "DELETE" });
  if (!r.ok) alert(await r.text());
  refreshDownloaded();
}

const params_ = (n) => (n >= 1e9 ? `${(n / 1e9).toFixed(n >= 1e10 ? 0 : 1)}B` : `${Math.round(n / 1e6)}M`);
function drawCapacity(c) {
  if (!c || !c.devices) { fill($("capacity"), el("p", { class: "muted small" }, "Connect a device to see what your mesh can run.")); return; }
  fill($("capacity"), el("div", { class: "kv" },
    stat(params_(c.max_params), "biggest model that fits"),
    stat(params_(c.params_at_10), "for ~10 tok/s"),
    stat(params_(c.params_at_5), "for ~5 tok/s"),
    stat(params_(c.params_at_1), "for ~1 tok/s")),
  el("p", { class: "muted small" }, `Based on ${c.devices} device(s) with ${c.gflops} GFLOP/s in total at their usage limits and the memory they offer. `
    + "Generating a token reads every weight once (about 2 operations per parameter), so speed falls as models grow; network delay lowers it a bit more. "
    + "\"B\" = billion parameters, e.g. Qwen2.5-1.5B is 1.5B."));
}

$("ggform").onsubmit = (e) => {
  e.preventDefault();
  const p = $("ggpath").value.trim().replace(/^"|"$/g, "");
  if (p) loadRepo(p.startsWith("gguf:") ? p : "gguf:" + p);
};

// Upload in 32 MB pieces: tunnels cap each request (Cloudflare at 100 MB), and a dropped
// connection resumes from what the main computer already has.
$("ggup").onclick = async () => {
  const f = $("ggfile").files[0];
  if (!f) { $("ggfile").click(); return; }
  const PIECE = 32 << 20, show = (t, cls = "muted small") => { $("up").hidden = false; $("uptext").className = cls; $("uptext").textContent = t; };
  let off = 0, tries = 0;
  $("ggup").disabled = true;
  try {
    while (off < f.size || off === 0) {
      const end = Math.min(f.size, off + PIECE), last = end >= f.size;
      const q = `name=${encodeURIComponent(f.name)}&offset=${off}${last ? "&done=1" : ""}`;
      let r;
      try { r = await api(`/api/models/upload?${q}`, { method: "POST", body: f.slice(off, end), headers: { "Content-Type": "application/octet-stream" } }); }
      catch (e) { if (++tries > 5) throw e; await new Promise((ok) => setTimeout(ok, 2000)); continue; }
      const j = await r.json();
      if (r.status === 409) { off = j.have; continue; }
      if (!r.ok) throw new Error(j.error || r.statusText);
      off = j.have; tries = 0;
      $("upbar").style.width = `${(100 * off / f.size).toFixed(1)}%`;
      show(`Uploaded ${gb(off)} of ${gb(f.size)}`);
      if (last) { show(`${f.name} is on the main computer. Loading it…`); refreshDownloaded(); loadRepo(j.repo); break; }
    }
  } catch (e) { show(`Upload stopped: ${e.message}. Press Upload again to continue.`, "bad small"); }
  $("ggup").disabled = false;
};

$("hfform").onsubmit = (e) => {
  e.preventDefault();
  const repo = $("hfrepo").value.trim();
  if (repo) loadRepo(repo);
};

// ================================================================ connect page

// Two links: the full one (chat, models, every device's settings) and a contribute-only one
// that can only lend compute. Both work at the same address.
let joinView = store.get("joinview", "full");
function setJoin(origin) {
  if (setJoin.last === origin + joinView + state.ckey) return;
  setJoin.last = origin + joinView + state.ckey;
  const full = `${origin}/?k=${encodeURIComponent(key)}`;
  const contrib = state.ckey ? `${origin}/contribute?c=${state.ckey}` : "";
  const link = joinView === "contribute" && contrib ? contrib : full;
  for (const b of document.querySelectorAll("#linkkind button")) b.classList.toggle("active", b.dataset.kind === joinView);
  $("linkhelp").textContent = joinView === "contribute"
    ? "Contribute only: the device lends its power and sees its own stats and the answer it's helping with. No chat, models or settings of other devices. Good for friends or a school Chromebook."
    : "Full access: chat, load models and change any device's limits. Only share it with your own devices.";
  $("joinlink").value = link;
  $("joincmd").textContent = `omnicompute join "${link}"`;
  $("qr").src = `/api/qr.svg?url=${encodeURIComponent(link)}&k=${encodeURIComponent(key)}`;
  if (state.ckey) {
    $("ps1").textContent = `irm "${origin}/install.ps1?c=${state.ckey}" | iex`;
    $("sh1").textContent = `curl -fsSL "${origin}/install.sh?c=${state.ckey}" | sh`;
  }
  const port = new URL(origin).port || (origin.startsWith("https") ? "443" : "80");
  $("port").textContent = $("port2").textContent = port;
  for (const e of document.querySelectorAll(".portv")) e.textContent = port;
}
for (const b of document.querySelectorAll("#linkkind button")) b.onclick = () => { joinView = b.dataset.kind; store.set("joinview", joinView); setJoin.last = null; refreshStatus(); };
document.addEventListener("click", async (e) => {
  const b = e.target.closest("[data-copyfrom]");
  if (!b) return;
  const src = $(b.dataset.copyfrom), text = src.value !== undefined ? src.value : src.textContent;
  try { await navigator.clipboard.writeText(text); b.textContent = "Copied"; } catch { b.textContent = "Select & copy"; }
  setTimeout(() => { b.textContent = "Copy"; }, 1500);
});

// ================================================================ start

// A key that doesn't match (an old mesh's key saved in this browser, or a mistyped one):
// forget it and ask again, instead of showing "wrong mesh key" forever.
function askForKey(wrong) {
  clearInterval(state.poll);
  store.set("key", null);
  $("pill").textContent = wrong ? "wrong mesh key" : "no mesh key";
  $("pill").className = "pill bad";
  $("keyerr").hidden = !wrong;
  for (const p of document.querySelectorAll(".page")) p.classList.remove("active");
  $("keybox").hidden = false;
  $("keybox").classList.add("active");
}

$("keyform").onsubmit = (e) => {
  e.preventDefault();
  let v = $("keyinput").value.trim().replace(/^["'<]+|["'>.,;)]+$/g, "");
  try { // a whole link pasted: take its key
    const u = new URL(v.includes("://") ? v : "http://" + v);
    if (u.searchParams.get("c") && !u.searchParams.get("k")) { location.href = `/contribute?c=${encodeURIComponent(u.searchParams.get("c"))}`; return; }
    if (u.searchParams.get("k")) v = u.searchParams.get("k");
  } catch { /* just a key */ }
  if (/^c[0-9a-f]{24}$/.test(v)) { location.href = `/contribute?c=${v}`; return; } // contribute-only key
  key = v;
  init();
};

async function init() {
  if (!key) { askForKey(false); return; }
  $("keybox").hidden = true;
  $("keybox").classList.remove("active");
  store.set("key", key);
  state.ident = await deviceIdentity();
  setJoin(location.origin);
  drawMyLimits();
  renderChat();
  updateCtxMeter();
  showTab(store.get("tab", "chat"));
  if (!(await refreshStatus())) return;
  clearInterval(state.poll);
  state.poll = setInterval(refreshStatus, 2500);
  refreshModelInfo();
  refreshDownloaded();
  loadChat(); refreshChats();
  if (store.get("mode", "controller") === "worker") setMode("worker");
}

init();
