// Shared by the full controller page (app.js) and the contribute-only page (contribute.js):
// DOM helpers, device identity, the usage-limits editor and the in-page worker.

"use strict";

const $ = (id) => document.getElementById(id);
const el = (tag, attrs = {}, ...kids) => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (v !== undefined && v !== null && v !== false) e.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat()) if (kid !== null && kid !== undefined && kid !== false) e.append(kid.nodeType ? kid : String(kid));
  return e;
};
const fill = (node, ...kids) => node.replaceChildren(...kids.flat().filter((k) => k !== null && k !== undefined && k !== false));
const store = {
  get(k, d) { try { const v = localStorage.getItem("edgemesh." + k); return v === null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem("edgemesh." + k, JSON.stringify(v)); } catch { /* private mode */ } },
};
const gb = (b) => (b >= 2 ** 30 ? `${(b / 2 ** 30).toFixed(1)} GB` : `${Math.max(1, Math.round(b / 2 ** 20))} MB`);
const fmtNum = (n) => (n >= 1e9 ? `${(n / 1e9).toFixed(1)}B` : n >= 1e6 ? `${Math.round(n / 1e6)}M` : String(n));
const stat = (v, label) => el("div", { class: "stat" }, el("b", {}, v), el("span", {}, label));

// ================================================================ device identity

async function deviceIdentity() {
  let name = store.get("name", null);
  const ua = navigator.userAgent;
  let os = /CrOS/.test(ua) ? "ChromeOS" : /iPhone|iPad/.test(ua) ? "iOS" : /Android/.test(ua) ? "Android"
    : /Mac/.test(ua) ? "macOS" : /Windows/.test(ua) ? "Windows" : "Linux";
  let model = "", version = "";
  try { // Chrome hides the real Android version and phone model from the user agent string
    if (navigator.userAgentData && navigator.userAgentData.getHighEntropyValues) {
      const h = await navigator.userAgentData.getHighEntropyValues(["model", "platformVersion", "platform"]);
      model = h.model || "";
      version = (h.platformVersion || "").split(".")[0];
      if (h.platform === "Chrome OS" || h.platform === "ChromeOS") os = "ChromeOS";
    }
  } catch { /* not available */ }
  if (!version) {
    const m = ua.match(/(?:Android|iPhone OS|CPU OS|Mac OS X|Windows NT) ([\d_.]+)/);
    if (m) version = m[1].replace(/_/g, ".").split(".")[0];
  }
  if (os === "Windows" && version === "10") version = ""; // Windows 10 and 11 both report 10
  if (!name) {
    name = `${(model || os).toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "") || "browser"}-${Math.random().toString(16).slice(2, 6)}`;
    store.set("name", name);
  }
  const browser = /Edg\//.test(ua) ? "Edge" : /SamsungBrowser/.test(ua) ? "Samsung Internet" : /Firefox/.test(ua) ? "Firefox"
    : /CriOS|Chrome/.test(ua) ? "Chrome" : /Safari/.test(ua) ? "Safari" : "browser";
  return { name, platform: [model, `${os}${version ? " " + version : ""}`, browser].filter(Boolean).join(" · ") };
}

// ================================================================ usage limits editor (used for every device)

const DEFAULT_LIMITS = { cpu: 80, schedule: [], always: false, gpu: true, cores: null, battery: null, charging: null };

function limitEditor(limits, onChange, opts = {}) {
  const cur = { ...DEFAULT_LIMITS, ...JSON.parse(JSON.stringify(limits || {})) };
  let timer = null;
  const emit = () => { clearTimeout(timer); timer = setTimeout(() => onChange(JSON.parse(JSON.stringify(cur))), 350); };
  const show = (v) => (v ? `${v}%` : "paused");
  const pct = el("span", { class: "pct" }, show(cur.cpu));
  const slider = el("input", { type: "range", min: 0, max: 100, step: 5, value: cur.cpu, "aria-label": "usage limit",
    oninput: (e) => { cur.cpu = +e.target.value; pct.textContent = show(cur.cpu); emit(); } });
  const set = (v) => { cur.cpu = v; slider.value = v; pct.textContent = show(v); emit(); };
  const check = (on, label, change, title) => {
    const box = el("input", { type: "checkbox", onchange: (e) => change(e.target.checked) });
    box.checked = !!on;
    return el("label", { class: "switch small", title }, box, label);
  };
  const num = (value, min, max, step, change, label) => el("input", { type: "number", min, max, step, value: value ?? "",
    placeholder: "all", "aria-label": label, onchange: (e) => change(e.target.value === "" ? null : Math.max(min, Math.min(max, +e.target.value))) });

  const rows = el("div");
  const drawRows = () => fill(rows, cur.schedule.map((w, i) => el("div", { class: "sched" },
    "From", el("input", { type: "time", value: w.from, onchange: (e) => { w.from = e.target.value; emit(); } }),
    "to", el("input", { type: "time", value: w.to, onchange: (e) => { w.to = e.target.value; emit(); } }),
    "use", num(w.cpu, 0, 100, 5, (v) => { w.cpu = v ?? 0; emit(); }, "percent"), "%",
    el("button", { class: "ghost small", type: "button", onclick: () => { cur.schedule.splice(i, 1); drawRows(); emit(); } }, "Remove"))));
  drawRows();

  const powerRule = (state, label, dflt) => {
    const body = el("span", { class: "sched" });
    const draw = () => fill(body, cur[state] ? ["use", num(cur[state].cpu, 0, 100, 5, (v) => { cur[state].cpu = v ?? 0; emit(); }, "percent"), "% and",
      num(cur[state].cores, 1, 64, 1, (v) => { cur[state].cores = v; emit(); }, "cores"), "cores"] : null);
    draw();
    return el("div", { class: "row wrap" }, check(cur[state], label, (on) => { cur[state] = on ? { ...dflt } : null; draw(); emit(); }), body);
  };

  return el("div", { class: "limitbox" },
    el("div", { class: "row" }, el("span", { class: "small muted" }, opts.label || "Usage limit"), slider, pct),
    el("div", { class: "row wrap" },
      el("button", { class: "small", type: "button", onclick: () => set(100) }, "100%"),
      el("button", { class: "small", type: "button", onclick: () => set(80) }, "80%"),
      el("button", { class: "small", type: "button", onclick: () => set(50) }, "50%"),
      el("button", { class: "small", type: "button", onclick: () => set(0) }, "Pause"),
      check(cur.always, "Always use", (on) => { cur.always = on; emit(); }, "keep it in the split even when the mesh would be faster without it"),
      check(cur.gpu, "Use GPU", (on) => { cur.gpu = on; emit(); }, "use the graphics chip when it is faster than the CPU")),
    el("details", { class: "more", open: opts.open || cur.schedule.length || cur.battery || cur.charging || cur.cores ? true : null },
      el("summary", { class: "small" }, "Cores, battery and time of day"),
      el("div", { class: "row wrap small" }, "CPU cores", num(cur.cores, 1, 64, 1, (v) => { cur.cores = v; emit(); }, "cores"),
        el("span", { class: "muted" }, "(blank = the device's default)")),
      powerRule("battery", "On battery:", { cpu: 30, cores: 2 }),
      powerRule("charging", "While charging:", { cpu: 100, cores: null }),
      rows,
      el("button", { class: "ghost small", type: "button", onclick: () => { cur.schedule.push({ from: "22:00", to: "07:00", cpu: 100 }); drawRows(); emit(); } }, "+ Time window"),
      el("p", { class: "muted small" }, "Time windows use the device's own clock and override the limit while they're active. "
        + "\"On battery\" caps everything while unplugged; \"While charging\" replaces the limit when plugged in. "
        + "Phones and laptops in a browser report charging on Chrome and Edge.")));
}

// ================================================================ this browser as a worker

// Battery state for the worker (the Battery API only exists on the page, not inside workers).
async function watchPower(onChange) {
  try {
    if (!navigator.getBattery) return;
    const b = await navigator.getBattery();
    const report = () => onChange(b.charging ? "charging" : "battery");
    b.addEventListener("chargingchange", report);
    report();
  } catch { /* not supported */ }
}

class PageWorker {
  // key: mesh or contribute key; onState({state, note, stats, gpu}); onLimits(limits); onCredit(credit)
  constructor(key, ident, limits, handlers) {
    this.w = new Worker("/static/worker.js");
    const mem = Math.min(2048, Math.round((navigator.deviceMemory || 2) * 1024 * 0.3));
    this.w.postMessage({
      cmd: "start", key, name: ident.name, limits, forceGpu: store.get("forcegpu", false), // forcegpu: for testing
      wsUrl: `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`,
      info: { role: "worker", kind: "browser", platform: ident.platform, mem_mb: mem },
    });
    this.w.onmessage = (ev) => {
      const d = ev.data;
      if (d.state === "limits") handlers.onLimits && handlers.onLimits(d.limits);
      else if (d.state === "credit") handlers.onCredit && handlers.onCredit(d.credit);
      else if (d.state === "live") handlers.onLive && handlers.onLive(d.text, d.done);
      else if (d.state === "removed") {
        for (const k of ["ckey", "running", "mode"]) store.set(k, null);
        handlers.onState && handlers.onState({ state: "error", note: "removed from the mesh by the main computer" });
        handlers.onRemoved && handlers.onRemoved();
      }
      else handlers.onState && handlers.onState(d);
    };
    watchPower((p) => this.w && this.w.postMessage({ cmd: "power", power: p }));
    try { navigator.wakeLock && navigator.wakeLock.request("screen").catch(() => {}); } catch { /* needs https */ }
  }

  limits(l) { this.w.postMessage({ cmd: "limits", limits: l }); }

  stop() {
    const w = this.w;
    this.w = null;
    w.postMessage({ cmd: "stop" });
    setTimeout(() => w.terminate(), 500);
  }
}

function workerStatusText(d) {
  const { state: st, note, stats } = d;
  const label = { connecting: "Connecting…", ready: "Contributing", stopped: "Stopped", error: "Error" }[st] || st;
  const held = stats && stats.layers ? ` · holding ${stats.layers[1] - stats.layers[0] === 1 ? `layer ${stats.layers[0]}` : `layers ${stats.layers[0]}–${stats.layers[1] - 1}`}` : "";
  const engine = stats && stats.gpu ? ` · GPU (${stats.gpu})` : stats && stats.cores > 1 ? ` · ${stats.activeCores || stats.cores} cores` : "";
  const power = stats && stats.power ? ` · ${stats.power === "charging" ? "charging" : "on battery"}` : "";
  return `${label}${note && !held ? ": " + note : ""}${held}${engine}${power}${stats && stats.tokens ? ` · ${stats.tokens} steps` : ""}`;
}

// Phones pause background tabs after a while. A tab that plays sound is left running, so
// "keep working in the background" plays a tone too quiet and too low to hear (20 Hz at
// 0.1% volume). It has to start from a tap. The phone may still stop it to save battery.
const keepAwake = {
  ctx: null,
  start() {
    try {
      if (this.ctx) { this.ctx.resume(); return true; }
      const AC = window.AudioContext || window.webkitAudioContext;
      if (!AC) return false;
      this.ctx = new AC();
      const osc = this.ctx.createOscillator(), gain = this.ctx.createGain();
      osc.frequency.value = 20;
      gain.gain.value = 0.001;
      osc.connect(gain).connect(this.ctx.destination);
      osc.start();
      if (navigator.mediaSession) navigator.mediaSession.metadata = new MediaMetadata({ title: "EdgeMesh is contributing", artist: "EdgeMesh" });
      return true;
    } catch { return false; }
  },
  stop() { if (this.ctx) { this.ctx.close(); this.ctx = null; } },
};
