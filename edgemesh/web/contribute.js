// Contribute-only page: this browser joins the mesh as a worker with the contribute key.
// It has no chat, models or other devices' settings; the hub refuses those for this key.

"use strict";

const params = new URLSearchParams(location.search);
const ckey = params.get("c") || store.get("ckey", "");
if (params.get("c")) history.replaceState(null, "", location.pathname); // keep the key out of screenshots
store.set("ckey", ckey);

let ident = null, pw = null;
let myLimits = store.get("limits", { ...DEFAULT_LIMITS });
const done = store.get("helped", { answers: 0, tokens: 0, compute_s: 0 });

function drawStats() {
  fill($("mystats"), stat(done.answers, "answers helped"), stat(done.tokens, "tokens"),
    stat(`${done.compute_s.toFixed(1)} s`, "compute time"));
}

function drawLimits() {
  fill($("mylimits"), limitEditor(myLimits, (l) => { myLimits = l; store.set("limits", l); if (pw) pw.limits(l); }, { label: "When working" }));
}

function setRunning(on) {
  $("toggle").textContent = on ? "Stop contributing" : "Start contributing";
  $("toggle").classList.toggle("stop", on);
  $("pill").textContent = on ? "contributing" : "not contributing";
  $("pill").className = on ? "pill ok" : "pill";
}

function start() {
  pw = new PageWorker(ckey, ident, myLimits, {
    onState: (d) => { $("wstatus").className = d.state === "ready" ? "ok" : d.state === "error" ? "bad" : "muted"; $("wstatus").textContent = workerStatusText(d); },
    onLimits: (l) => { myLimits = l; store.set("limits", l); drawLimits(); },
    onCredit: (c) => {
      done.answers++; done.tokens += c.tokens; done.compute_s += c.compute_s;
      store.set("helped", done); drawStats();
      $("lastcredit").textContent = `Last answer: you ran layers ${c.layers[0]}–${c.layers[1] - 1} and did ${c.share}% of the work (${c.ms_per_token} ms per token).`;
    },
    onRemoved: () => { pw = null; keepAwake.stop(); setRunning(false); $("toggle").disabled = true; },
    onLive: (text) => { $("livecard").hidden = false; $("livetext").textContent = text; },
  });
  if ($("bg").checked) keepAwake.start();
  store.set("running", true);
  setRunning(true);
}

function stop() {
  if (pw) pw.stop();
  pw = null;
  keepAwake.stop();
  store.set("running", false);
  setRunning(false);
  $("wstatus").textContent = "";
}

$("toggle").onclick = () => (pw ? stop() : start());
$("bg").checked = store.get("background", true);
$("bg").onchange = () => { store.set("background", $("bg").checked); if (pw && $("bg").checked) keepAwake.start(); else keepAwake.stop(); };

(async () => {
  if (!ckey) { $("wstatus").textContent = "This link is missing its key. Ask for the contribute link again."; $("toggle").disabled = true; return; }
  ident = await deviceIdentity();
  $("ps").textContent = `irm "${location.origin}/install.ps1?c=${ckey}" | iex`;
  $("sh").textContent = `curl -fsSL "${location.origin}/install.sh?c=${ckey}" | sh`;
  drawStats();
  drawLimits();
  setRunning(false);
  if (store.get("running", false)) start(); // resume after a reload (background audio needs a tap first)
})();
