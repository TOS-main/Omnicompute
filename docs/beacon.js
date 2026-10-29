// Finds where a mesh's host is now (see omnicompute/beacon.py): the host posts its address,
// sealed with a key derived from the contribute key, to a public ntfy topic. Used by the web
// page to follow a host that moved, and by the permanent link page (docs/go.html).

"use strict";

const BEACON_SERVER = "https://ntfy.sh";
const _enc = new TextEncoder();

async function _derive(secret, info, bytes = 32) {
  const ikm = await crypto.subtle.importKey("raw", _enc.encode(secret), "HKDF", false, ["deriveBits"]);
  const bits = await crypto.subtle.deriveBits({ name: "HKDF", hash: "SHA-256", salt: _enc.encode("edgemesh-v1"), info: _enc.encode(info) }, ikm, bytes * 8);
  return new Uint8Array(bits);
}
const _hex = (b) => [...b].map((x) => x.toString(16).padStart(2, "0")).join("");

async function contributeKeyOf(key) {
  if (/^c[0-9a-f]{24}$/.test(key)) return key;
  return "c" + _hex((await _derive(key, "contribute")).slice(0, 12));
}

async function beaconTopic(key) {
  return "edgemesh-" + _hex((await _derive(await contributeKeyOf(key), "beacon-topic")).slice(0, 12));
}

async function _unseal(ck, text) {
  try {
    const b64 = text.replace(/-/g, "+").replace(/_/g, "/") + "=".repeat((4 - (text.length % 4)) % 4);
    const raw = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
    const k = await crypto.subtle.importKey("raw", await _derive(ck, "beacon-key"), "AES-GCM", false, ["decrypt"]);
    const plain = await crypto.subtle.decrypt({ name: "AES-GCM", iv: raw.slice(0, 12), additionalData: _enc.encode("edgemesh-beacon") }, k, raw.slice(12));
    return JSON.parse(new TextDecoder().decode(plain));
  } catch { return null; }
}

// -> {url, lan, t} or null
async function findHost(key, server = BEACON_SERVER) {
  const ck = await contributeKeyOf(key);
  const r = await fetch(`${server}/${await beaconTopic(ck)}/json?poll=1&since=12h`, { cache: "no-store" });
  if (!r.ok) return null;
  let best = null;
  for (const line of (await r.text()).split("\n")) {
    let ev;
    try { ev = JSON.parse(line); } catch { continue; }
    if (ev.event && ev.event !== "message") continue;
    const info = await _unseal(ck, ev.message || "");
    if (info && info.url && (!best || (info.t || 0) >= (best.t || 0))) best = info;
  }
  return best;
}

// The origin the host answers on now, if it moved away from this page's origin; else null.
// Tries the Wi-Fi address first (faster when this device is on the same network).
async function hostMovedTo(key, server = BEACON_SERVER) {
  const info = await findHost(key, server);
  for (const origin of [info && info.lan, info && info.url]) {
    if (!origin || origin === location.origin) continue;
    try {
      const r = await fetch(origin + "/api/hello", { cache: "no-store", signal: AbortSignal.timeout(4000) });
      if (r.ok) return origin;
    } catch {}
  }
  return null;
}

// Call while the host is unreachable; after 15 s it looks the host up every 15 s and moves there.
// key and path may be functions, read when the lookup happens.
function hostFollower(key, path) {
  const val = (v) => (typeof v === "function" ? v() : v);
  let since = 0, last = 0;
  return {
    lost() {
      const now = Date.now();
      if (!since) since = now;
      if (now - since < 15000 || now - last < 15000) return;
      last = now;
      if (!val(key)) return;
      hostMovedTo(val(key)).then((o) => { if (o) location.href = o + val(path); }).catch(() => {});
    },
    ok() { since = 0; },
  };
}

if (typeof module !== "undefined") module.exports = { findHost, hostMovedTo, hostFollower, beaconTopic, contributeKeyOf };
