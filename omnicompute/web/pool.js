// One core of a browser worker's compute pool. Holds a column slice of each weight matrix
// (or a row slice of a tied embedding) and multiplies activations by it on request.

"use strict";
importScripts("/static/model.js");

const W = {};

onmessage = (ev) => {
  const m = ev.data;
  if (m.op === "put") {
    W[m.id] = m;
    postMessage({ seq: m.seq });
  } else if (m.op === "mm") {
    const w = W[m.id];
    let out;
    if (w.layout === "rows") { // out[r] = x . E[r]  (tied output head)
      out = new Float32Array(w.R);
      for (let r = 0; r < w.R; r++) {
        let s = 0;
        const o = r * w.K;
        for (let k = 0; k < w.K; k++) s += m.x[k] * w.data[o + k];
        out[r] = s;
      }
    } else {
      out = matmul(m.x, m.T, w.K, w.data, w.N, null);
    }
    postMessage({ seq: m.seq, out }, [out.buffer]);
  } else if (m.op === "clear") {
    for (const k of Object.keys(W)) delete W[k];
    postMessage({ seq: m.seq });
  } else if (m.op === "bench") {
    postMessage({ seq: m.seq, gflops: benchmark() });
  }
};
