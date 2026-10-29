// Usage: node js_forward_ex.js <request.bin>: runs one browser shard on given step inputs
// ("s0.x", "s0.ple", ...) and writes its outputs ("s0.x", ...) as a tensor pack to stdout.
const fs = require("fs");
const { Shard, decodeTensors, encodeTensors, unpackMsg } = require("../omnicompute/web/model.js");
const [hdr, payload] = unpackMsg(new Uint8Array(fs.readFileSync(process.argv[2])));
const all = decodeTensors(payload), w = {}, steps = hdr.pos.map(() => ({}));
for (const [name, t] of Object.entries(all)) {
  const m = /^s(\d+)\.(.*)$/.exec(name);
  if (m) steps[Number(m[1])][m[2]] = t; else w[name] = t;
}
const shard = new Shard(hdr.cfg, w, hdr.start, hdr.end);
(async () => {
  const outs = {};
  for (let i = 0; i < steps.length; i++) {
    for (const [name, t] of Object.entries(await shard.forwardEx("j", steps[i], hdr.pos[i]))) outs[`s${i}.${name}`] = t;
  }
  process.stdout.write(Buffer.from(encodeTensors(outs)));
})().catch((e) => { console.error(e); process.exit(1); });
