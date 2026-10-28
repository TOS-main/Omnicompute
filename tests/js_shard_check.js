// Usage: node js_shard_check.js <request.bin>  -> writes logits tensor pack to stdout
const fs = require("fs");
const { Shard, decodeTensors, encodeTensors, unpackMsg } = require("../edgemesh/web/model.js");
const [hdr, payload] = unpackMsg(new Uint8Array(fs.readFileSync(process.argv[2])));
const w = decodeTensors(payload);
const ids = w["__ids__"]; delete w["__ids__"];
const shards = hdr.splits.map(([s, e]) => new Shard(hdr.cfg, w, s, e));
const run = (x, pos) => { for (const sh of shards) x = sh.forward("j", x, pos); return x; };
const outs = {};
outs.prompt = run(ids, 0);
outs.step = run({ data: new Int32Array([hdr.next]), shape: [1] }, ids.data.length);
process.stdout.write(Buffer.from(encodeTensors(outs)));
