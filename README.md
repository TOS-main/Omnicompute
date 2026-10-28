# EdgeMesh

Run one AI model across several devices (phones, laptops, desktops), even when they sit on
different Wi-Fi networks or on cellular data. Each device holds a slice of the model's layers,
and activations hop from device to device for every token.

```
            ┌──────────── hub (any reachable machine: laptop, home PC, $5 cloud VM) ────────────┐
            │  web page · WebSocket control · UDP reflector · relay · orchestrator + model file │
            └──────────────▲──────────────────────▲─────────────────────────▲──────────────────┘
                           │ control              │ control                 │ WebSocket (relay)
      layers 0-5   ┌───────┴──────┐  direct UDP   ┌┴─────────────┐  relay    ┌┴──────────────────┐
  prompt ────────► │ Windows PC   │ ════════════► │ MacBook      │ ────────► │ Android phone     │ ──► logits
                   │ (home Wi-Fi) │  hole-punched │ (office Wi-Fi)│ via hub  │ (cellular, CGNAT) │    to hub
                   └──────────────┘               └──────────────┘           └───────────────────┘
```

## Quick start

**1. Start a hub** on one computer (it also contributes its own compute):

```
pip install .            # or double-click the edgemesh app from the build step below
edgemesh host            # serves a small random demo model
```

It prints a join link like `http://192.168.1.20:8000/?k=Xy3...`.

**2. Add devices**

| Device | How |
|---|---|
| Android phone, iPhone, iPad, any tablet | Open the join link in the browser, tap **Contribute compute**. No app or Termux needed. Keep the page open with the screen on. |
| Windows, macOS, Linux | Double-click the app, choose **Join**, paste the link. Or run `edgemesh join "<link>"`. |

**3. Run a prompt** from any device that can open the link (the same page has a prompt box), or:

```
edgemesh run "<link>" "Once upon a time" --max-new 50
```

**Devices on different networks or on cellular:** the hub has to be reachable from all of them.
Either run the hub on a small cloud VM (open TCP+UDP port 8000), or forward TCP+UDP 8000 on the
router of the machine that runs it, then share `http://<public-ip>:8000/?k=<key>`.

**Real model (GPT-2):** `edgemesh convert-gpt2 models/gpt2` downloads GPT-2 from Hugging Face and
converts it (no torch needed), then `edgemesh host --model models/gpt2/gpt2.emm`. Also works with
`--variant gpt2-medium`, `gpt2-large`, `gpt2-xl`, `distilgpt2`.

## How it works

**Pipeline split by layers.** The hub measures each device (a quick matmul benchmark, plus how
much RAM it offers) and gives each a contiguous block of layers in proportion to its speed, never
more than fits in its memory. The first device also holds the token embeddings, the last one the
output head. Weights for each slice are sent over the mesh, so workers don't need the model file.

**Device-to-device chain.** For each token the hub sends one message to the first device, which
runs its layers and passes the hidden state straight to the next device, and so on; the last one
returns logits to the hub. Each device keeps a KV cache per generation, so after the prompt only
one small vector per token crosses the network.

**Getting through NAT.** Most phones and home networks sit behind NAT, and cellular often behind
carrier-grade NAT, so devices can't simply connect to each other:

1. Every native device logs in to the hub over a WebSocket and sends a UDP probe to the hub's
   port; the hub replies with the public `ip:port` it saw (the device's NAT mapping, like STUN).
2. When device A needs to talk to device B, the hub sends each one the other's public mapping and
   LAN addresses, and both fire encrypted UDP "punch" packets at the same time. On the same Wi-Fi
   the LAN address wins; across home routers and most cone NATs the public mappings open up.
3. If no punch gets through in 3 seconds (symmetric NAT, strict CGNAT, UDP blocked), traffic goes
   through the hub relay over the WebSocket. Messages never wait for punching: they use the relay
   until the direct path is up, and fall back to it if the direct path stops answering.

Over UDP, messages are chunked and sent with a selective-ACK sliding window. NAT mappings are kept
alive with keepalives, and a peer's changed address (e.g. Wi-Fi to cellular) is picked up from its
packets.

**Failures.** If a device disconnects mid-generation, the hub re-plans over the remaining devices
and replays the sequence so far, so the generation continues.

**Security.** Everything derives from the mesh key in the join link. Native devices log in with an
HMAC challenge-response, and all their UDP packets and relayed messages are sealed with AES-256-GCM.
Browser devices present the key and talk to the hub over the WebSocket; put the hub behind HTTPS
(any reverse proxy or tunnel) when you use it across the internet. The hub knows the key, so run
it yourself; treat the link like a password.

## Platforms

| | Contribute compute | Run prompts | Notes |
|---|---|---|---|
| Windows / macOS / Linux | native app or `pip install` | yes | direct UDP + relay; numpy (BLAS) speed |
| Android | browser page | yes | relay only, plain-JS math, keep screen on |
| iPhone / iPad | browser page (Safari) | yes | same as Android; iOS pauses background tabs |
| Termux, Pythonista, a-Shell | native client also works | yes | optional |

## Tests

```
pip install -e ".[test]"
pytest                       # model split == single device, relay + direct paths, device dropout,
                             # browser workers in Chromium, JS math == numpy, GPT-2 conversion
sudo scripts/nat_sim.sh      # three simulated networks: two home NATs + a symmetric "cellular" NAT
```

`nat_sim.sh` builds network namespaces with Linux NAT routers and checks that the two home
devices hole-punch a direct path while the symmetric-NAT device falls back to the relay, and that
generation completes across all three.

Build the double-click app for the current OS with `pip install pyinstaller . && python
scripts/build_app.py`; CI builds Windows, macOS and Linux versions on every push.

## Limits of this MVP and next steps

- Browser workers compute in plain JavaScript and always use the relay. Next: WebGPU kernels and
  WebRTC data channels so phones can connect peer to peer too.
- fp32 weights only; GPT-2 family only. Next: int8/fp16 weights and Llama-style models (RoPE,
  RMSNorm, GQA), which also shrinks what phones must download.
- One generation at a time, one hub. Next: request batching, several models, hub failover.
- IPv4 only for hole punching; add IPv6 candidates (many cellular networks hand out public IPv6,
  which needs no punching at all).
- Native mobile apps (Android/iOS) would allow background running and faster math.
