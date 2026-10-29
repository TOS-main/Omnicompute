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

## Quick start (no setup, any network)

The project page, https://tos-main.github.io/edgemesh/, has every command below with copy buttons.

**Start your own mesh** on the device that stays on (a phone works well):

* Android: install the app from [the latest release](https://github.com/TOS-main/edgemesh/releases/latest/download/edgemesh.apk). It hosts in the background, starts again after a restart, and has a Stop button in its notification. No Termux needed.
* Windows (PowerShell): `$env:EDGEMESH_HOST="1"; irm https://raw.githubusercontent.com/TOS-main/edgemesh/main/edgemesh/installers/install.ps1 | iex`
* Mac / Linux: `curl -fsSL https://raw.githubusercontent.com/TOS-main/edgemesh/main/edgemesh/installers/install.sh | EDGEMESH_HOST=1 sh`

The main device prints a **permanent link** (`https://tos-main.github.io/edgemesh/go.html#c=CODE`). It always finds the main device, even after its Cloudflare address or Wi-Fi address changes. Each mesh gets its own random Cloudflare address, with no account needed.

**Add your other devices** from any network: open the permanent link in a browser, or install with the code:

* Windows: `$env:EDGEMESH_CODE="CODE"; irm https://raw.githubusercontent.com/TOS-main/edgemesh/main/edgemesh/installers/install.ps1 | iex`
* Mac / Linux: `curl -fsSL https://raw.githubusercontent.com/TOS-main/edgemesh/main/edgemesh/installers/install.sh | EDGEMESH_CODE=CODE sh`

The installer asks how many cores to use, and whether to keep the computer awake while it helps. Change these later with `edgemesh limits --use both --cpu 50 --gpu 80 --cores 4 --awake on`. That command works on a running contributor, and the CPU and GPU each have their own limit.

**Help the creator's public mesh:** use `EDGEMESH_CODE=creator`, or run the installer with no code and choose 3. Helpers only compute. Answers are never shown to devices on a contribute link, unless the mesh owner turns that on.

## Quick start (from source)


**1. Start the main computer** (it also contributes its own compute). On Windows double-click
`start-windows.bat`; on macOS/Linux run `sh start-mac-linux.sh`. Or:

```
pip install .
edgemesh host            # add --public for a link that works from any network (see below)
```

It prints two links and two one-line installers:

- **Full link** `http://192.168.1.20:8000/?k=...`: chat, load models, change every device's limits.
  Only for your own devices.
- **Contribute-only link** `http://192.168.1.20:8000/contribute?c=...`: the device can only lend
  its compute. It sees its own stats and the answer it is helping with, nothing else. Good for a
  friend's phone or a school Chromebook. Both links work at the same time.

**2. Add devices**

| Device | How |
|---|---|
| Phone, tablet, Chromebook | Open a link (or scan the QR code on the Connect tab). With the full link, pick **Controller + worker** on the Devices tab; with the contribute link, tap **Start contributing**. No app or Termux. |
| Windows | Paste in PowerShell: `irm "http://<main>/install.ps1?c=<key>" \| iex` |
| macOS, Linux | Paste in Terminal: `curl -fsSL "http://<main>/install.sh?c=<key>" \| sh` |

The installers come from your main computer itself (no GitHub or website needed). They say what
they will do and ask first, install into a folder of their own (no admin rights), install Python
on Windows with winget if it's missing, and start `edgemesh contribute`: a background process
with no window that starts again whenever you sign in. It is not hidden: it shows in Task Manager
/ Activity Monitor, logs to `~/.edgemesh/contribute.log`, and is removed with
`edgemesh contribute --uninstall` on that computer or **Remove** on the main computer's Devices
tab (which also deletes the installed files). Only install it on computers you own or are allowed
to use. It starts at sign-in, not before: running before anyone signs in would need an admin service.

**3. Chat** on the Chat tab of any device with the full link. The conversation is kept by the
main computer, so every controller sees the same chat and answers stream onto all of them.
Each answer shows which devices computed it and how much of the work each did. From a terminal:
`edgemesh run "<link>" "Once upon a time" --max-new 50`.

### The mesh key

The main computer makes its mesh key once and keeps it in `~/.edgemesh/mesh.key`
(`%USERPROFILE%\.edgemesh\mesh.key` on Windows), so links and installed devices keep working
after restarts and updates. To pick your own, run `edgemesh host --key my-own-key` once; it is
saved and used from then on. `edgemesh links` (option 3 in the menu) prints the key and both
links again. If a device says "wrong mesh key", its link is from another mesh or for another
port: copy it again from `edgemesh links`.

### A phone as the main computer

A phone stays on with the screen off, so it can host instead of a laptop that sleeps. Install
**Termux** (from F-Droid or GitHub), then paste the line `edgemesh links` prints on the current
main computer: `curl -fsSL "http://MAIN:8000/termux.sh?k=KEY" | sh`. It installs EdgeMesh in
Termux, keeps your mesh key (so only the address in your links changes), starts hosting with an
https `--public` link, and holds a wake lock. By default the phone only coordinates
(`--no-worker`), which uses little battery; the setup can also let it compute on the side, capped
at 50% (`edgemesh-host --compute` / `--no-compute`). Other phones can compute natively in Termux
too, which is usually faster than the browser page: `curl -fsSL "http://MAIN:8000/termux.sh?c=CONTRIBUTE-KEY" | sh`. `edgemesh-host` starts it again,
`edgemesh-host-stop` stops it, and with the Termux:Boot app it starts after a restart. Set
Termux's battery setting to Unrestricted. Hosting downloads the model to the phone, so it needs
the free space.

## Devices on other networks (school, cellular)

`edgemesh host --public` opens a free Cloudflare quick tunnel and prints an `https://….trycloudflare.com`
link (new address on every start). It is ordinary web traffic, so it gets through most school and
office networks, and https lets phones and Chromebooks use WebGPU. Other options; pass the address
with `edgemesh host --public-url https://…` so the Connect tab shows it:

- **Tailscale** (or ZeroTier): a private network across all your devices; use the main computer's
  100.x address. Fastest for your own devices, since they can connect directly. Tailscale Funnel
  can also publish an https address.
- **ngrok**: `ngrok http 8000`, like Cloudflare; a free account gets a fixed address.
- **Port forwarding**: forward TCP+UDP 8000 on your router. Not possible on cellular/CGNAT.
- **A small cloud server** running `edgemesh host --no-worker`.

## Models

Load any supported model straight from Hugging Face, either from the **Model** box on the hub's web
page (paste a repo id or link) or on the command line:

```
edgemesh host --model Qwen/Qwen2.5-0.5B-Instruct
edgemesh host --model hf:meta-llama/Llama-3.2-1B-Instruct --hf-token hf_...   # gated models
edgemesh host --model ./my-local-hf-folder
```

Supported architectures: **Llama 2/3.x, Mistral, Qwen2/2.5/3, Phi-3, SmolLM/SmolLM2, TinyLlama,
GPT-2**, and any fine-tune of them (instruct, roleplay, "uncensored"/abliterated, merges). The repo
needs `.safetensors` weights and a `tokenizer.json`; GGUF-only repos aren't supported yet. Only the
hub downloads (with resume); it memory-maps the files and sends each device just its layers as
float16. The model's own chat template is applied when "Chat format" is on. `edgemesh pull <repo>`
downloads ahead of time.

Rough sizes at float16: 0.5B params ≈ 1 GB, 1.5B ≈ 3 GB, 3B ≈ 6 GB, 7-8B ≈ 15 GB, spread across
the devices. Browser workers need twice their share (they compute in float32).


### GGUF files (LM Studio, llama.cpp)

GGUF models load too: Q4_0/Q4_K/Q5_K/Q6_K/Q8_0/F16 and the rest that llama.cpp writes, for the
Llama, Mistral, Qwen 2/3 and Phi-3 families. On the Models tab:

- files in LM Studio's models folder on the main computer (`~/.lmstudio/models`) are listed under
  Downloaded models with a Load button (more folders: `EDGEMESH_GGUF_DIRS`);
- **Import a GGUF file** loads one by its path on the main computer, or uploads one from the device
  you're on (in 32 MB pieces, so it works through the https link and resumes if it drops);
- a Hugging Face repo that only has GGUF files downloads its Q4_K_M file, or the one you name:
  `org/model:Q8_0`.

The main computer expands each device's layers to float16 when it sends them, so devices
compute exactly as with safetensors models. Models split into several GGUF files, Gemma and
long-context (LongRoPE) Phi-3.5 aren't supported yet.

Tokenizers: the main computer uses the `tokenizers` package when it is installed and EdgeMesh's
own pure-Python BPE otherwise (so a phone in Termux can host), with the same tokens.

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

## Usage limits, battery and GPU

Every device has limits, editable on that device or from any controller (Devices tab), and from the
command line with `--cpu 60 --schedule "22:00-07:00=100,09:00-17:00=0" --always`:

- **Usage %** (default 80, up to 100; 0 pauses). Enforced by resting after each step.
- **Time windows**: "from 22:00 to 07:00 use 100%", by the device's own clock.
- **On battery / while charging**: e.g. on battery use 30% and 2 cores, plugged in use 100%.
  Browsers report charging on Chrome and Edge; native workers read it on Windows, macOS and Linux.
- **Cores**: how many CPU cores to use at once.
- **Always use**: by default a device that would slow the mesh down (the extra network hop costs
  more than it saves) waits on standby; this keeps it in the split anyway.
- **Use GPU**: on by default. NVIDIA cards use CUDA when PyTorch's CUDA build is installed (the
  installers offer it when they find an NVIDIA card; or run `edgemesh gpu --install-cuda`), with
  float16 weights. Other PCs use WebGPU (`wgpu`: Direct3D 12, Metal or Vulkan, so AMD, Intel and
  Apple GPUs too), and browsers use WebGPU when the page is https (the `--public` link) or
  localhost. On WebGPU the weights are 8-bit with one scale per 128 rows, which halves the memory
  each token reads (about twice as fast; `EDGEMESH_WEIGHTS=16` keeps float16). `edgemesh gpu`
  shows which GPU is used and its speed. A device uses the GPU only when it measures faster than
  its CPU; the usage % applies to the GPU too.

The Model tab estimates tok/s and memory per device for any Hugging Face repo before you load it,
shows the biggest model your devices can hold and run at ~10/5/1 tok/s, lists downloaded models
with **Load** and **Uninstall**, and defaults the max-token slider to the model's own default.

Phones pause background tabs. "Keep working when this tab is in the background" plays an
inaudible tone so the tab keeps running; Android may still stop it after a long time.

## Platforms

| | Contribute compute | Run prompts | Notes |
|---|---|---|---|
| Windows / macOS / Linux | one-line installer, app, or `pip install` | yes | direct UDP + relay; numpy (BLAS) or GPU |
| Android, ChromeOS | browser page | yes | relay; multi-core JS, WebGPU on https |
| iPhone / iPad | browser page (Safari) | yes | same; iOS pauses background tabs sooner |

## Tests

```
pip install -e ".[test]"
pytest                       # model split == single device, relay + direct paths, device dropout,
                             # browser workers in Chromium, JS math == numpy, every architecture
                             # checked against Hugging Face transformers, load-from-HF end to end
sudo scripts/nat_sim.sh      # three simulated networks: two home NATs + a symmetric "cellular" NAT
```

`nat_sim.sh` builds network namespaces with Linux NAT routers and checks that the two home
devices hole-punch a direct path while the symmetric-NAT device falls back to the relay, and that
generation completes across all three.

Build the double-click app for the current OS with `pip install pyinstaller . && python
scripts/build_app.py`; CI builds Windows, macOS and Linux versions on every push.

## Limits of this MVP and next steps

- Browser workers always use the relay. Next: WebRTC data channels so phones connect peer to peer.
- The GPU path multiplies matrices on the GPU and does the rest on the CPU, with one round trip
  per group of matrices. Next: whole layers on the GPU, and 4-bit weights.
- float16 weights. Next: int8/int4 quantization (4x smaller downloads for phones) and GGUF import.
- Gemma, Mixtral/MoE and long-context RoPE variants (longrope, yarn) aren't supported yet.
- One generation at a time, one hub, one shared chat. Next: batching, several models, hub failover.
- IPv4 only for hole punching; add IPv6 candidates.
- Native mobile apps would allow real background running on phones.
