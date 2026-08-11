# ROCmplete

ROCm without the scavenger hunt. ROCmplete builds and runs useful AMD GPU
applications as local, rootless containers with pinned software and verified
content.

Possibly the least frustrating way to run ROCm. That is pretty much the design
goal. Bringing up a new machine should take a few commands that you can inspect
and understand. If the host is not ready, the tool should tell you what is
wrong and what to do next.

ROCmplete targets AMD RDNA 4 discrete GPUs. That includes the 32 GB Radeon AI
PRO R9700, RX 9070 XT, RX 9070, and RX 9070 GRE using `gfx1201`, plus the RX
9060 XT and RX 9060 using `gfx1200`. It also targets AMD Ryzen AI Max / Strix
Halo (`gfx1151`) and AMD Ryzen AI 300 / Strix Point (`gfx1150`). The build and
runtime enforce all four architectures, but the project remains pre-release
while its target-hardware acceptance matrix is being completed.

ROCmplete currently manages:

- ComfyUI for image generation, image editing, and video on port 8188
- llama.cpp as an OpenAI-compatible API server and interactive GGUF CLI on
  port 8080, with selectable ROCm and Vulkan backends
- DwarfStar as an experimental, high-memory DeepSeek V4 Flash server and CLI
  on port 8000

## Hardware exercised so far

These are real machines used during development. The scope column matters. A
working workload is useful evidence, but it does not imply that every row in
the [target-hardware acceptance matrix](docs/hardware-acceptance.md) has
passed.

| Host | GPU target | Workloads exercised |
| --- | --- | --- |
| Fedora Kinoite 44, Ryzen AI 9 HX 370, 128 GB DDR5-5600 SODIMM | Strix Point, `gfx1150` | DwarfStar DeepSeek V4 Flash and the managed Qwen3.6 llama.cpp presets |
| Fedora Linux 44 (non-OSTree), Ryzen AI Max+ 395, 128 GB LPDDR5X-8000 | Strix Halo, `gfx1151` | DwarfStar DeepSeek V4 Flash at 4K and 128K context; managed Qwen3.6 llama.cpp MTP/tool, OMP, and ROCm/Vulkan paths; Muse Glimmer agent probes; Laguna XS and Ling feasibility controls |
| Ubuntu 26.04, Ryzen AI Max+ 395, 128 GB LPDDR5X-8000 | Strix Halo, `gfx1151` | DwarfStar DeepSeek V4 Flash and the managed Qwen3.6 llama.cpp presets |
| SteamOS 3.8, Radeon RX 9070 XT 16 GB | RDNA 4, `gfx1201` | ComfyUI and the Qwen3 0.6B llama.cpp smoke |

## Why ROCmplete

Getting a container to start is the easy part. It does not prove that PyTorch
found the right GPU or that inference survives contact with the hardware.
ROCmplete tries to cover the whole path.

- Ubuntu base images, ROCm and PyTorch versions, application commits,
  dependencies, source patches, models, and workflows have explicit pins.
- One checkout builds for `gfx1201`, `gfx1200`, `gfx1151`, and `gfx1150`.
  Runtime checks confirm the architecture and apply the matching policy.
- `doctor` checks Podman, GPU devices, permissions, SELinux, shared-memory
  setup, the detected architecture, and a real PyTorch tensor operation.
- Downloads are pinned by revision or model-version ID, size, and SHA-256.
  License state remains explicit, downloads resume, and content survives image
  rebuilds.
- Containers are rootless, read-only, capability-free, and expose only the
  selected GPU devices. Web applications publish on loopback by default.
- Optional OpenCode, Pi, Oh My Pi, and Maki launchers add a bubblewrap filesystem
  boundary around local coding-agent work.
- `acceptance run` checks more than startup. It runs small real workloads,
  checkpoints progress, and collects visual review after unattended work.

## Requirements

The host needs rootless Podman and Python 3.12 or newer. The field-tested hosts
above use SteamOS 3.8, Fedora 44 in conventional and Kinoite deployments, and
Ubuntu 26.04. A minimal installation may not include Podman yet.

GPU use needs read/write access to `/dev/kfd` and the selected
`/dev/dri/renderD*` nodes. Run Doctor before changing permissions or kernel
settings. It reports the problem and the host-specific action when one is
needed. The [host GPU access guide](guide/tuning.md#host-gpu-access) explains
the SELinux, group, and udev choices.

The optional sandboxed agent launchers also need bubblewrap (`bwrap`). Use the
distribution `bubblewrap` package from `apt`, `dnf`, or `pacman` when possible.
Doctor reports Ubuntu's AppArmor user-namespace policy when it can interfere
with a launcher.

## Quick start

Clone the source and inspect the host. ROCmplete has no host-side Python
dependencies and does not publish prebuilt application images.

```bash
git clone https://github.com/fff7d1bc/rocmplete.git
cd rocmplete
./rocmplete --version
./rocmplete doctor
```

Before the first build, Doctor reports that its containerized PyTorch probe was
skipped. Build all applications, inspect the available content, and choose a
recipe:

```bash
./rocmplete build all
./rocmplete doctor
./rocmplete content install
```

`build all` builds ComfyUI, llama.cpp, and the experimental DwarfStar image.
Build one application instead when that is all you need. Models range from
hundreds of MiB to more than 100 GiB, so inspect a dry run before a large
installation.

```bash
./rocmplete content install llama-cpp qwen3.6 --dry-run
```

Use the built-in guides for short, copyable walkthroughs:

```bash
./rocmplete guide comfyui
./rocmplete guide llama-cpp
./rocmplete guide dwarfstar
```

### ComfyUI

The practical image recipe installs Qwen Image FP8 Lightning and its curated
workflow:

```bash
./rocmplete build comfyui
./rocmplete content install comfyui image
./rocmplete run comfyui
```

Open `http://127.0.0.1:8188`. Image editing, T2V, I2V, imported content,
Manager, and multi-GPU graphs are covered in the
[application guide](guide/applications.md#comfyui).

### llama.cpp

The Qwen3.6 recipe installs the practical dense and sparse MTP choices:

```bash
./rocmplete build llama-cpp
./rocmplete content install llama-cpp qwen3.6
./rocmplete run llama-cpp server --preset qwen3.6-27b-mtp-q8-0
```

For an API serving several installed presets, use the managed router:

```bash
./rocmplete run llama-cpp server --router --models-max 1
```

The [llama.cpp guide](guide/applications.md#llamacpp) explains presets,
contexts, MTP, DFlash, translations, the terminal CLI, tool calling, and
choosing between the managed Qwen variants. High-memory hosts can also install
the Ornith and KAT-Coder Q8 coding-agent candidates independently without
changing the default:

```bash
./rocmplete content install llama-cpp ornith
./rocmplete content install llama-cpp kat-coder
```

Laguna XS 2.1 is a separate, smaller sparse coding model. Its official
Q4_K_M GGUF starts at 256K, preserves interleaved reasoning across tool turns,
and is exposed by all managed agent clients without replacing Qwen as their
default:

```bash
./rocmplete content install llama-cpp laguna-xs-2.1 --accept-license
./rocmplete run llama-cpp server --preset laguna-xs-2.1-q4-k-m
```

Muse Glimmer is a separate 30B family using Meta's official dynamic K-quant
target and DFlash draft. Its recipe starts the 128K accelerated preset by
default and also installs a non-speculative control plus an experimental
forced-256K DFlash policy without duplicating model content:

```bash
./rocmplete content install llama-cpp muse-glimmer
./rocmplete run llama-cpp server \
  --preset muse-glimmer-30b-kquant-dynamic-dflash
```

Use `muse-glimmer-30b-kquant-dynamic-dflash-256k` only for long-context
acceptance; Meta's released target and draft metadata declare 128K. The
managed agent clients expose all three presets and llama.cpp preserves Muse's
parsed reasoning across turns, but task depth still depends on the client
scaffold and prompt.

For Japanese and English translation on a high-memory host, the separate
Shisa V2.1 recipe installs the 70B Q8_0 model and requires acknowledgment of
the Llama 3.3 terms:

```bash
./rocmplete content install llama-cpp shisa-v2.1 --accept-license
./rocmplete run llama-cpp server \
  --preset shisa-v2.1-llama3.3-70b-q8-0
```

### DwarfStar

DwarfStar serves the pinned DeepSeek V4 Flash 0731 Q2 imatrix model. It uses
IQ2_XXS for routed gate/up weights, Q2_K for routed down weights, and Q8 for
attention projections, shared experts, and output. The model is about 80.76
GiB before context and working allocations, so this path is for a host with
enough GPU-mapped memory:

```bash
./rocmplete build dwarfstar
./rocmplete content install dwarfstar flash-0731-q2-imatrix
./rocmplete run dwarfstar server
```

The [DwarfStar guide](guide/applications.md#dwarfstar) covers its 128K managed
context, memory setup, API, agent-client providers, and bounded acceptance run.

## Everyday use

Inspect local state, start an installed application, and let `auto` select the
hardware profile:

```bash
./rocmplete status
./rocmplete run comfyui
```

Override the profile only when testing or diagnosing:

```bash
./rocmplete run comfyui --profile rdna4
./rocmplete run comfyui --profile strix-halo
./rocmplete run comfyui --profile strix-point
./rocmplete run comfyui --profile cpu
```

Web applications publish on `127.0.0.1` by default. Select one exact LAN or
Tailscale address only when unauthenticated network access is intentional:

```bash
./rocmplete run comfyui --listen 192.168.1.50
```

For local coding-agent work, start the llama.cpp router and use the sandboxed
PATH launcher. At least one managed agent model must already be installed.

```bash
./rocmplete content install llama-cpp qwen3.6
./rocmplete run llama-cpp server --router --models-max 1
export PATH="$PWD/bin:$PATH"
opencode
# or: pi
# or: omp
# or: maki
```

Muse Glimmer is available through the same four clients after
`./rocmplete content install llama-cpp muse-glimmer`. Select its base, 128K
DFlash, or experimental forced-256K DFlash preset in the client's model
picker.

OpenCode starts new sessions in read-only Investigate mode. All four launchers
keep the current directory and private client state writable while hiding the
real home directory, credentials, Podman state, and GPU devices. The
[tool-using client guide](guide/applications.md#tool-using-clients) documents
models, reasoning variants, agent modes, sandbox limits, and escape hatches.

On a multi-GPU host, repeat `--render-node` for every card intended for one
supported workload. ROCmplete never guesses the set and requires matching
architectures:

```bash
./rocmplete run llama-cpp server \
  --model /path/to/large-model.gguf \
  --render-node /dev/dri/renderD128 \
  --render-node /dev/dri/renderD129
```

llama.cpp uses layer splitting automatically. ComfyUI needs graph nodes that
place components or work on the selected cards. See the
[application guide](guide/applications.md) and
[runtime tuning guide](guide/tuning.md#runtime-policies).

Foreground runs own the container lifecycle. Ctrl-C stops and removes the
container. Detached runs are managed explicitly:

```bash
./rocmplete logs comfyui --follow
./rocmplete stop comfyui
```

Update without rewriting local work or persistent content:

```bash
git pull --ff-only
./rocmplete doctor
./rocmplete build all
```

## Builds, content, and storage

The three main operations remain independent and retryable:

```text
build  ->  content install  ->  run
image      models/workflows     application
```

Normal builds reuse Podman layers and a local Python package cache. Use
`--no-layer-cache` to rerun one application image or `--no-cache` for a fully
cold build. The [operations guide](guide/operations.md#builds-and-local-caches)
explains prerequisite images, cache boundaries, cleanup, and image transfer.

Content discovery starts with practical recipes and expands only when asked:

```bash
./rocmplete content list
./rocmplete content list --bundles
./rocmplete content list --families
./rocmplete content list --models --details
./rocmplete content import
```

Set `HF_TOKEN` before a large Hugging Face installation when you have one.
`CIVITAI_TOKEN` is needed only for authenticated Civitai imports and
user-owned packs. ROCmplete passes supplied tokens only to its download tools
and does not store them in images or persistent state. The
[content guide](guide/content.md) covers recipes, exact bundles, terms,
verification, resumable downloads, mirrors, imports, and workflows.

Persistent data defaults to
`${XDG_DATA_HOME:-$HOME/.local/share}/rocmplete`. Put large content on another
filesystem with `${XDG_CONFIG_HOME:-$HOME/.config}/rocmplete/config.toml`:

```toml
[storage]
data_dir = "/mnt/ai/rocmplete"
```

The configuration is optional and ROCmplete never creates or migrates it.
See [persistent data](guide/operations.md#persistent-data) before moving or
cleaning application state, models, inputs, or outputs.

## Acceptance and benchmarks

After onboarding a host or updating software, run the bounded checkpointed
smoke suite:

```bash
./rocmplete acceptance run --dry-run
./rocmplete acceptance run
```

Automated workloads finish before the visual review pass, so the run can be
left unattended. The [operations guide](guide/operations.md#target-hardware-smoke-acceptance)
explains resume behavior, result files, and what `PASS`, `FAIL`, and `BLOCKED`
mean.

Compare llama.cpp's ROCm and Vulkan backends on the exact model you use:

```bash
./rocmplete benchmark llama-cpp \
  --preset qwen3.6-27b-q8-0 \
  --compare-backends
```

Sparse and dense models, and even different quantizations from one family,
can prefer different backends on the same GPU. The
[tuning guide](guide/tuning.md#benchmarks) covers repeatable comparisons.

Evaluate a managed model as a coding agent against the frozen Go and Python
task suite:

```bash
./rocmplete benchmark agent --list-tasks
./rocmplete benchmark agent \
  --preset qwen3.6-27b-q8-0 --dry-run
./rocmplete benchmark agent \
  --preset qwen3.6-27b-q8-0
```

This is separate from smoke acceptance and native token-speed benchmarking.
It runs Pi against disposable single-commit fixtures, applies hidden tests
after each implementation attempt, and preserves raw transcripts, patches,
server logs, and a Markdown summary below managed application data. The
[operations guide](guide/operations.md#coding-agent-evaluation) explains the
fixed-harness policy, review tasks, grading, repetitions, and result scope.

## User guides

- [Applications](guide/applications.md) covers ComfyUI, llama.cpp, DwarfStar,
  managed models, APIs, OpenCode, Pi, Oh My Pi, Maki, and multi-GPU workloads.
- [Content](guide/content.md) covers recipes, exact bundles, licenses,
  verification, resumable downloads, mirrors, imports, and workflows.
- [Operations](guide/operations.md) covers acceptance, builds, caches, image
  archives, persistent state, logs, stop, and scoped cleanup.
- [Tuning and benchmarks](guide/tuning.md) covers host GPU access, runtime
  policies, RDNA 3.5 shared memory, RDNA 4, and repeatable measurements.

Command-specific help remains the authoritative interface reference:

```bash
./rocmplete --help
./rocmplete build --help
./rocmplete content --help
./rocmplete run --help
./rocmplete acceptance --help
./rocmplete benchmark agent --help
```

Human-facing output uses semantic colors on terminals. Redirected output stays
plain, and `NO_COLOR=1` disables styling.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and
[catalog/README.md](catalog/README.md) for provenance and catalog policy.
Maintainers should start with [docs/README.md](docs/README.md). The bounded
smoke command complements, but does not replace, the complete
[maintainer acceptance matrix](docs/hardware-acceptance.md).

## Contributing and security

Focused fixes, hardware results, and careful improvements are welcome. See
[CONTRIBUTING.md](CONTRIBUTING.md) for the source map, validation commands, and
the details that make a useful bug report. Report security-sensitive issues
through the private path documented in [SECURITY.md](SECURITY.md), not a
public issue.

## License

ROCmplete source is available under the [BSD 3-Clause License](LICENSE).
Downloaded models, workflows, and bundled third-party components retain their
own terms. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and the catalog
license metadata before using them.
