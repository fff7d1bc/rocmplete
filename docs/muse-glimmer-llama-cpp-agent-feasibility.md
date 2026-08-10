# Muse Glimmer llama.cpp agent feasibility snapshot

This is a dated maintainer research record from 2026-08-10. It captures the
comparison that selected Meta's official dynamic K-quant Muse Glimmer GGUF,
kept DFlash and the managed coding-agent integrations, and declined to add a
second full-size BF16 choice.

This document is evidence, not a current version or support declaration.
Always inspect `Containerfile`, `catalog/catalog.json`, source, and tests for
the current pins and behavior before using it. Hostnames, usernames, private
paths, and other machine-specific identifiers are intentionally omitted.

## Question and decision

An initial OpenCode session with the Unsloth Dynamic Q8 target returned a very
short repository summary. That raised four questions which needed to be kept
separate:

1. Was Muse Glimmer intrinsically unsuitable for coding-agent work?
2. Was the Unsloth conversion behaving differently from Meta's official
   release?
3. Was DFlash or the experimental forced-256K context responsible?
4. Was the client scaffold asking for a shallow answer?

The decision was to **keep the family and agent integrations, but replace the
managed target with Meta's official dynamic K-quant GGUF**:

- the official target is substantially smaller than the Q8 and BF16
  alternatives and had the best measured decode throughput;
- it completed a broad, tool-using repository task through Maki at 128K with
  DFlash;
- separate Q8 runs at both 128K and forced 256K also completed substantial
  tool-using tasks, so neither DFlash nor the extended preset explained the
  original shallow response;
- the exact BF16 task also completed correctly, but its much larger resident
  memory, slower decode, and similar final utility did not justify a second
  catalog choice; and
- llama.cpp's `--reasoning-preserve` policy is now owned by all three managed
  Muse presets, while Muse remains distinct from models that support a
  client-selectable reasoning-effort budget.

The result is one installed model and draft with three orthogonal launch
policies: a non-speculative 128K control, the default 128K DFlash policy, and
an experimental forced-256K DFlash policy.

## Snapshot under test

| Component | Tested value |
| --- | --- |
| ROCmplete source | `6d59816` before the integration change |
| Project ROCm | `7.14.0` |
| llama.cpp | `62bf73d25c53b8161f8a22894d4f90c4aebbd7d0` |
| Normal llama.cpp image | `localhost/rocmplete:llama-cpp-ubuntu26.04-rocm7.14-62bf73d-r15` |
| Profile and architecture | Strix Halo, `gfx1151`, Radeon 8060S |
| Host memory | 128 GB LPDDR5X, 112 GiB TTM/GTT policy |
| Host software | Fedora Linux 44 non-OSTree, kernel `7.1.7-200.fc44.x86_64`, rootless Podman |
| Agent clients | Maki 0.4.5 and OpenCode 1.18.15 |

The server probes used the normal ROCmplete confinement: rootless Podman,
read-only root, dropped capabilities, `no-new-privileges`, the selected render
node, and `/dev/kfd`. No tested run produced a GPU reset, device loss, OOM, or
kernel fault.

## Immutable model candidates

### Selected official Meta target and draft

Both files come from
[`meta-models/Muse-Glimmer-30B-GGUF`](https://huggingface.co/meta-models/Muse-Glimmer-30B-GGUF/tree/93769bc7ab5ad1e9cd22d857e3138cf5d977ae81)
at revision `93769bc7ab5ad1e9cd22d857e3138cf5d977ae81`.

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `muse-glimmer-30B-kquant-dynamic.gguf` | 19,653,957,984 | `513109c8319115f69eb09fb7b118c97c8167d15bc014fd7670d2e30489bf106c` |
| `dflash-kquant.gguf` | 1,631,205,312 | `27d9a805fa29b943cfb6ad4843367cd4eaaaf06bd452d8cc3e00a2cd18a677bc` |

The installed total is 21,285,163,296 bytes, or about 19.82 GiB. Meta
describes the dynamic K-quant as approximately four-bit and reports about
0.2% average degradation across its 15-benchmark comparison. Those upstream
quality results were not independently reproduced here.

### Previously managed Unsloth Q8 target

The comparison retained the exact previously cataloged file from
[`unsloth/Muse-Glimmer-30B-GGUF`](https://huggingface.co/unsloth/Muse-Glimmer-30B-GGUF/tree/c3ac2ebf47426591b4c6d408103c8c15a1e2afd6)
at revision `c3ac2ebf47426591b4c6d408103c8c15a1e2afd6`:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `Muse-Glimmer-30B-UD-Q8_K_XL.gguf` | 32,300,651,040 | `e63bf23b7710ecdea2579e4b1de58980c4a2b446e8ecf48b782cfcefd2e31770` |

It was removed from the managed catalog because the official K-quant is the
better practical single choice, not because this Q8 conversion failed agent
workloads.

### BF16 comparison

Meta publishes the official BF16 safetensors checkpoint at revision
`f84ecc3a0ea984a4c04542a84269e3d065350a6e`, but no official BF16 GGUF was
available in the tested repository. The llama.cpp comparison therefore used
Unsloth's BF16 GGUF conversion at revision
`faa5b025c584459c13febfa5c59883516710ae39`:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `Muse-Glimmer-30B-BF16-00001-of-00002.gguf` | 29,805,191,936 | `9ed3c904ca80be99e787d95cc430c5b1ce3a44cb7acc012fda63c8e1649df7e5` |
| `Muse-Glimmer-30B-BF16-00002-of-00002.gguf` | 25,920,319,232 | `bbec2aef54b01bff34977a7da2a6e9e73c3edc063fabf0ae4bd62817b7efe1ae` |

The target totals 55,725,511,168 bytes, or about 51.90 GiB, before the DFlash
draft, KV cache, and runtime buffers.

## Fixed native benchmark

All three target files were measured through the same image, backend, host,
and llama-bench shape:

- ROCm backend and one `gfx1151` device;
- prompt processing 512 and text generation 128;
- three repetitions at context depth zero;
- batch and microbatch 4096;
- Q8_0 K and V caches; and
- Flash Attention enabled.

| Target | pp512 tokens/s | tg128 tokens/s |
| --- | ---: | ---: |
| Official dynamic K-quant | 341.168 ± 3.178 | 10.318 ± 0.013 |
| Unsloth Dynamic Q8_K_XL | 376.800 ± 6.109 | 7.351 ± 0.012 |
| Unsloth BF16 | 483.192 ± 6.756 | 4.109 ± 0.004 |

Higher precision improved prompt processing in this fixed shallow benchmark,
but decode moved in the opposite direction. The selected K-quant decoded
about 40% faster than Q8 and about 151% faster than BF16. Do not generalize
these figures across llama.cpp revisions, backends, cache policies, context
depths, or hardware.

The three native benchmark JSON results were retained outside the repository.
Machine-specific result paths are deliberately omitted here.

## Server capacity observations

Fresh 128K servers used four slots, Q8_0 K/V cache, Flash Attention, DFlash,
and reasoning preservation.

| Target | Ready time | Container memory after load |
| --- | ---: | ---: |
| Official dynamic K-quant | 13.997 s | 30.64 GB |
| BF16 | 37.494 s | 66.77 GB |

After the long BF16 agent task, container memory was 73.83 GB. These are
observations from the container accounting on one integrated-memory host,
not exact model-allocation or peak-memory guarantees.

## Agent-harness results

The repeated task prompt was `Can you review the code in this project for
me?`. The checkout was read-only to the model and was checked for mutations
after each run.

| Target and policy | Client | Observed behavior |
| --- | --- | --- |
| Official K-quant, DFlash, 128K | Maki | Completed a broad repository review in about 7.5 minutes, used roughly 40.7K context, and returned about 1.3K final tokens |
| Unsloth Q8, DFlash, 128K | Maki | Completed a substantive review in about 8.5 minutes, used roughly 53.8K context and about 30 turns, and returned about 1.5K final tokens |
| Unsloth Q8, DFlash, forced 256K | Maki | Completed a substantive review in about 5 minutes, used roughly 33.8K context, and returned about 1.4K final tokens |
| Unsloth BF16, DFlash, 128K | Maki | Completed in 453.363 seconds with 28 turns, 31 tool calls, 33,278 final-slot tokens, and 1,734 final tokens |

The exact BF16 tool-call breakdown was 11 shell calls, two glob calls, two
grep calls, five index calls, seven reads, and four todo updates. It ran the
project compile and unit-test checks and returned a coherent architecture and
maintenance review. It did not produce a uniquely valuable coding finding
that justified the additional memory or lower decode rate. The final BF16
DFlash turn accepted 1,115 of 9,270 proposals, or 12.03%; earlier tool turns
were commonly in the 20% to 49% range.

The earlier short OpenCode response did not correspond to a server crash.
OpenCode had completed the turn normally after reading only the root README.
Its built-in short-answer bias and the prompt/scaffold used for that session
allowed a minimal answer. A stronger investigation prompt can drive much more
hidden reasoning, while Maki's default project-task scaffold naturally drove
breadth. This is a harness and prompting difference, not proof that one GGUF
is categorically more diligent.

The test evidence therefore supports all three managed agent catalogs. Maki
has the strongest live Muse evidence. OpenCode and Pi remain protocol-
compatible and useful, but task depth is scaffold-sensitive and should be
judged on the actual requested workflow. `reasoning_preserve` keeps parsed
reasoning available to the server's multi-turn history; it does not force a
client to explore more files or expose a Qwen-style effort selector.

## Failures and misleading signals

Several failures were environmental or fixture-related rather than model
failures:

- A plain HTTP resume of the first BF16 shard produced a file about 9.3 MB too
  large. Its size and hash failed and the file was discarded. Fresh downloads
  through the pinned Hugging Face/Xet path matched both recorded sizes and
  hashes.
- A disposable research image contained the correct llama.cpp binary commit
  but an older entrypoint without DFlash handling. The normal r15 image's
  current entrypoint succeeded. This is why an entrypoint policy change must
  bump the managed application image tag; source commit identity alone is not
  sufficient.
- An initial Maki fixture used `git clone --shared`. Its object store lived
  outside Maki's bubblewrap view, so Git objects appeared inaccessible and the
  model spent turns diagnosing a broken checkout. A self-contained
  `git clone --no-local` fixed the fixture. This was not model confusion.
- Unit-test failures involving `/run/rocmplete/runtime` occurred inside the
  agent sandbox and reflected its intentionally restricted runtime view. They
  were not an inference crash.

These cases are worth preserving because they can otherwise make a healthy
model or server look unreliable.

## Integration consequences

The accepted integration deliberately does not add vision, an mmproj file, a
BF16 preset, a second Muse recipe, a llama.cpp source update, or client prompt
hacks. It keeps model family, precision, speculation, context, and harness
policy orthogonal.

All three Muse presets enable Jinja, advertise the reviewed function-tool
contract, and set `reasoning_preserve`. The 128K DFlash preset remains the
recipe default. The 256K preset still forces both target and draft
`context_length` metadata and remains experimental until useful prompts
beyond 128K pass retrieval, quality, draft-acceptance, latency, and memory
checks.

The removed Q8 file is managed content from an older catalog, but ROCmplete
does not silently delete persistent model bytes during an upgrade. After the
new bundle installs and verifies successfully, an operator may manually
remove the obsolete Q8 file if it is no longer needed.

## Candidate integration verification

The completed integration was synced to the same target host and built as
`localhost/rocmplete:llama-cpp-ubuntu26.04-rocm7.14-62bf73d-r16`, image ID
`15aa29c45b41f011f5edacd9f5fb761db26eae488d447453414e2d1b2a9e07a3`.
The content installer reused the retained official target through its normal
local-mirror path, rechecked all 19,653,957,984 bytes against the catalog
SHA-256, copied them atomically into the managed partition, and recorded a
current verification receipt. The DFlash receipt remained current.

A direct managed 128K DFlash server then:

- reached healthy state on ROCm with four 131072-token slots;
- ran the official target and draft with `--jinja --reasoning-preserve`,
  `draft-dflash`, and 15 maximum draft tokens;
- used 25.17 GB of container memory after load in this warm follow-up run;
- returned exact content `OK` for a bounded request while exposing its parsed
  reasoning separately; and
- generated at 24.73 tokens/s for that short request, accepting 81 of 420
  draft proposals.

The values from one short request are a wiring probe, not a performance
benchmark. The server stopped and its container was removed cleanly.

The managed router also started successfully from the generated private INI
and loaded the 128K DFlash section on demand. Its bounded request returned
exact content `ROUTER_OK`. All three Muse sections contained
`reasoning-preserve = true`; the two DFlash sections retained their draft
settings, and the 256K section retained its two architecture overrides plus
`fit = off`. This closes the stale-entrypoint failure found during research
and verifies both runtime paths without another long agent evaluation.

## Retest triggers

Repeat the comparison when any of these changes materially:

- Meta publishes an official full-precision GGUF or a materially different
  dynamic quant;
- Muse Glimmer's template, DFlash implementation, reasoning parser, or
  `--reasoning-preserve` behavior changes in llama.cpp;
- the ROCm backend, quantized KV cache, or Flash Attention changes on
  `gfx1151`;
- a maintained client changes its system scaffold, compaction, tool protocol,
  or model metadata schema; or
- forced-256K prompts demonstrate a quality, stability, or memory problem
  that the 128K control does not reproduce.

Use the same immutable bytes, server image, prompt corpus, slot/context
settings, sampler policy, tool permissions, and fresh self-contained checkout
when comparing. Record complete turn/tool traces and output sanity, not just
load success, token rate, or final-answer length.
