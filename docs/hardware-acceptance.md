# Target-hardware acceptance

ROCmplete targets `gfx1201`, `gfx1200`, `gfx1151`, and `gfx1150`, but a
successful build, CPU startup, or unit test is not GPU inference acceptance.
This matrix defines a finite pre-release gate and prevents one convenient
workflow from standing in for the complete target surface.

Keep machine-specific logs, benchmark JSON, generated media, and measurements
outside the source tree. When a row is accepted, record a short durable
summary here with the date, Git commit, image tag, profile, and result
location. Use `PASS`, `FAIL`, `BLOCKED`, or `N/P`; `N/P` requires a documented
hardware-capacity or application-policy reason.

## Field-tested hosts

The following machines have completed useful runtime testing during
development. These observations establish practical coverage but do not mark
the detailed acceptance rows below as `PASS`. Formal results still need the
commit, image IDs, exact commands, and retained result locations described in
this document.

| Host | Architecture | Observed workload scope |
| --- | --- | --- |
| Fedora Kinoite 44, Ryzen AI 9 HX 370, 128 GB DDR5-5600 SODIMM | Strix Point, `gfx1150` | DwarfStar DeepSeek V4 Flash and the managed Qwen3.6 llama.cpp presets; DwarfStar generation was about 3.9 tokens/s |
| Ubuntu 26.04, Ryzen AI Max+ 395, 128 GB LPDDR5X-8000 | Strix Halo, `gfx1151` | DwarfStar DeepSeek V4 Flash and the managed Qwen3.6 llama.cpp presets |
| SteamOS 3.8, Radeon RX 9070 XT 16 GB | RDNA 4, `gfx1201` | ComfyUI and the Qwen3 0.6B llama.cpp smoke |

## Automated bounded smoke

Start acceptance on a new or updated host with:

```bash
./rocmplete acceptance run --dry-run
./rocmplete acceptance run
```

The checkpointed suite covers exact GPU/CPU device isolation, short ComfyUI
image and video workloads, one llama.cpp workload, and a bounded DwarfStar
generation. DwarfStar is included by default on Strix Halo and must be
selected explicitly elsewhere because of its memory footprint. Generated
media requires explicit human review before its case becomes `PASS`; the
review phase starts only after all selected automated workloads finish.
Prompts and the Markdown report retain case-specific functional criteria so
aesthetic defects are not confused with broken inference. Unattended runs
leave those cases `BLOCKED`. Preserve the JSON and Markdown result below
`apps/acceptance/results/` and resume an interrupted run with `--resume`.

This command is the fast recurring gate. It intentionally does not replace
the rows below: edit and I2V behavior, additional model families, precision
comparisons, forced-profile failures, memory policy comparisons, and sustained
performance still require the finite manual matrix.

## Host diagnostics

Complete every row on all target hosts before describing their profiles as
accepted.

| Check | RX 9060 family / `gfx1200` | R9700 or RX 9070 family / `gfx1201` | Strix Halo / `gfx1151` | Strix Point / `gfx1150` |
| --- | --- | --- | --- | --- |
| `doctor` reports the exact architecture | pending | pending | pending | pending |
| `auto` resolves the expected profile | pending | pending | pending | pending |
| forcing the expected profile succeeds | pending | pending | pending | pending |
| forcing either other profile fails closed | pending | pending | pending | pending |
| exactly the selected render-node set is exposed | pending | pending | pending | pending |
| CPU mode exposes no GPU devices | pending | pending | pending | pending |
| device access needs no broader container privileges or disabled SELinux labels | pending | pending | pending | pending |
| RAM, TTM module/ceiling, and effective GTT are reported | N/P: dedicated VRAM | N/P: dedicated VRAM | pending | pending |

Record for each host:

```text
Date:
Git commit:
Kernel and distribution:
GPU and system RAM:
Render node:
ROCm/PyTorch base image ID:
Application image IDs:
Persistent-data location:
Result/log location:
```

## Representative application gate

Use the catalog-owned workflow or preset and its default runtime policy unless
the row says otherwise. For every generation row, inspect output sanity as
well as successful process completion. Exercise I2V/edit rows with a fixed
local source image.

| Application path | Exact content or comparison | RX 9060 family | R9700 or RX 9070 family | Strix Halo | Strix Point |
| --- | --- | --- | --- | --- | --- |
| ComfyUI image | `qwen-image-2512-fp8-lightning` | pending | pending | pending | pending |
| ComfyUI edit | `qwen-image-edit-2511-fp8-lightning` | pending | pending | pending | pending |
| ComfyUI Wan T2V | `wan-2.2-t2v-14b-fp8-lightning` | pending | pending | pending | pending |
| ComfyUI Wan I2V | `wan-2.2-i2v-14b-fp8-lightning` | pending | pending | pending | pending |
| LTX-2 T2V camera graph | `ltx-2-t2v-19b-fp8-full`, ordinary and one enabled camera adapter | pending | pending | pending | pending |
| LTX-2 I2V camera graph | `ltx-2-i2v-19b-fp8-full`, ordinary and one enabled camera adapter | pending | pending | pending | pending |
| Hunyuan T2V | `hunyuan-video-1.5-t2v-480p-cfg-distilled` | pending | pending | pending | pending |
| Hunyuan I2V | `hunyuan-video-1.5-i2v-480p-step-distilled` | pending | pending | pending | pending |
| llama.cpp offload smoke | `llama-qwen3-0.6b-q8-0` | pending | pending | pending | pending |
| llama.cpp assistant | `llama-qwen3.6-27b-mtp-q8-0` | pending | pending | pending | pending |
| llama.cpp Qwen tool protocol | `qwen3.6-35b-a3b-mtp-ud-q8-k-xl`, complete nested tool round trip | N/P unless model and context fit the card | N/P unless host memory is deliberately used | pending | pending |
| llama.cpp 35B-A3B agent evaluation | `qwen3.6-35b-a3b-ud-q8-k-xl` first, then `qwen3.6-35b-a3b-mtp-ud-q8-k-xl` | N/P unless model and context fit the card | N/P unless host memory is deliberately used | pending | pending |
| llama.cpp coding/agent | `llama-laguna-s-2.1-q4-k-m`, 256K context | N/P unless model and context fit the card | N/P unless host memory is deliberately used for offload | pending | pending |
| DwarfStar direct-answer smoke | DeepSeek V4 Flash 0731 Q2 imatrix (routed IQ2_XXS/Q2_K, Q8 attention/shared/output), 4K context, 64-token ceiling | N/P unless host memory offload is deliberately provisioned | pending | pending | pending |

DwarfStar remains experimental after the bounded smoke. Before promoting it,
also run the 128K server default, normal thinking and direct-answer requests,
multi-turn cache reuse, a long enough generation to expose decode faults, and
clean interruption/removal. Record model-load peak memory and sustained
generation speed. On non-Halo profiles, invoke the smoke with
`--application dwarfstar`; that explicit selection is a capacity opt-in, not
evidence of prior acceptance. Record whether an APU host used the general 112
GiB TTM/GTT starting point or DwarfStar's roughly 124 GiB upstream
recommendation, and whether `amd_iommu=off` was enabled. The initial 112 GiB
manual 128K run used
it, so memory-capacity acceptance and IOMMU performance remain separate
questions. DSpark, MTP, multi-GPU, distributed
execution, and SSD streaming are not part of the current application contract.

Laguna remains experimental even after basic startup until chat templating,
tool use, sustained generation, and output sanity are accepted.

For the Qwen tool-protocol row, start the managed router and inspect `/props`
for the expected template capabilities and context. Send a developer message
and a tool schema with a nested object, require a structured tool call, return
the result as a tool message, and require a final answer. Repeat the exchange
with streaming enabled. If the MTP preset fails, repeat it with
`qwen3.6-27b-q8-0` to separate template handling from speculative decoding.
Only after the raw API exchange passes should read-only OpenCode and Pi tasks
be used as the final integration checks.

For the 35B-A3B agent-evaluation row, start the non-MTP preset first and leave
OpenCode's edit, shell, and subagent approvals enabled. Repeat a focused
Investigate-mode repository question, a delegated local investigation, a
bounded delegated web investigation, a raw nested-tool round trip, and the
concurrent nonce corruption probe before allowing a disposable edit task.
Confirm that only the two hidden read-only workers are available to
Investigate and that their reports return to the parent without mutation
prompts. Repeat the bounded repository task through Pi and verify its tool
calls and result replay before testing the matching MTP preset. Record quality,
protocol, or state-corruption failures instead of promoting either candidate
to a default.

For a host with two matching GPUs, add these single-workload checks:

| Multi-GPU path | Check |
| --- | --- |
| Doctor | select both render nodes; verify both named devices, one architecture, and a passed tensor operation on each |
| llama.cpp | run one server with both render nodes; verify managed layer split, activity on both cards, successful generation, and enough per-card headroom for KV/runtime buffers |
| ComfyUI component placement | run one graph with model, CLIP, or VAE deliberately placed on different cards; verify both activity and sane output |
| ComfyUI CFG split | run one graph with `MultiGPU CFG Split`; verify both activity and sane output without claiming pooled VRAM |

## Precision and policy coverage

The representative gate above establishes application and task behavior. Add
these comparisons before making performance or memory recommendations:

| Comparison | RX 9060 family | R9700 or RX 9070 family | Strix Halo | Strix Point |
| --- | --- | --- | --- | --- |
| one Qwen Image FP8/BF16 pair with the same workflow | pending | pending | pending | pending |
| one Wan FP8/FP16 pair with the same workflow | pending | pending | pending | pending |
| one LTX-2 FP8/BF16 full-model pair | pending | pending | pending | pending |
| Wan T2V Seko V1.1 versus V2.0 with identical inputs | pending | pending | pending | pending |
| balanced versus conservative memory policy | pending | pending | pending | pending |
| default versus experimental kernel policy | pending | pending | pending | pending |
| persistent versus isolated benchmark cache | pending | pending | pending | pending |

Use the same image, catalog commit, workflow, prompt, input, dimensions,
frames, seed, and render-node set for each comparison. Performance conclusions
require repeated measurements, not merely successful inference.

## Per-run result

For a manual inference result that is not already captured by managed
benchmark JSON, record:

```text
Status:
Date:
Git commit:
Application image ID:
Catalog bundle and source revisions:
Profile and render-node set:
Runtime policies:
Prompt/input identity:
Dimensions, frames, steps, and seed:
Peak memory:
Prompt/generation timing:
Output sanity:
Warnings or deviations:
Result/log location:
```

## Deferred acceptance handoff

When one contributor cannot reach every required hardware class, leave a
bounded, reproducible handoff instead of a generic request to “test on another
GPU.” Keep hostnames and personal infrastructure out of this document; record
the architecture and observable requirements.

```text
Change under test:
Source commit and application image ID:
Already accepted on:
Deferred architecture/profile:
Behavior or patch path requiring coverage:
Required managed content:
Exact commands:
Expected success criteria:
Result and log destination:
Known warnings or capacity constraints:
```

Commands must name the profile, render-node set, backend, preset, context,
cache policy, prompt and generation sizes, and repetition count whenever those
values affect the conclusion. The person completing the handoff should append
the observed result to the original change or its review record and retain
the generated JSON rather than returning only “works for me.”

The first publication may call uncompleted rows experimental, but it must not
present them as accepted. A later dependency, runtime-policy, ROCm/PyTorch, or
model change invalidates only the coupled rows; repeat those rows on every
hardware class where practical.
