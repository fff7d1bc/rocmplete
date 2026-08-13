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
| Fedora Linux 44 (non-OSTree), Ryzen AI Max+ 395, 128 GB LPDDR5X-8000 | Strix Halo, `gfx1151` | DwarfStar DeepSeek V4 Flash at 4K and 128K context; Qwen3.6 MTP tool protocol, OMP probe, and fixed ROCm/Vulkan llama.cpp benchmarks; Muse Glimmer agent probes; Laguna XS and Ling feasibility controls |
| Ubuntu 26.04, Ryzen AI Max+ 395, 128 GB LPDDR5X-8000 | Strix Halo, `gfx1151` | DwarfStar DeepSeek V4 Flash and the managed Qwen3.6 llama.cpp presets |
| SteamOS 3.8, Radeon RX 9070 XT 16 GB | RDNA 4, `gfx1201` | ComfyUI and the Qwen3 0.6B llama.cpp smoke |

### Fedora 44 Strix Halo observation (2026-08-09)

This field observation used kernel `7.1.7-200.fc44.x86_64`, profile
`strix-halo`, and `/dev/dri/renderD128`. The final bounded DwarfStar smoke used
source commit `a333de2`, DwarfStar image
`localhost/rocmplete:dwarfstar-ubuntu26.04-rocm7.14-d250a7c-r4` (image ID
`531fdef3be07c78825c573a3b2ceb333ab6b47245158f65894150fe3a54e902d`),
and the managed DeepSeek V4 Flash 0731 Q2 imatrix bundle. The llama.cpp tests
used image `localhost/rocmplete:llama-cpp-ubuntu26.04-rocm7.14-ddd4ec1-r14`
(image ID
`e3459ab8f0d7f96d3b846ebd90b29f8fd5187a7eb7689226cf69877c1ba8fd29`).

The host used the general 112 GiB TTM/GTT configuration:
`amdgpu.gttsize=114688`, `ttm.pages_limit=29360128`, and
`ttm.page_pool_size=29360128`. IOMMU remained enabled; the kernel command line
did not contain `amd_iommu=off`. Doctor passed the GPU operation and exact
device-isolation probes with enforcing SELinux and `container_use_devices`
enabled.

Observed results:

- The committed 4K DwarfStar smoke loaded 80.76 GiB of model spans in 18.106
  seconds, planned 81.18 GiB, passed its exact direct-answer check, and measured
  33.87 prompt tokens/s and 15.68 generated tokens/s. Its retained result is
  `apps/acceptance/results/20260809T191506Z-79592f44.json`, with the adjacent
  Markdown report.
- The managed 131072-context DwarfStar server planned 83.80 GiB. Direct-answer
  and normal-thinking requests passed, a two-turn continuation reused 78
  cached tokens, and a 640-token direct decode completed at 15.78 tokens/s.
  The receipt stayed current across runtime, explicit stop removed the
  container, and the kernel recorded no GPU reset, page fault, ring timeout,
  device-loss, or OOM event. This was a manual exploratory run, not a formal
  matrix `PASS`; the engine-reported memory plan was not an independently
  measured peak.
- The managed Qwen3.6 27B Q8_0 MTP server ran at its 262144-token context and
  completed nested tool-call/result round trips in both streaming and
  non-streaming modes. A fixed three-repetition pp512/tg128 comparison on the
  non-MTP model measured 343.40/7.82 tokens/s with ROCm and 254.60/7.84
  tokens/s with Vulkan. ROCm prompt processing was 34.9% faster; decode was
  effectively tied. Results are retained as
  `apps/llama-cpp/benchmarks/20260809T184312Z-f70ab4d7.json`,
  `20260809T184422Z-3a1edea5.json`, and
  `20260809T184422Z-backend-comparison-860c0bbf.json` below the same benchmark
  directory.
- A 2026-08-11 OMP 17.2.12 probe used the managed launcher and
  llama.cpp image `localhost/rocmplete:llama-cpp-ubuntu26.04-rocm7.14-62bf73d-r16`
  to load Qwen3.6 35B-A3B MTP at 262144 context.
  OMP called its `read` tool inside the default bubblewrap boundary, replayed
  the 318-line `AGENTS.md` result, and returned the exact requested heading.
  The first turn processed 21,358 prompt tokens at 708.34 tokens/s and decoded
  158 tokens at 66.94 tokens/s, with 112 of 138 MTP proposals accepted. The
  cached follow-up processed 4,702 new tokens at 535.68 tokens/s and decoded
  44 at 64.43 tokens/s, with 33 of 39 proposals accepted. Neither turn was
  truncated. A separate managed DwarfStar pass through the same OMP sandbox
  called `read`, replayed its result, and returned the same exact heading; its
  tool and final turns took 111.4 and 128.1 seconds. Both containers stopped
  cleanly. This verifies OMP model selection, tool replay, and sandbox behavior
  on two providers, not the complete agent matrix.
- A later manual Muse Glimmer run used source commit `9de3587`, llama.cpp
  commit `62bf73d25c53b8161f8a22894d4f90c4aebbd7d0`, image
  `localhost/rocmplete:llama-cpp-ubuntu26.04-rocm7.14-62bf73d-r15`, and the
  then-managed Unsloth Q8 target through preset
  `muse-glimmer-30b-ud-q8-k-xl-dflash` at 128K. OpenCode 1.18.15 completed a five-turn
  read-only repository task with structured grep and read calls, replayed
  their results, recovered from one lookup that found no files, and returned
  the requested catalog schema and llama.cpp pin exactly. DFlash accepted
  20.5% to 40.7% of proposed tokens across the turns, with reported generation
  rates of roughly 20 to 35 tokens/s. This accepts the representative
  OpenCode tool contract for those historical bytes at 128K.
- A 2026-08-10 comparison at source commit `6d59816` used the same llama.cpp
  commit and host to compare Meta's official dynamic K-quant, the former
  Unsloth Q8, and an Unsloth BF16 GGUF. Fixed pp512/tg128 results were
  341.17/10.32, 376.80/7.35, and 483.19/4.11 tokens/s respectively. Fresh
  128K DFlash servers used about 30.64 GB for K-quant and 66.77 GB for BF16.
  Maki completed broad, structured repository tasks with official K-quant at
  128K, Q8 at 128K and forced 256K, and BF16 at 128K. No run produced a GPU
  reset, OOM, or device loss. This selected K-quant for the catalog but did
  not mark the formal Muse matrix row `PASS`; immutable inputs, tool counts,
  fixture failures, and caveats are recorded in
  [the feasibility snapshot](muse-glimmer-llama-cpp-agent-feasibility.md).
- The candidate integration image
  `localhost/rocmplete:llama-cpp-ubuntu26.04-rocm7.14-62bf73d-r16` (image ID
  `15aa29c45b41f011f5edacd9f5fb761db26eae488d447453414e2d1b2a9e07a3`)
  subsequently loaded the verified official K-quant and DFlash files through
  the managed 128K preset. Direct startup included `--reasoning-preserve`, a
  bounded request returned exact content `OK`, and router startup accepted
  `reasoning-preserve = true` in all three Muse sections, loaded the 128K
  DFlash section on demand, and returned exact content `ROUTER_OK`. Both
  containers stopped cleanly. This verifies the new wiring, but remains a
  field observation rather than a formal matrix `PASS`.
- The 2026-08-12 Muse ATEM template correction was accepted with image
  `localhost/rocmplete:llama-cpp-ubuntu26.04-rocm7.14-62bf73d-r17` (image ID
  `98369219e680a5e44517ba1955a4fb3ce18fbcbf80cc3d89961e76648ddcb193`).
  Direct and routed 128K DFlash servers both received the pinned managed
  template and completed required structured tool calls; the direct path also
  completed a tool-result continuation. Pi 0.84.1 then solved the version 5
  `re-align` task in 597.8 seconds and 36 tool calls. Ordinary and hidden
  tests, the build, and dependency and artifact checks passed. The result is
  retained as
  `apps/agent-evaluation/results/20260812T152234Z-muse-glimmer-30b-kquant-dynamic-dflash.json`.
  This is a template-regression field observation, not a new comparative
  quality ranking or a formal matrix `PASS`.
- A 2026-08-13 follow-up on the same Fedora Strix Halo system and llama.cpp
  revision compared the official dynamic target with Meta's current 17 GB
  Q4_K_M target, and compared DFlash depths on both backends. Controlled 128K
  workloads selected depth 15 for ROCm and depth 4 for Vulkan. At a 64347-token
  prompt, the 17 GB target reached 27.35 generated tokens/s on ROCm and 27.23
  on Vulkan, versus 22.08 and 20.68 for the dynamic target. The 17 GB Pi probe
  solved `re-align` in 800.1 seconds and 65 tool calls with the same patch hash
  as dynamic; all ordinary and hidden tests, build, dependency, artifact, and
  network checks passed. Managed direct Vulkan and routed Vulkan startup both
  resolved depth 4 and returned exact bounded content, while the Pi path
  independently exercised ROCm depth 15. Containers stopped cleanly and the
  kernel journal showed no matching GPU fault. Exact inputs, the AMD reference
  caveats, complete matrix, and cleanup warning are in the
  [Muse feasibility record](muse-glimmer-llama-cpp-agent-feasibility.md).
- A 2026-08-12 KAT-Coder template audit used the same Fedora Strix Halo host,
  llama.cpp commit, KAT Q8_0 artifact, Pi 0.84.1, ROCm backend, 131072-token
  evaluation context, and high thinking. Candidate image
  `localhost/rocmplete:llama-cpp-ubuntu26.04-rocm7.14-62bf73d-r18` had image
  ID `98b81f0ab4b1cf948de4564e3c01bd722acc360a1c25538380e1b9f9361f0a04`.
  Its managed template matched Kwaipilot base revision
  `3a7d874090df0cd4399401982eca67df2c5a7e82` byte for byte and rendered a
  non-leading system message that the template embedded in the unchanged
  GGUF rejects. A direct server completed a required structured tool call and
  tool-result continuation. The final router policy loaded KAT on demand with
  the exact managed template, without `--reasoning-preserve`, then completed
  another required tool call and continuation before stopping cleanly. The
  same image's Qwen3 0.6B override rendered array-form text content and
  returned exact bounded content `TEMPLATE_OK`.

  An exploratory KAT `re-align` comparison separated the embedded template,
  updated template, and updated template with reasoning preservation. All
  three attempts produced the same patch SHA-256
  `cea8b4bc7fc8ba2cfd5c7952bf22b2a15fa1fafe37c186eabda7bca0fed1e215`
  and passed ordinary tests, hidden tests, build, dependency, artifact, and
  network checks. Their wall times and generated-token counts were 684.8
  seconds/23,221, 458.5/15,493, and 349.0/10,224 respectively. The companion
  `fz-eintr` attempts took 193.0 seconds with the embedded template, 110.5
  with the updated template, and 194.6 with the updated template plus
  preservation. Only the embedded-template attempt remained valid; both
  updated-template attempts ran `go build ./...` and left a forbidden `fzr`
  executable even though all code tests passed. Since KAT sampling remained
  stochastic and the upstream diff affects only non-leading system messages,
  the timing spread is not attributed to the template. The compatibility fix
  is retained, but reasoning preservation is not enabled and no quality or
  speed promotion is claimed. Raw results are retained as
  `apps/agent-evaluation/results/kat-template-ab-*.json` and
  `apps/agent-evaluation/results/kat-template-fz-*.json`.
- A 2026-08-11 Laguna XS 2.1 probe used the same pinned llama.cpp commit and
  Fedora Strix Halo host with Poolside's official Q4_K_M GGUF. Under the
  project's Strix policy, Flash Attention off, F16 K/V cache, batch 2048, and
  microbatch 512, a fixed three-repetition pp512/tg128 run measured
  885.81/60.59 tokens/s. The raw API returned exact arithmetic, a valid
  structured tool call, and a correct tool-result continuation. A fresh 256K
  allocation used about 58.42 GB of container memory and left about 68 GiB
  available, but did not fill the window with a 256K prompt. A read-only Pi
  task sustained valid tool loops through roughly 30K context and compaction;
  an overly broad review prompt did not converge before manual interruption.
  The candidate catalog then reused and rehashed the retained 18.88 GiB file
  through the normal mirror installer. Direct managed startup completed a
  structured tool-call/result round trip. Router startup exposed all 16
  installed presets and loaded Laguna XS on demand with `--jinja`,
  `--reasoning-preserve`, 262144 context, Flash Attention off, and
  `--load-mode none`; its bounded tool request was also valid. Both startup
  modes warned that `special_eos_id` and `special_eot_id` were absent from the
  special EOG set, but the accepted requests terminated normally. Revisit the
  warning if longer runs show premature or missing termination. Both
  containers stopped and were removed. No run produced an OOM, GPU reset,
  device loss, or kernel fault.
- The same investigation rejected Ling 3.0 Flash on ROCm. Atomic Q4 and Q5
  GGUFs were coherent through CPU and Vulkan, while the patched TurboQuant
  ROCm path returned corrupted text and malformed tool arguments on `gfx1151`.
  Exact inputs, backend controls, benchmarks, and retest criteria are in the
  [Ling feasibility snapshot](ling-3.0-flash-llama-cpp-feasibility.md).
- The 2026-08-13 Qwen27 tuning acceptance used the candidate bytes subsequently
  committed as `9fa54a5` and image
  `localhost/rocmplete:llama-cpp-ubuntu26.04-rocm7.14-62bf73d-r19` (image ID
  `d4b7065b465a85efbfc5ff0aa10895283bdc5e79b2aae5b528f9f1b6e9647147`).
  Direct and router ROCm startup at the default 262144 context both resolved
  MTP depth three, Flash Attention on, and symmetric Q8_0 target K/V while
  leaving the draft cache F16. Exact bounded requests passed through both
  paths. A 131072-context Vulkan control also returned exact content and
  accepted 114 of 117 MTP proposals. Pi 0.84.1 then solved version 5
  `re-align` with high thinking in 451.4 seconds. Pi, ordinary tests, hidden
  tests, and the build exited zero; only `probe.go` and `reencode_test.go`
  changed, with no dependency, generated artifact, or network violation. The
  server processed 21,125 prompt tokens at 187.13 tokens/s and generated 7,169
  tokens at 19.16 tokens/s. The retained result is
  `apps/agent-evaluation/results/qwen27-mtp3-q8-kv-re-align.json`. Containers
  stopped cleanly and the kernel recorded no GPU fault. Qwen35-A3B and other
  hardware profiles remain unchanged. Full controls and caveats are in the
  [Qwen tuning snapshot](qwen3.6-strix-halo-llama-cpp-tuning-feasibility.md).

#### Coding-agent comparison (2026-08-11)

The [model quality baseline](coding-agent-model-quality.md) groups these
results by demonstrated coding quality. This section retains the exact
measurement and artifact record.

This comparison used source commit
`ad5a2ce730a63f4a145c23a7af44f55a9971fc92`, frozen suite
`rocmplete-coding-v4` with fingerprint
`8825f9235c854fdf693c4881faa035c5efe99a545f4111372ca95e6f2def1160`,
Pi 0.84.1, ROCm, high thinking, 131072 context, and one fresh fixture and
session per task. The runtime was
`localhost/rocmplete:llama-cpp-ubuntu26.04-rocm7.14-62bf73d-r16` (image ID
`15aa29c45b41f011f5edacd9f5fb761db26eae488d447453414e2d1b2a9e07a3`)
on the Fedora 44 Strix Halo host described above. Each preset retained its
reviewed model-specific sampling policy. This tests the practical managed
configuration, not identical sampling across unrelated models.

The first gate used the same easy implementation task, `re-align`. `Solved`
means Pi exited normally, ordinary and hidden tests passed, the project built,
and the attempt had no dependency change, retained build artifact, or network
attempt. Tool calls are counted from Pi's structured transcript. Prompt and
generation rates are aggregate server metrics for the complete attempt.

| Managed preset | Outcome | Wall time | Tool calls | Input | Cache read | Output | Prompt tok/s | Generate tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `qwen3.6-35b-a3b-mtp-ud-q8-k-xl` | solved | 229.9 s | 23 | 17,752 | 394,740 | 12,596 | 111.30 | 70.03 |
| `kat-coder-v2.5-dev-q8-0` | solved | 319.3 s | 25 | 18,020 | 199,431 | 12,321 | 493.52 | 41.42 |
| `ornith-1.0-35b-q8-0` | solved | 484.2 s | 18 | 52,055 | 603,758 | 15,049 | 358.97 | 37.34 |
| `muse-glimmer-30b-kquant-dynamic-dflash` | solved | 670.2 s | 40 | 18,566 | 958,355 | 14,080 | 161.33 | 18.69 |
| `qwen3.6-27b-mtp-q8-0` | solved | 766.7 s | 22 | 42,722 | 814,547 | 8,699 | 48.43 | 15.57 |
| `gemma4-31b-it-q8-0-mtp` | solved | 902.5 s | 17 | 55,580 | 595,333 | 6,389 | 70.87 | 11.38 |

The exact managed artifacts and retained result records were:

- Qwen3.6 35B-A3B MTP, 39,099,447,584 bytes, SHA-256
  `6c6b816537abad90b250a0972b345466028d861ddfe316d5f0de31ca6440f781`;
  result `apps/agent-evaluation/results/20260811T154638Z-qwen3.6-35b-a3b-mtp-ud-q8-k-xl.json`.
- KAT Coder 2.5 Dev Q8_0, 36,914,690,464 bytes, SHA-256
  `5fa510f44779b0e3d38a6678985f417a1c65e3000405ca5d6dcf7fd065e47a15`;
  result `apps/agent-evaluation/results/20260811T175021Z-kat-coder-v2.5-dev-q8-0.json`.
- Ornith 1.0 35B Q8_0, 36,903,138,880 bytes, SHA-256
  `cbc992bca07901c1a51f33e65e6fc5d687de179c852a772dfd15e4c3261dbf5c`;
  result `apps/agent-evaluation/results/20260811T164947Z-ornith-1.0-35b-q8-0.json`.
- Muse Glimmer official dynamic K-quant, 19,653,957,984 bytes, SHA-256
  `513109c8319115f69eb09fb7b118c97c8167d15bc014fd7670d2e30489bf106c`,
  with its 1,631,205,312-byte DFlash draft, SHA-256
  `27d9a805fa29b943cfb6ad4843367cd4eaaaf06bd452d8cc3e00a2cd18a677bc`;
  result `apps/agent-evaluation/results/20260811T175621Z-muse-glimmer-30b-kquant-dynamic-dflash.json`.
- Qwen3.6 27B MTP Q8_0, 29,047,084,160 bytes, SHA-256
  `9408dcb356cc061a05c139e5647cbde0698ff980c6a69f7fc214e9989f86cfa8`;
  result `apps/agent-evaluation/results/20260811T173701Z-qwen3.6-27b-mtp-q8-0.json`.
- Gemma 4 31B Q8_0, 32,635,676,896 bytes, SHA-256
  `fcd52cebacb165a98df5abe6fb70dbf076835f4a06e064ffb33dd739b8835c9c`,
  with its 514,687,104-byte MTP draft, SHA-256
  `6b52ab20af503aee320dc09e93f886133b18d89ffc9075c7d9dcaf681e20b375`;
  result `apps/agent-evaluation/results/20260811T165827Z-gemma4-31b-it-q8-0-mtp.json`.

All six completed the easy task correctly, so this screen rejects obvious
agent-loop failures but does not establish broad correctness. Qwen3.6 35B-A3B
MTP was fastest end to end. KAT Coder was the closest alternate at 1.39 times
the wall time. Ornith was correct and used fewer tool calls, but read much
more context and took 2.11 times as long. Muse's DFlash decode did not offset
40 tool calls and repeated spacing probes. Qwen3.6 27B MTP and Gemma 4 were
correct but took 3.34 and 3.93 times the reference wall time. These are
single-repetition observations, not estimates of variance.

Laguna XS 2.1 used its exact 20,274,300,032-byte Q4_K_M artifact, SHA-256
`1ac7079101fca5a6df8c5a7523a3c30ea7d1c0e4b1258090e7d6d4039287f6cb`,
under the same version 4 conditions. It identified the relevant formatter but
made no edit in more than 21 minutes, repeatedly recalculated the same width
boundary, and fell to about 5 generated tokens/s after reading roughly 20K
context. The run was interrupted and cleaned up. Its checkpoint is
`apps/agent-evaluation/results/20260811T171415Z-laguna-xs-2.1-q4-k-m.json`.
Older version 2 screens are not part of the table: Laguna S 2.1 was
interrupted after about 27 minutes without an edit, and DwarfStar DeepSeek V4
Flash derived the likely fix but entered a repetitive loop before applying
it. Their exact checkpoints are
`20260811T135443Z-laguna-s-2.1-q4-k-m.json` and
`20260811T142629Z-deepseek-v4-flash.json` in the same result directory.
Those runs showed working runtime and Pi tool protocols, but not useful coding
completion.

A 2026-08-12 Laguna S control used source commit
`74477f25cca6b3491e74e33dec72a06bef0309d5`, frozen suite fingerprint
`9da456c1820080d032896fe0e69fafbf3722addc39008068ba62daff84b5aad7`, Pi
0.84.1, ROCm, high thinking, 131072 context, and the current 45-minute
per-attempt ceiling. The preset did not preserve interleaved reasoning and
advertised no reasoning-effort support, so Pi's requested level was ignored.
The server warned about the missing preservation setting. Result:
`apps/agent-evaluation/results/v5-45m-laguna-s-re-align.json`. Retain this as a
misconfigured control rather than a final model-quality result.

Commit `a2256eb948c3bf74bd83cfd17156a47e991603ff` enabled reasoning
preservation and the project's 1024, 4096, and 8192-token effort mapping for
Laguna S. Two corrected trials used the same suite, Pi version, backend,
context, image, and model bytes. The 68,248,760,064-byte Q4_K_M artifact had
SHA-256
`a34c74e46688122bef83122f4133031bababbefcf57436dde97048c91e2cc6ff`.
The runtime was image
`localhost/rocmplete:llama-cpp-ubuntu26.04-rocm7.14-62bf73d-r16`, image ID
`15aa29c45b41f011f5edacd9f5fb761db26eae488d447453414e2d1b2a9e07a3`.

The first corrected trial was manually interrupted at about 40 minutes while
Laguna was beginning an edit tool call after deriving the correct width-10
change. Its checkpoint is
`apps/agent-evaluation/results/v5-45m-laguna-s-preserved-re-align.json`; do not
score it as a pass or failure. The fresh trial received the complete 45-minute
ceiling. It made 11 completed read or shell calls, correctly identified width
10, and reproducibly stopped a high-effort reasoning turn at 8,329 decoded
tokens, but never edited. Retained history reached about 29K tokens, and
generation declined from 16.53 tokens per second initially to 5.65 on the
bounded turn and 3.93 near timeout. The final 3,906-token reasoning turn was
unfinished when the timeout checkpointed the attempt and removed the
container. Result:
`apps/agent-evaluation/results/v5-45m-laguna-s-preserved-re-align-r2.json`.
The hard gate was not run.

This accepts the corrected runtime and client wiring, not Laguna S as an
autonomous coding choice on the tested host. Faster decoding could let the
near-edit trajectory finish, but does not address the excessive reasoning
token count. The experiment does not isolate Q4 quantization, and the official
Q8_0 artifact cannot fit this 128 GiB host together with runtime state.

Only Qwen3.6 35B-A3B MTP completed all six implementation and two review
tasks under the final version 4 rules. The run took 2,046.8 seconds and used
319,303 input, 7,627,558 cached, and 76,966 output tokens. Its strict
implementation solve rate was 3/6:

| Task | Result | Durable finding |
| --- | --- | --- |
| `re-align` | solved | correct shared-width change |
| `fz-eintr` | solved | correct exact EINTR retry behavior |
| `fz-symlink` | solved | correct symlink identity and lazy metadata behavior |
| `re-cancel` | unsolved | all tests and build passed, but a generated `reencode` executable invalidated the attempt |
| `fz-sort-cancel` | unsolved | hidden behavior failed because cancellation was checked only before and after the blocking stable sort |
| `re-source-race` | unsolved | hidden behavior failed because output publication preceded source quarantine and quarantine did not reverify source identity |
| `review-reencode-lifecycle` | review pending | useful review, but it overstated cancellation safety after validation |
| `review-fzr-concurrency` | review pending | contained a material version-order error and missed that Enter can select from an older accepted snapshot before refresh |

The version 4 conclusion was therefore narrower than "Qwen is trustworthy."
Qwen3.6 35B-A3B MTP is the best-supported and fastest managed coding choice
on this host. The reference still failed both hard safety implementations and
made a material factual error in one review. The version 5 follow-up below
supersedes the earlier recommendation to promote KAT and Ornith directly to a
full-suite run.

#### Version 5 coding follow-up (2026-08-12)

The follow-up used source commit
`d62ef8cae0a49bed6cab1b2ee85ecd3b720c4d08`, frozen suite
`rocmplete-coding-v5` with fingerprint
`9da456c1820080d032896fe0e69fafbf3722addc39008068ba62daff84b5aad7`,
and the same host, image ID, Pi 0.84.1, ROCm backend, high thinking, 131072
context, and reviewed per-model sampling policies as version 4.

Qwen3.6 35B-A3B MTP completed all eleven tasks in 52 minutes 28 seconds
elapsed. The agent attempts accounted for 3,105.2 seconds, 286 tool calls,
513,182 input tokens, 11,205,961 cached tokens, and 114,087 output tokens. Its
implementation solve rate remained **3/9**: `re-align`, `fz-eintr`, and
`fz-symlink` passed. The retained result is
`apps/agent-evaluation/results/v5-qwen35-full.json`.

The six implementation failures were bounded completions, not loops:

- `re-cancel` passed ordinary and hidden tests but retained a generated
  executable.
- `fz-sort-cancel` passed ordinary tests, but its changed sorting helper was
  incompatible with the withheld cancellation contract.
- `re-source-race` passed ordinary tests, but its snapshot and quarantine
  design was incompatible with the withheld replacement-race contract.
- `proxy-late-probe` failed to close a successful connection that returned
  after timeout.
- `rc-selinux-verify` invoked recursive host relabeling at the wrong boundary,
  broke 14 ordinary tests, and missed the required per-file no-dereference
  command.
- `nonet-lifecycle` passed ordinary tests, but its changed wait-helper contract
  was incompatible with the withheld lifecycle test.

Human review also rejected both read-only answers as fully trustworthy. The
reencode answer mislocated core lifecycle functions and overstated
cancellation safety after validation. The fzr answer incorrectly said an
older entry snapshot is discarded; the picker accepts it and schedules a
follow-up, which also made the answer's Enter-selection explanation
inconsistent.

The five other easy-screen winners then received only `re-source-race`, with
a 20-minute practical ceiling. A remote SIGINT produced a durable interrupted
checkpoint and clean container removal on timeout.

```bash
timeout --foreground --signal=INT --kill-after=90s 20m \
  ./rocmplete benchmark agent --preset PRESET --task re-source-race
```

- Gemma4 31B MTP completed in 1,000.2 seconds and 27 tool calls. Ordinary tests
  and the build passed, no generated artifact remained, and the hidden
  contract failed. Result: `v5-gemma4-mtp-source-race.json`.
- Ornith 1.0 35B completed in 1,000.6 seconds and 54 tool calls. Ordinary tests
  passed, the hidden contract failed, and a generated executable remained.
  Result: `v5-ornith-source-race.json`.
- KAT-Coder v2.5 Dev completed in 1,139.7 seconds and 90 tool calls. Ordinary
  tests passed, the hidden contract failed, and a generated executable
  remained. Result: `v5-kat-source-race.json`.
- Muse Glimmer 30B DFlash reached ordinary tests but had not exited after 20
  minutes and 67 tool calls. Result: `v5-muse-dflash-source-race.json`.
- Qwen3.6 27B MTP had a substantial patch but had not reached tests after 20
  minutes and 31 tool calls. Result: `v5-qwen27-mtp-source-race.json`.

An extended capability probe then isolated those two interrupted models on
the same task. It used source commit
`75cd17ce43b8ded05d2e6ec096b452f927ae724a`, the unchanged suite fingerprint,
Pi 0.84.1, ROCm backend, high thinking, 131072 context, and the same model
presets and grading contract. The only policy change was a 60-minute outer
ceiling:

```bash
timeout --foreground --signal=INT --kill-after=90s 60m \
  ./rocmplete benchmark agent --preset PRESET --task re-source-race
```

- Qwen3.6 27B MTP exited normally after 1,674.5 seconds and 49 tool calls.
  Ordinary tests and the build passed, while the hidden suite failed to
  compile because `snapshotSource` was absent and the verification and
  quarantine contracts were incompatible. Live context peaked at 98,943
  tokens without compaction. Result:
  `v5-capability-qwen27-mtp-source-race.json`.
- Muse Glimmer 30B DFlash exited normally after 1,929.2 seconds and 82 tool
  calls. Ordinary tests and the build passed, while the hidden suite failed to
  compile because `snapshotSource` was absent and `quarantineSource` had an
  incompatible signature. Live context peaked at 80,558 tokens without
  compaction. Result: `v5-capability-muse-dflash-source-race.json`.

The longer runs show that both models can eventually produce a coherent,
ordinary-test-passing candidate. Neither solved the withheld source-safety
contract, neither approached the context limit, and neither qualifies for a
full-suite promotion. The 20-minute records remain the practical-runtime
comparison; these 60-minute records answer only the narrower capability
question.

This calibration sets the operator ceiling for future coding-model evaluation
at 45 minutes per attempt. It is not an aggregate suite timeout and does not
change the historical policies recorded above. A clearly repetitive loop may
still be interrupted earlier. A progressing timeout blocks promotion on the
tested host but remains inconclusive about model capability; retain its
checkpoint and revisit it when hardware or runtime throughput changes
materially.

All result paths are below `apps/agent-evaluation/results/`. No challenger
passed the hard gate, so none consumed a full-suite run. Laguna and DwarfStar
were not repeated because their unchanged earlier runs had already established
non-convergence. The installed non-MTP Qwen variants were also not repeated:
the version 4 comparison had selected the MTP variants as its practical
configurations, and the older non-MTP Qwen 27B easy run was slower than its
MTP counterpart. This follow-up filled missing suite coverage rather than
reopening the MTP-versus-non-MTP comparison.

For a newly integrated model, preserve the suite and host inputs and use the
same promotion sequence:

```bash
./rocmplete benchmark agent --preset NEW_PRESET --task re-align
./rocmplete benchmark agent --preset NEW_PRESET --task re-source-race
./rocmplete benchmark agent --preset NEW_PRESET
```

The first command permits a direct comparison with the version 4 table and
the version 5 Qwen baseline. The safety screen prevents an easy formatting
fix from qualifying a model by itself.
Run the complete suite only after both bounded tasks converge, then repeat a
finalist three times before changing the recommended default. Compare results
only when suite fingerprint, host, image, backend, context, harness, thinking,
and repetition count agree. A changed llama.cpp image or model-specific
sampling policy remains useful practical evidence, but must be called out as
a changed runtime rather than folded silently into this baseline.

Two host-specific failures were also useful. The first shared SELinux mount of
the newly installed DwarfStar file normalized its label after verification,
changing ctime and making the durable receipt stale. Commit `a333de2` now
applies the complete shared container label before hashing; a subsequent mount
left the receipt current. Also, a detached rootless container launched from an
SSH-only session received SIGINT and exited with status 130 when that host's
last login-scoped user manager stopped. Keeping the session active through the
explicit stop passed; a persistent remote service needs the host's user-manager
lingering policy handled separately. Neither failure was a GPU reset or model
inference crash.

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
| llama.cpp Laguna XS coding/agent | `llama-laguna-xs-2.1-q4-k-m`, 256K allocation and smaller deep task | N/P unless model and context fit the card | N/P unless host memory is deliberately used | pending | pending |
| llama.cpp coding/agent | `llama-laguna-s-2.1-q4-k-m`, 256K context | N/P unless model and context fit the card | N/P unless host memory is deliberately used for offload | pending | pending |
| llama.cpp Muse DFlash | dynamic target against no draft at 128K, ROCm depth 15 or Vulkan depth 4, advanced 17 GB target, then forced 256K beyond 128K | N/P unless model and context fit the card | N/P unless host memory is deliberately used for offload | pending | pending |
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

Laguna S remains experimental even after basic startup until chat templating,
tool use, sustained generation, and output sanity are accepted. Laguna XS has
useful Strix Halo field evidence, but still needs formal row acceptance and
coverage on the other applicable hardware profiles.

For the Qwen tool-protocol row, start the managed router and inspect `/props`
for the expected template capabilities and context. Send a developer message
and a tool schema with a nested object, require a structured tool call, return
the result as a tool message, and require a final answer. Repeat the exchange
with streaming enabled. If the MTP preset fails, repeat it with
`qwen3.6-27b-q8-0` to separate template handling from speculative decoding.
Only after the raw API exchange passes should OpenCode, Pi, OMP, and Maki tasks be
used as the final integration checks.

For the 35B-A3B agent-evaluation row, start the non-MTP preset first and leave
OpenCode's edit, shell, and subagent approvals enabled. Repeat a focused
Investigate-mode repository question, a delegated local investigation, a
bounded delegated web investigation, a raw nested-tool round trip, and the
concurrent nonce corruption probe before allowing a disposable edit task.
Confirm that only the two hidden read-only workers are available to
Investigate and that their reports return to the parent without mutation
prompts. Repeat the bounded repository task through Pi, OMP, and Maki and
verify their tool calls and result replay before testing the matching MTP preset.
Record quality, protocol, or state-corruption failures instead of promoting
either candidate to a default.

For the Muse row, use the 128K DFlash preset for a complete managed OpenCode
tool loop, then repeat the task with the non-speculative control when behavior
or output is suspect. Exercise the forced-256K preset with prompts extending
beyond 128K and inspect retrieval quality, tool selection, draft acceptance,
latency, and memory rather than treating successful startup as acceptance.
Maki has completed substantial live repository tasks and is the strongest
validated scaffold for the official target. OpenCode, Pi, and OMP remain exposed
through the same reviewed function-tool contract, but depth is sensitive to
their scaffold and prompt. Complete the intended live task in each client
before allowing unattended writes. Confirm `reasoning-preserve = true` in
router mode or `--reasoning-preserve` in direct-server startup; do not confuse
that history policy with a client-selectable reasoning-effort budget.

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
