# Applications

Every application follows the same basic path:

```text
build  ->  content install  ->  run
```

Use the [README quick start](../README.md#quick-start) when you just want one
working setup. This page is for choosing between the applications and
understanding the bits that differ once they are running.

For a shorter walkthrough made of copyable commands, use the built-in guide:

```bash
./rocmplete guide
./rocmplete guide comfyui
./rocmplete guide llama-cpp
./rocmplete guide dwarfstar
```

## ComfyUI

ComfyUI is the flexible option. Use it for graph workflows, the broadest
ROCmplete model selection, and imported models or workflows.

Prepare and run the default image-generation recipe:

```bash
./rocmplete build comfyui
./rocmplete content install comfyui image
./rocmplete run comfyui
```

The image includes pinned ComfyUI-GGUF, rgthree-comfy, and the exact Manager
version required by the pinned ComfyUI source. These copies move with the
locally built image, so a routine ROCmplete update does not depend on a custom
node updating itself at startup. Enable Manager through ComfyUI's upstream
flag after the argument separator:

```bash
./rocmplete run comfyui -- --enable-manager
./rocmplete run comfyui \
  --listen 192.168.1.50 \
  -- --enable-manager
```

Arguments before `--` belong to ROCmplete; arguments after it are passed to
ComfyUI. ROCmplete rejects forwarded versions of options it owns, including
listen addresses, ports, data directories, and `--cpu`.

The image remains read-only. Manager stores custom-node source under
`apps/comfyui/custom_nodes/` and installs its Python dependencies into the
separate persistent environment at `apps/comfyui/custom-node-python/`. That
environment can override an image package for ComfyUI runs, but cannot modify
the pinned environment inside the image. Custom nodes still execute arbitrary
third-party code, so review what Manager proposes before installing it.

A persistent custom node with the same directory name as a bundled node takes
precedence. The startup summary reports it under `persistent overrides`.
This keeps an existing Manager-installed copy usable and avoids loading two
copies of the same node. Remove or relocate that persistent directory when
you want the image-pinned copy to take over again.

Manager decides whether installation is safe from the host address ROCmplete
publishes, rather than ComfyUI's unavoidable wildcard bind inside the private
container network. Registered custom-node installation is available when the
host publication is loopback. A non-loopback listener can run already
installed nodes, but Manager keeps software installation disabled because
ROCmplete does not add authentication.

For remote administration, keep the default loopback publication and carry it
through SSH:

```bash
# On the GPU host:
./rocmplete run comfyui -- --enable-manager

# On the workstation where the browser runs:
ssh -N -L 8188:127.0.0.1:8188 gpu-host.local
```

Then open `http://127.0.0.1:8188`. Stop ComfyUI before switching between this
loopback administration run and a deliberately exposed non-loopback run.

Other ComfyUI options can be forwarded in the same way:

```bash
./rocmplete run comfyui --profile rdna4 -- --lowvram
./rocmplete run comfyui --disable-bundled-extensions
```

`--disable-bundled-extensions` disables both ComfyUI-GGUF and rgthree-comfy.
It does not disable nodes installed persistently through Manager.

To expose two cards to one ComfyUI process, select both explicitly:

```bash
./rocmplete run comfyui \
  --render-node /dev/dri/renderD128 \
  --render-node /dev/dri/renderD129
```

Simply exposing both cards does not change an existing graph. Use ComfyUI's
built-in nodes under `Advanced > multigpu` to make the graph place work:

- `Select Model Device`, `Select CLIP Device`, and `Select VAE Device` put
  those components on chosen cards. Splitting components this way is the
  useful option when the complete workflow is too large for one card.
- `MultiGPU CFG Split` distributes CFG work across cards. Add it after LoRA
  loaders and other model-changing nodes so every worker sees the final model.
  It clones the model to each selected card, so it is a throughput tool rather
  than pooled VRAM for one oversized diffusion model.

Use only cards with the same supported architecture in one process. ROCmplete
checks that condition at startup.

See [Content](content.md#comfyui-content) for curated and imported workflows,
model families, and exact bundle installation.

## llama.cpp

Use llama.cpp for local text models, an OpenAI-compatible API, or an
interactive terminal session.

For a useful general assistant:

```bash
./rocmplete build llama-cpp
./rocmplete content install llama-cpp qwen3.6
./rocmplete run llama-cpp server \
  --preset qwen3.6-27b-mtp-q8-0
```

The default preset is approximately 27.05 GiB with a 262144-token starting
context. It uses the dense Qwen3.6 27B MTP Q8_0 model. `--context` overrides
the catalog default.

All managed Qwen3.6 presets start at their native 256K. The pinned
[Qwen3.6 model card](https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF/blob/5cb35eb3dcbf52dbce5f87dbc64df6aaffadcace/README.md)
calls 256K the native context and recommends at least 128K to preserve thinking
capabilities. The value is a ceiling, not an amount of prompt text ROCmplete
feeds the model. llama.cpp prepares context capacity at startup, while request
work still follows the tokens actually sent.

Use 128K or 64K when memory matters more than the full native window:

```bash
# Smaller working set:
./rocmplete run llama-cpp server \
  --preset qwen3.6-27b-mtp-q8-0 \
  --context 65536

# Reduced but still substantial working set:
./rocmplete run llama-cpp server \
  --preset qwen3.6-27b-mtp-q8-0 \
  --context 131072
```

In router mode, `--context` overrides every loaded preset, not only the model
named by the next request. That is reasonable for a router containing only
256K agent presets. Omit it for a mixed router so TranslateGemma and other
bounded-task models retain their catalog-owned limits.

The generated agent-client model maps describe the 256K catalog default. Keep
the server at that default when using a launcher; a manual server override
does not silently rewrite client context metadata.

### Models, presets, and request settings

A model and a ROCmplete preset are related, but they are not interchangeable:

| Layer | What it owns | Examples |
| --- | --- | --- |
| GGUF model | Architecture, trained weights, and quantization | Qwen3.6 27B Q6_K, Q8_0, or Dynamic Q8_K_XL |
| ROCmplete preset | Model artifact, context size, and required model runtime policy | Chat template, Jinja, profile-specific Flash Attention, MTP or DFlash settings |
| Server launch | Machine and service policy for this run | ROCm or Vulkan, hardware profile, render nodes, listen address, port |
| API request or client | The current task and generation behavior | System message, conversation history, temperature, top-p, maximum tokens |

The API calls the router selector `model`, but its value is a ROCmplete preset
identifier. It selects both the GGUF and the runtime policy ROCmplete has
validated for it. The `translategemma-27b-it-q8-0` preset adds only a thin string
template around the model. The translation direction and output rules remain
part of the user message.

### Choosing a managed Qwen preset

Start with `qwen3.6-27b-mtp-q8-0` when you do not yet have workload results. It
is the smaller general baseline. For agent work on a high-memory host,
start with `qwen3.6-35b-a3b-mtp-ud-q8-k-xl`; keep the dense model as a smaller
general assistant. Install either non-MTP control when you need to isolate the
effect of speculative decoding:

| Preset | What changes | Good use |
| --- | --- | --- |
| `qwen3-0.6b-q8-0` | Tiny 0.6B model | Startup, API, and GPU-offload smoke tests |
| `qwen3.6-27b-mtp-q8-0` | Dense 27B MTP Q8_0 | General assistant and smaller baseline |
| `qwen3.6-27b-q8-0` | Dense 27B Q8_0 | Non-MTP control for the dense model |
| `qwen3.6-35b-a3b-ud-q8-k-xl` | Sparse 35B-A3B Dynamic Q8_K_XL | Non-MTP control for the recommended agent model |
| `qwen3.6-35b-a3b-mtp-ud-q8-k-xl` | Matching sparse model with MTP heads | Recommended agent-client default on a high-memory host |

Keep whichever model succeeds on representative tasks rather than choosing
from the parameter count or quantization name alone.

### Ornith and KAT-Coder

Ornith and KAT-Coder have separate family-oriented recipes, so either Q8_0 35B
MoE model can be installed independently on a high-memory host:

```bash
./rocmplete content install llama-cpp ornith
./rocmplete content install llama-cpp kat-coder
```

| Preset | Source | Current role |
| --- | --- | --- |
| `ornith-1.0-35b-q8-0` | DeepReinforce's official Ornith 1.0 GGUF | Agentic-coding candidate |
| `kat-coder-v2.5-dev-q8-0` | Bartowski's plain Q8_0 conversion of Kwaipilot's public text-only checkpoint | Newer agentic-coding candidate kept separate from APEX and MTP derivatives |

Both presets use their embedded tool-aware Jinja template and native 256K
context. They appear in both managed client model pickers after installation, but
neither replaces ROCmplete's Qwen3.6 default. Treat benchmark claims as leads,
then compare tool-call correctness, task completion, repetition, wall time,
and recovery from long sessions on the same repositories. ROCmplete does not
catalog the community APEX mixed-precision or grafted-MTP variants because
the Q8 baselines fit the maintained high-memory target and provide a clearer
quality control.

MTP is a decoding optimization, not a reasoning mode or a more capable model.
Its prediction heads propose several future tokens and the main model verifies
them. Accepted drafts can reduce generation time; poor acceptance can erase
the gain. MTP does not accelerate prompt ingestion, and the client's
`--thinking` switch is independent of it.

For an MTP comparison, use the server API and measure complete response wall
time, output validity, and draft acceptance on the same prompts. The native
`benchmark llama-cpp` path deliberately rejects managed speculative presets
because `llama-bench` does not apply their MTP or DFlash policy. A matching
non-speculative preset remains the useful control.

See the launch policy and catalog footprint for every installed managed
preset:

```bash
./rocmplete content list --models --details
```

ROCmplete presets currently do not store a general system prompt, persona, or
sampling policy. Those belong in the caller because they can change between
tasks without changing the model's safe runtime setup. Send a `system` message
and sampling fields in the OpenAI-compatible request, and send the complete
message history when the conversation should continue:

```json
{
  "model": "qwen3.6-27b-mtp-q8-0",
  "messages": [
    {
      "role": "system",
      "content": "You translate technical Japanese into concise English."
    },
    {
      "role": "user",
      "content": "Translate this text..."
    }
  ],
  "temperature": 0.2,
  "max_tokens": 1024
}
```

Use an OpenAI-compatible API client or a small task-specific harness for
automation, system messages, sampling controls, and conversation history. A
new managed preset makes sense when a model requires stable context, template,
Jinja, Flash Attention, or speculative-decoding policy. A reusable writing
persona or summarization instruction does not need a catalog preset. Do not
add a hardware profile for task tuning either. In ROCmplete, profiles describe
GPUs such as Strix Halo or RDNA 4.

The managed OpenCode and Pi launchers are coding-task callers, so their
generated model entries apply reviewed model-family sampling defaults. That
does not change direct API requests, terminal mode, or the server defaults.

For a quick human check without an API client, run the model directly in a
terminal:

```bash
./rocmplete run llama-cpp cli --preset qwen3.6-27b-mtp-q8-0
```

CLI mode owns an interactive conversation in that terminal. API callers must
send the complete message history when they want a conversation to continue.

List the exact managed variants and their launch policy before installing a
large model:

```bash
./rocmplete content list --models --details

# Install every managed llama.cpp model in one resumable operation.
./rocmplete content install llama-cpp all

# Or install the tiny startup and GPU-offload smoke test.
./rocmplete content install llama-qwen3-0.6b-q8-0
./rocmplete run llama-cpp server --preset qwen3-0.6b-q8-0
```

The Gemma 4 preset starts at its native 256K and is maintained for agent use:

```bash
./rocmplete content install llama-gemma4-31b-it-q8-0-mtp
./rocmplete run llama-cpp server --preset gemma4-31b-it-q8-0-mtp
```

Use `--context 131072` or `--context 65536` when the full KV cache leaves too
little memory for the model, runtime buffers, or other workloads. Its
official embedded template handles developer instructions, structured tool
calls, tool results, and reasoning turns through llama.cpp's chat-template
path. It appears in both generated client model maps and supports the same
reasoning choices as the managed Qwen agents.

### Tool-using clients

Managed Qwen, Gemma 4, and Muse Glimmer agent presets enable llama.cpp's Jinja
engine and use the chat template embedded in their pinned GGUF. This is
required for structured OpenAI-style tool calls. The pinned Unsloth Qwen3.6
templates include their developer-role and tool-calling fixes. Gemma 4 uses
Google's canonical tool-calling template, and Muse retains Meta's tool and
reasoning framing. ROCmplete does not replace them with a generic one.

The Qwen3 0.6B preset follows the same protocol and is useful for a cheap API
smoke test, but it is too small to treat as a dependable repository agent.
Template support means a client and server can exchange tool calls; it does
not make every Qwen variant equally capable of choosing and using tools.

Agents normally send a model name with every request, so the managed router is
the least ambiguous launch mode:

```bash
./rocmplete run llama-cpp server --router --models-max 1
```

ROCmplete configures but does not install the clients. Install an upstream
client by its supported method. For Linuxbrew or Homebrew:

```bash
brew install opencode
brew install pi-coding-agent
```

Maki is distributed separately. Put its `maki` executable on `PATH` before
using the ROCmplete launcher.

ROCmplete ships PATH-friendly OpenCode, Pi, and Maki launchers. They render the
current provider and model catalog every time they start. Install the
recommended model and start the router first, then choose a client:

```bash
./rocmplete content install llama-cpp qwen3.6
./rocmplete run llama-cpp server --router --models-max 1
export PATH="$PWD/bin:$PATH"
opencode
# or: pi
# or: maki
```

The generated provider lists every preset explicitly maintained for agent
work, including ones not currently installed. Selecting an absent model does
not download it; the router reports that it is unavailable. Use the client's
model picker to choose a different installed model. OpenCode accepts `-m
rocmplete/PRESET`. Pi accepts `--model PRESET`, with `--provider rocmplete`
available when the provider would otherwise be ambiguous. Maki accepts `-m
rocmplete/PRESET` and exposes the same entries through `/model`.

OpenCode and Pi use these llama.cpp request defaults for coding turns:

| Model family | Temperature | Top-p | Top-k | Min-p | Presence penalty | Repeat penalty |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6 27B and 35B-A3B | 0.6 | 0.95 | 20 | 0 | 0 | 1 |
| Ornith 1.0 35B | 1.0 | 0.95 | 20 | 0 | 0 | 1 |
| KAT-Coder V2.5 Dev | 1.0 | 0.95 | 20 | 0 | 1.5 | 1 |
| Gemma 4 31B IT | 1.0 | 0.95 | 64 | 0 | 0 | 1 |
| Laguna S 2.1 | 1.0 | 1.0 | 20 | 0 | 0 | 1 |
| Muse Glimmer 30B | 1.0 | 0.95 | 64 | 0 | 0 | 1 |

The sources are Qwen's precise-coding recommendation for
[Qwen3.6 27B](https://huggingface.co/Qwen/Qwen3.6-27B/blob/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9/README.md)
and
[35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/995ad96eacd98c81ed38be0c5b274b04031597b0/README.md),
Ornith's
[generation configuration](https://huggingface.co/deepreinforce-ai/Ornith-1.0-35B/blob/5df2ed3f675c7beaa490328cc70bb573b65fb660/generation_config.json),
KAT-Coder's
[agent API example](https://huggingface.co/Kwaipilot/KAT-Coder-V2.5-Dev/blob/7be56fe773e72b6f5ca93c1ae45d828ddb893922/README.md),
Gemma 4's
[standardized sampling guidance](https://huggingface.co/google/gemma-4-31B-it/blob/842da3794eaa0b77d5f08bae87a17459d91ff475/README.md),
Laguna's
[generation configuration](https://huggingface.co/poolside/Laguna-S-2.1/blob/00af5a51782109b587a3b3bbf11875e566036fa7/generation_config.json),
and Muse Glimmer's
[model card](https://huggingface.co/meta-models/Muse-Glimmer-30B/blob/f84ecc3a0ea984a4c04542a84269e3d065350a6e/README.md).
Neutral values make parameters omitted by an upstream configuration explicit
and disable llama.cpp's otherwise nonzero min-p default. The upstream name
`repetition_penalty` is sent to llama.cpp as `repeat_penalty`.

These are client defaults, not locks. OpenCode model options or an agent's
provider options and Pi per-request sampling parameters can override them.
OpenCode's managed Investigate agents still force temperature zero. The tested
Maki 0.4.5 dynamic-provider schema cannot express per-model sampling fields,
so managed Maki sessions currently inherit llama.cpp's sampler defaults;
ROCmplete does not emit fields that Maki would silently ignore.

`bin/opencode` delegates to `./rocmplete agent opencode`, injects the
generated main configuration directly into the child process, and points it
at the read-only TUI keymap in the checkout. It writes no integration files.
Pulling a catalog or policy update therefore changes the next launch without
an install step.
The recommended MTP preset is the default when present. If it is absent, the
launcher selects another installed agent-capable preset; `-m
rocmplete/PRESET` still overrides that selection through OpenCode itself.
Project OpenCode configuration continues to load around the runtime settings.
The user's normal global OpenCode configuration and state are hidden by the
default sandbox.

Informational and installation-management commands retain their normal
upstream behavior through the PATH launcher. The commands `opencode --help`,
`opencode --version`, `opencode completion`, `opencode plugin`, `opencode
upgrade`, and `opencode uninstall` bypass ROCmplete's model configuration and
sandbox. A plugin installed this way remains disabled by the default sandbox's
`--pure` policy; use the documented `--no-sandbox` launch only when loading
that host-side plugin is intentional.

`bin/pi` delegates to `./rocmplete agent pi` and writes only the generated
`models.json` inside Pi's ROCmplete-owned private state. The file uses Pi's
`openai-completions` provider, lists the same reviewed llama.cpp presets and
DwarfStar model, and is refreshed atomically on every launch. Pi's normal
`~/.pi/agent` state is not read or modified. The launcher disables Pi's update
checks and telemetry during managed sessions. It declines project `.pi`
resources by default, while ordinary `AGENTS.md` context still loads.

Pi's package commands keep their upstream shape through the PATH launcher.
For example, `pi install npm:pi-code-indexer`, `pi list`, and `pi update
--extensions` act on Pi's private ROCmplete-owned state, without requiring an
installed model or a running server. The upstream `pi update` command updates
Pi itself by default, so self, `--self`, and `--all` updates bypass the sandbox
and operate on the real installation. The positional aliases are `self` and
`pi`. An explicit package command may use the network. Installed user packages
and their extensions, skills, prompts, and themes are available on later
managed launches. They are trusted executable inputs with access to the
writable project and host network inside the sandbox, so review them before
installation. A local `pi install -l` still requires explicit project approval
before its project resources can load.

`bin/maki` delegates to `./rocmplete agent maki`. It atomically refreshes two
executable provider descriptions and a small generated `init.lua` inside
Maki's ROCmplete-owned XDG directories. The providers inherit Maki's native
llama.cpp Chat Completions adapter and publish the exact context, output, and
thinking capabilities of the reviewed presets. Maki's normal global config,
sessions, and model choices are not read or modified.

The recommended installed model starts at medium thinking. Maki remembers an
explicit `/model` or `/thinking` choice in its private state. On the first
launch, ROCmplete assigns the selected default to Maki's strong, medium, weak,
and compaction tiers so a local subagent does not silently select another
alphabetically sorted model. Later tier changes in `/model` are preserved.
An unchanged generated assignment follows the default if installed content
changes. The generated config also limits Maki to one concurrent task subagent
because several simultaneous 256K local sessions can exhaust shared GPU
memory.

The PATH launchers require bubblewrap and start sandboxed by default. Their
writable host paths are the launch directory and private per-client state
below `DATA_DIR/apps/CLIENT/sandbox`. The launch directory keeps its absolute
path inside the sandbox so sessions retain stable project paths. Fedora's
`/home` alias to `/var/home` is preserved, so either spelling of an absolute
project path resolves consistently. Private state keeps sessions and client
preferences useful across launches without exposing ordinary host-side
history. The real home directory, SSH agent, credentials, Podman state, and
GPU devices are not mounted. A sanitized Git author and committer identity is
passed without mounting `.gitconfig`.

On Ubuntu, AppArmor may restrict unprivileged user namespaces to executables
with matching profiles. The distribution bubblewrap package may be covered
while a Linuxbrew or other custom build is not. `./rocmplete doctor` reports
the active kernel policy and, when restricted, prints a persistent opt-out.
That opt-out applies system-wide and reduces protection against kernel bugs
reachable through unprivileged user namespaces, so Doctor says so alongside
the command.

System files and the resolved client installation are read-only. OpenCode's
TUI keymap is mounted as one read-only file at a synthetic path. The launchers
recognize Linuxbrew below `/home/linuxbrew/.linuxbrew` and mount that prefix
read-only.

Maki can load executable project configuration from `.maki/init.lua`. Treat it
like code from the checkout. The bubblewrap boundary still prevents that code
from reaching the rest of the home directory, but it retains access to the
writable project and host network.

Networking cannot be unshared because the managed router is normally on host
loopback. The sandbox can therefore still reach the Internet, LAN, and other
localhost services, and it can still alter or delete anything in the writable
launch directory. It is a practical damage boundary, not a VM or network
policy.

The launcher refuses `/`, the host home directory, or another ancestor of the
home as the writable scope. Start it from the repository you intend to expose.
Launching from a broader directory such as `~/src` is allowed but deliberately
exposes every project below that directory; the launcher prints the resolved
writable path before the client starts.

Use the direct command for the explicit escape hatch:

```bash
./rocmplete agent opencode --no-sandbox --
./rocmplete agent pi --no-sandbox --
./rocmplete agent maki --no-sandbox --
```

This restores ordinary host filesystem access and should be reserved for a
toolchain or linked worktree that cannot operate inside the narrow mount set.
It still uses ROCmplete's generated provider catalog and private client state.

OpenCode's generated config leaves reads and searches automatic. Build and Plan ask
before file edits, shell commands, and subagent launches. This keeps an
explanation or review request from silently turning into implementation if a
local model loses the task during a long session or after context compaction.
Approval is still a trust decision: OpenCode's auto-approve mode approves
`ask` actions, and a higher-precedence project config can override the
generated policy. The sandbox remains useful even when an in-application
policy is weakened because that policy cannot add host mounts.

New sessions start in ROCmplete's Investigate agent. Press `Tab` to cycle
through Investigate, Plan, then Build; `Shift+Tab` cycles in reverse.
Investigate is a hard read-only primary agent for focused repository and
external research. It cannot edit, run shell commands, or create todos. Small
questions stay in the main session. For broader work it may invoke only two
hidden ROCmplete workers: one can read and search the
current repository but has no web access, while the other can research the
web but has no local-file access. Both independently deny edits, commands,
todos, and further delegation. Each receives a separate child-session context
and is instructed to return at most 500 words, so raw files and fetched pages
do not consume the main session unless the worker ignores its output bound.

Investigate's temperature is zero and it has no artificial step limit because
OpenCode's limit injects a forced summary, remaining-tasks list, and next-step
recommendations when reached. Its prompt instead requires focused searches,
evidence, explicit inference, and a stop after answering the original
question. These are containment and focus controls, not a guarantee that
every factual conclusion from a local model is correct. A delegated report is
evidence for the primary agent to reconcile, not automatically a reliable
fact.

Pi uses its standard coding-agent tool loop rather than ROCmplete's OpenCode
agent modes. Project `.pi` resources are declined by default so a checkout
cannot silently add executable extensions or settings. That is an input guard,
not an authorization prompt for model tool calls. Pi has no built-in sandbox,
so the bubblewrap boundary is the control that limits filesystem damage. Pass
`--approve` only when project Pi resources are intentionally trusted.

Maki starts in Build mode. Press `Tab` to toggle its built-in Plan mode, where
only the plan file may be written, then return to Build when the plan is ready.
Its `--print` mode always starts a new Build-mode session, so use the TUI when
the Plan boundary matters.

For long-context presets, ROCmplete advertises a 16K per-turn output ceiling to
all three clients. That leaves more of the native context available before
automatic compaction while still accommodating the managed 8K high reasoning
budget and a final response. This limit is per model turn, not the total
session.

Presets with reviewed bounded reasoning expose disabled, low, medium, and high
choices. OpenCode names the disabled choice `instant`; Pi and Maki name it
`off`. OpenCode uses `ctrl+t` or `/variants`, while Pi uses `Shift+Tab`, the
thinking selector in `/settings`, or the startup `--thinking` option. Pi's
`/model` changes the model rather than opening a separate reasoning selector.
Maki uses `/thinking`. The disabled choice sends `reasoning_effort: none`. The
other three are real llama.cpp thinking ceilings of 1024, 4096, and 8192
tokens, not just UI labels.
ROCmplete uses medium until a per-model choice takes precedence. A preset
without reviewed budget support does not advertise reasoning choices.

All three clients use `/v1/chat/completions` and expose their file and shell
tools as ordinary function calls. That matches llama.cpp's current tool
adapter. Still test a complete read, edit, command, and tool-result loop in
each maintained client before letting a newly added model work unattended.

Use the ROCmplete preset ID as the API model ID. Configure the client with the
preset's actual starting context rather than advertising a larger limit. For
example, `qwen3.6-27b-mtp-q8-0` starts at 262144 tokens. Once a router
model is loaded, inspect the template llama.cpp recognized:

```bash
curl -sS --get http://127.0.0.1:8080/props \
  --data-urlencode 'model=qwen3.6-27b-mtp-q8-0' |
  jq '{template_caps: .chat_template_caps, slots: .total_slots,
       context: .default_generation_settings.n_ctx}'
```

Test one complete tool round trip before giving an agent write access: request
a tool call, return its result as a tool message, and require a final answer.
Then test nested object arguments and streaming. A successful ordinary chat
response is not evidence that the structured tool-call path works.

### Japanese and English translation

There are three focused translation models. Shisa V2.1 is the quality-first
Japanese and English choice for a high-memory host. This is a 69.83 GiB Q8_0
quant of the Llama 3.3 70B model:

```bash
./rocmplete content install llama-cpp shisa-v2.1 --accept-license
./rocmplete run llama-cpp server \
  --preset shisa-v2.1-llama3.3-70b-q8-0
```

The preset starts at 16384 tokens. That is enough for a glossary, speaker and
scene notes, several nearby lines, and the requested translation without
committing a large KV cache to every server run. Raise it with `--context`
only after measuring the complete workload. The model supports ordinary
`system`, `user`, and `assistant` messages through its embedded Jinja template.
For a visual-novel translation request, put stable rules and the glossary in a
system message, then give each user message the source text plus only the
local context needed to disambiguate it.

Start deterministic evaluation at temperature zero. If the literal result is
correct but stiff, compare a small nonzero value such as `0.2` against the same
fixed test set. Keep names, control codes, terminology, and output shape under
validation rather than relying on sampling to preserve them. The installer
requires acknowledgment of the Llama 3.3 Community License Agreement.

TranslateGemma is the smaller 27B IT Q8_0 choice for constrained, manually
prompted translation:

```bash
./rocmplete content install llama-cpp translation-gemma --accept-license
./rocmplete run llama-cpp server \
  --preset translategemma-27b-it-q8-0
```

The preset supplies Gemma's turn markers but does not choose a language
direction or hide a translation instruction. Put the direction, output rules,
and source text together in the user message:

```bash
curl -sS http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "Translate the following English text into Japanese. Produce only the Japanese translation, without commentary.\n\nThis is the text to translate."
      }
    ],
    "temperature": 0,
    "max_tokens": 512
  }' | jq -r '.choices[0].message.content'
```

Use the same running server for Japanese-to-English or another supported
language pair by changing that first instruction. TranslateGemma accepts only
`user` and `assistant` roles, so keep the instruction in the user message
rather than sending a separate `system` message. HTTP requests do not inherit
earlier messages. Repeat the instruction in each standalone request, or resend
the complete conversation beginning with that first user message.

Google documents a 2K-token input limit for TranslateGemma. ROCmplete starts
the server with 4096 tokens so the translation has room to finish, but that
does not make longer source text supported. Split long documents at sensible
boundaries.

For a large translation job, use one running server and a bounded pool of
independent HTTP requests. The current llama.cpp server chooses its number of
slots automatically and enables continuous batching. Check what the running
model received before choosing a client worker count:

```bash
curl -sS http://127.0.0.1:8080/props |
  jq '{slots: .total_slots, context: .default_generation_settings.n_ctx}'
```

Start by comparing one, two, and four requests in flight on a representative
set of source lines. More requests can improve total throughput by batching
generation across sequences, but each individual request may take longer and
additional slots consume KV-cache capacity. Measure total generated tokens
divided by the whole job's wall time, along with lines per second and request
latency. Do not add the per-request token rates together.

Give every source item an ID, accept responses in any order, validate empty or
truncated output, and restore source order when writing the result. Repeat the
translation rules, relevant glossary, and a small scene context in each
request. Do not build one ever-growing conversation merely to give independent
lines context. ROCmplete does not currently expose an explicit server slot
setting, so keep the worker count configurable and measure it again after a
llama.cpp, backend, model, or context change.

Tencent HY-MT1.5 is the smaller 7.43 GiB multilingual option. Its prompt
chooses the target language at request time:

```bash
./rocmplete content install llama-cpp translation-hy --accept-license
./rocmplete run llama-cpp server --preset hy-mt1.5-7b-q8-0

curl -sS http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "Translate the following segment into Japanese, without additional explanation.\n\nThis is the text to translate."
      }
    ],
    "temperature": 0.7,
    "top_k": 20,
    "top_p": 0.6,
    "repeat_penalty": 1.05,
    "max_tokens": 512
  }' | jq -r '.choices[0].message.content'
```

The Tencent HY Community License excludes the EU, United Kingdom, and South
Korea from its territory and says use outside that territory is unauthorized.
The installer shows the complete license link and requires
`--accept-license`, but that switch is an acknowledgment, not a way around the
restriction. In Poland, use TranslateGemma instead.

### Backend selection

ROCm is the default llama.cpp backend. The same locally built image also has
Vulkan, so a server, terminal session, or benchmark can select it explicitly:

```bash
./rocmplete run llama-cpp cli \
  --preset qwen3.6-27b-q8-0 \
  --backend vulkan
```

If you do not know which one to use, run them back to back:

```bash
./rocmplete benchmark llama-cpp \
  --preset qwen3.6-27b-q8-0 \
  --compare-backends
```

The comparison runs unattended, shows the prompt and generation winners
separately, and estimates the combined time for the selected pp/tg ratio. Use
the winning backend with `run llama-cpp ... --backend`.

This is useful for measuring the machine in front of you, not a claim that one
backend always wins. Benchmark every preset you plan to use instead of carrying
one result across a model family. Sparse Q4_K_XL can favor Vulkan while dense
Q6_K favors ROCm on the same machine. Quantization, dense or
mixture-of-experts layout, and active parameter count all change the work each
backend sees.

Keep the model, context, profile, and render-node set the same when comparing
them. The switch belongs to llama.cpp. ComfyUI uses PyTorch and remains a
ROCm application.

### Multiple GPUs

llama.cpp can divide one model across an explicitly selected GPU set. Repeat
the option once per card:

```bash
./rocmplete run llama-cpp server \
  --model /path/to/large-model.gguf \
  --render-node /dev/dri/renderD128 \
  --render-node /dev/dri/renderD129
```

ROCmplete enables llama.cpp's compatible layer split automatically whenever
more than one card is selected. The same selection works with `cli`, the
managed router, and `benchmark llama-cpp`. It is still worth leaving capacity
on every card for the KV cache and runtime buffers. ROCmplete does not enable
the experimental tensor split or add an RCCL dependency.

### MTP models

The `-mtp` presets enable llama.cpp's `draft-mtp` speculative decoder from
catalog-owned settings. Qwen GGUFs contain their MTP heads; Gemma installs a
separate pinned draft. Performance and output behavior depend on the llama.cpp
revision, context, backend, and hardware. Use the server API for end-to-end MTP
measurements as described in the Qwen section.

### Muse Glimmer and DFlash

Muse Glimmer is a separate 30B model family rather than another Qwen variant.
The recipe installs [Meta's official dynamic K-quant target and DFlash
draft](https://huggingface.co/meta-models/Muse-Glimmer-30B-GGUF/blob/93769bc7ab5ad1e9cd22d857e3138cf5d977ae81/README.md),
approximately 19.82 GiB in total:

```bash
./rocmplete content install llama-cpp muse-glimmer
./rocmplete run llama-cpp server \
  --preset muse-glimmer-30b-kquant-dynamic-dflash
```

The DFlash preset is the recipe default. It starts at 131072 tokens and allows
up to fifteen draft tokens, matching the draft's 16-token block. The same
installed bundle also exposes
`muse-glimmer-30b-kquant-dynamic` as a non-speculative control. This makes it
possible to compare output, draft acceptance, wall time, and memory without
changing the target GGUF.

The bundle also exposes an experimental forced-window policy without another
download:

```bash
./rocmplete run llama-cpp server \
  --preset muse-glimmer-30b-kquant-dynamic-dflash-256k
```

That preset sets both `muse-glimmer.context_length` and
`dflash.context_length` to 262144 and disables llama.cpp automatic fitting.
Meta's pinned target and DFlash metadata declare 131072 tokens. Treat 256K as
forced extrapolation until retrieval, quality, memory, and draft acceptance
pass beyond 128K. The 128K DFlash preset therefore remains the recipe default.
`--context 0` is intentionally refused for the forced preset; a positive
override such as `--context 196608` applies to both target and draft metadata.

On one `gfx1151` host, a fixed pp512/tg128 ROCm benchmark measured the
official K-quant at 341.17 prompt and 10.32 generated tokens/s. The previously
managed Unsloth Q8 target measured 376.80 and 7.35 tokens/s, while a 51.90 GiB
BF16 conversion measured 483.19 and 4.11 tokens/s. The K-quant also started a
fresh 128K DFlash server in 14.0 seconds at about 30.64 GB of container memory,
versus 37.5 seconds and 66.77 GB for BF16. These are one-host observations,
not cross-hardware promises. The complete immutable inputs and caveats are in
the [maintainer feasibility record](../docs/muse-glimmer-llama-cpp-agent-feasibility.md).

The base, 128K DFlash, and forced-256K DFlash presets are advertised to the
managed OpenCode, Pi, and Maki clients. OpenCode and Pi use the pinned upstream
temperature 1.0, top-p 0.95, and top-k 64 defaults; Maki retains the server
sampler defaults because its dynamic-provider schema cannot express per-model
sampling. Live Maki tasks completed substantial repository reviews with the
official K-quant at 128K, the former Q8 target at both 128K and forced 256K,
and BF16 at 128K. An earlier short OpenCode answer was a completed shallow
turn, not a server crash: repository-review depth remains sensitive to the
client scaffold and prompt. Test the actual workflow before granting
unattended write access.

All three presets enable llama.cpp reasoning preservation so parsed reasoning
remains available to multi-turn history. This is separate from a bounded
reasoning-effort selector: Muse does not advertise the Qwen-style effort
variants. Meta's `Reasoning strength: high` guidance is still a prompt-level
instruction owned by the active agent or project scaffold.

ROCmplete previously installed
`Muse-Glimmer-30B-UD-Q8_K_XL.gguf` for this recipe. Upgrading the catalog does
not delete persistent model content. After the official replacement installs
and verifies successfully, remove the obsolete file manually if it is no
longer needed. The forced-256K entry remains available for pre-release field
testing, but its long-context quality and DFlash acceptance are experimental.

### Laguna S 2.1

Laguna S 2.1 is a 118B-A8B mixture-of-experts model intended for agentic
coding and long-horizon work. Its pinned official Q4_K_M file is approximately
63.56 GiB. It quantizes the routed experts with an importance matrix while
keeping the attention, shared experts, and embeddings at Q8_0. The official
Q8_0 file is about 129 GB before allocating a KV cache or runtime buffers, so
it is not a viable 128 GiB-host alternative. ROCmplete starts Laguna at its
recommended 262144 tokens, enables Jinja for chat and tool templates, and
disables Flash Attention on both RDNA 3.5 APU profiles
because that backend/model combination has conflicting early reports. RDNA4
retains llama.cpp's automatic Flash Attention policy. DFlash is not enabled
because the pinned upstream llama.cpp supports the base Laguna architecture
but not Poolside's fork-only DFlash path.

Treat the preset as experimental until it has been accepted on target
hardware. Use `--context 131072` or `--context 65536` if the native window
does not leave enough room for the KV cache, runtime buffers, and the host.
Poolside's GGUF repository points to the base OpenMDW-1.1 terms but does not
independently declare license
metadata for the conversion, so installation requires both flags shown below.

```bash
./rocmplete content install llama-cpp laguna-s-2.1 \
  --accept-license --acknowledge-license-risk
./rocmplete run llama-cpp server \
  --preset laguna-s-2.1-q4-k-m --profile strix-halo
```

Managed presets are text-only; ROCmplete does not install vision projectors.
For document work, extract text outside ROCmplete and send bounded chunks
through the API. Parsing, OCR, and retrieval are outside these bundles.

### Router mode and local GGUF files

Expose every completely installed preset through the multi-model router:

```bash
./rocmplete run llama-cpp server \
  --router --models-max 2
```

Select a preset using the OpenAI API `model` field. Missing models are skipped;
partial managed installs stop startup. The generated router INI and model
partition are mounted read-only.

An ordinary Chat Completions request selects the router preset through
`model`:

```bash
curl -sS http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.6-27b-mtp-q8-0",
    "messages": [{"role": "user", "content": "Explain unified memory."}],
    "max_tokens": 512
  }' | jq -r '.choices[0].message.content'
```

Any local regular `.gguf` file can also be used:

```bash
./rocmplete content list --models --scan /path/to/model-directory
./rocmplete build llama-cpp
./rocmplete run llama-cpp server --model /path/to/model.gguf

./rocmplete run llama-cpp cli \
  --model /path/to/model.gguf \
  --prompt 'Explain unified memory briefly.'
```

The model's parent directory is mounted read-only, supporting multi-file GGUF
shards. A CLI invocation with `--prompt` answers once and exits; omit it for
an interactive terminal conversation. The server exposes an OpenAI-compatible
API on port 8080. Remote model fetching and the mutable upstream embedded web
UI are disabled. Non-loopback publication has no authentication unless
`--api-key-file` names a readable key file.

On Strix Halo and Strix Point the entrypoint enables unified-memory policy and
non-mmap model loading. RDNA 4 discrete GPUs use normal automatic layer
placement.

## DwarfStar

DwarfStar is the deliberately narrow path for DeepSeek V4 Flash. ROCmplete
compiles the CLI, HTTP server, and benchmark binary locally from one pinned
`antirez/ds4` source commit against the same ROCm 7.14 runtime as the other
applications. It does not run upstream host setup scripts or use upstream
runtime binaries.

The first manually exercised workload is the 0731 IQ2XXS imatrix GGUF on a
128 GB Strix Halo. A 128 GB Strix Point host also completed a manual run at
about 3.9 generated tokens per second. That establishes feasibility, not
formal acceptance or a performance recommendation. The image contains HIP
code for `gfx1150`, `gfx1151`, `gfx1200`, and `gfx1201`, and the launcher
allows every corresponding ROCmplete profile. The exact 80.76 GiB model is
downloaded from a pinned Hugging Face revision, checked by size and SHA-256,
and mounted read-only. All architectures remain experimental until they
complete the hardware acceptance matrix. Multi-GPU execution, DSpark, MTP,
distributed execution, SSD streaming, and the native coding agent remain
outside this integration.

The regular `doctor` 112 GiB TTM/GTT result is the project's starting point
for a nominal 128 GB Strix Halo. Strix Point also needs enough TTM/GTT mapping
space for the resident model, context, and transient allocations. A discrete
GPU may spill through ROCm into system memory, but that does not make a small
VRAM host accepted or guarantee useful performance. During the manual Strix
Halo run, the managed 128K server and an OpenCode tool workflow passed at this
setting. At 100 GiB, a 32K server could start but a roughly 6K-token tool
prefill ran out of mapping space on a transient 320 MiB allocation. The bounded
acceptance smoke still uses a 4K context so routine checks remain short. The
[pinned upstream Strix Halo guide](https://github.com/antirez/ds4/blob/d250a7c07c6beb753e9b0a33951d8c00d6ef30ee/STRIXHALO.md)
uses `amdgpu.gttsize=126976`, `ttm.pages_limit=32505856`, and
`ttm.page_pool_size=32505856`, which is roughly a 124 GiB ceiling. It also uses
`amd_iommu=off`. The 112 GiB manual 128K run also used that setting. It can
improve this particular unified-memory workload but reduces DMA isolation, so
treat it as an explicit host security tradeoff, not an automatic ROCmplete
setting.

Build, install, and run it explicitly:

```bash
./rocmplete build dwarfstar
./rocmplete content install dwarfstar flash-0731-q2-imatrix
./rocmplete run dwarfstar server
```

The server listens on `127.0.0.1:8000` by default and exposes DwarfStar's
OpenAI-compatible API. The DwarfStar process binds to `0.0.0.0` inside its
private container namespace so Podman can forward the port, while Podman
publishes it only on the selected host address. The default publication is
therefore loopback-only, not LAN-accessible. Pass `--listen 0.0.0.0` or one
exact non-loopback host address only when unauthenticated network publication
is intentional.

Without `--model`, ROCmplete selects the installed and verified
`flash-0731-q2-imatrix` model. This is the 0731 chat-v2 imatrix GGUF with
IQ2_XXS routed gate/up weights, Q2_K routed down weights, and Q8 attention
projections, shared experts, and output. It is the upstream Q2 model intended
for 96/128 GB machines, not a uniformly IQ2_XXS fallback.

The pinned upstream Strix Halo guide selects this quantization layout and
warns that mixed Q2/Q4 builds can put enough pressure on the ROCm path to
trigger system OOM. ROCmplete therefore does not offer the larger mixed
Q2/Q4, Q4, MXFP4, or PRO models as managed alternatives on this hardware
class. DSpark is a separate experimental speculative-decoding aid, not a
higher-quality model, and is not part of the managed bundle.

A different DwarfStar-compatible local GGUF can be selected explicitly; its
containing directory is mounted read-only:

```bash
./rocmplete run dwarfstar server \
  --model /path/to/deepseek-v4.gguf
```

ROCmplete does not scan a directory and guess which file is compatible. The
managed starting point is 131072 context tokens with a 16000-token response
ceiling, following the upstream exercised server configuration rather than
assuming the model's largest possible context will fit beside an 80 GiB
resident model. Reduce the allocation when diagnosing memory pressure:

```bash
./rocmplete run dwarfstar server --context 32768
```

For one local prompt, CLI mode uses thinking by default. Disable it for a
short direct-answer check:

```bash
./rocmplete run dwarfstar cli --no-thinking \
  --prompt 'Reply with exactly: DwarfStar ready'
```

The API chooses thinking behavior per request. A `deepseek-chat` model alias,
`"think": false`, or `"thinking": {"type": "disabled"}` requests a direct
answer; DwarfStar otherwise defaults to its normal thinking mode. Keep the
server on loopback unless an authenticated trusted proxy protects it. An
explicit non-loopback publication has no built-in authentication.

Both agent launchers include DwarfStar as a separate provider. Start the
server, then choose it explicitly in either client:

```bash
./rocmplete run dwarfstar server
opencode -m dwarfstar/deepseek-v4-flash
pi --provider dwarfstar --model deepseek-v4-flash --thinking high
maki -m dwarfstar/deepseek-v4-flash
```

Its OpenCode variants are `instant` and `thinking`. Instant sends
`reasoning_effort: none`; thinking sends `reasoning_effort: high`, which is
DwarfStar's normal thinking mode. The engine maps low, medium, and high to the
same mode below its much larger Think Max context threshold, so ROCmplete does
not expose three misleading labels. It also suppresses OpenCode's inherited
`max` label because the managed 128K server cannot activate the 384K-minimum
Think Max mode. If the DwarfStar server uses another port, pass
`./rocmplete agent opencode --dwarfstar-port PORT --` or set
`ROCMLETE_OPENCODE_DWARFSTAR_PORT`. Pi exposes the same two behaviors as `off`
and `high`; pass `./rocmplete agent pi --dwarfstar-port PORT --` or set
`ROCMLETE_PI_DWARFSTAR_PORT` for its provider. Maki runs DwarfStar in its
normal server-side thinking mode and does not advertise a selector because
Maki's llama.cpp adapter sends a different budget field. Use the raw API when
a direct-answer DwarfStar request is required. Set
`ROCMLETE_MAKI_DWARFSTAR_PORT` when its server uses a different port.

Run the hardware-bound smoke separately after initial setup. Outside Strix
Halo, selecting DwarfStar explicitly is also the opt-in that prevents the
80.76 GiB model from joining an ordinary default acceptance run:

```bash
./rocmplete acceptance run --application dwarfstar --dry-run
./rocmplete acceptance run --application dwarfstar
```
