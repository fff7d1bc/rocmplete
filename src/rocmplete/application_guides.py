"""Short, copyable guides for each managed application."""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

from .config import APPLICATIONS
from .errors import LauncherError
from .recipes import application_recipes
from .ui import ColumnSpec, print_columns, style


@dataclass(frozen=True)
class GuideAction:
    command: str
    description: str


@dataclass(frozen=True)
class GuideSection:
    title: str
    paragraphs: Tuple[str, ...] = ()
    actions: Tuple[GuideAction, ...] = ()
    role: str = "label"


@dataclass(frozen=True)
class ApplicationGuide:
    application: str
    title: str
    summary: str
    sections: Tuple[GuideSection, ...]
    reference: str


def _action(command: str, description: str) -> GuideAction:
    return GuideAction(command, description)


def _recipe_actions(application: str) -> Tuple[GuideAction, ...]:
    return tuple(
        _action(
            "./rocmplete content install {} {}".format(
                application, recipe.identifier
            ),
            recipe.description,
        )
        for recipe in application_recipes(application)
    )


def _web_access_section(application: str) -> GuideSection:
    port = APPLICATIONS[application].port
    if port is None:
        raise ValueError("{} is not a web application".format(application))
    return GuideSection(
        "Open it",
        (
            "{} listens only on this machine by default. Open "
            "http://127.0.0.1:{} in a browser.".format("ComfyUI", port),
            "Publishing a non-loopback address has no built-in "
            "authentication.",
        ),
        (
            _action(
                "./rocmplete run {} --listen 192.168.1.50".format(
                    application
                ),
                "Replace the address with one assigned to the GPU host.",
            ),
            _action(
                "./rocmplete run {} --port {}".format(
                    application, port + 100
                ),
                "Change the host and application port together.",
            ),
        ),
        role="warning",
    )


def _comfyui_guide() -> ApplicationGuide:
    return ApplicationGuide(
        application="comfyui",
        title="ComfyUI",
        summary="Node graphs for image generation, editing, and video.",
        sections=(
            GuideSection(
                "Start here",
                (
                    "Build ComfyUI, install the practical Qwen Image recipe, "
                    "then start the application.",
                ),
                (
                    _action(
                        "./rocmplete doctor",
                        "Check host access, ROCm, PyTorch, and the GPU first.",
                    ),
                    _action(
                        "./rocmplete build comfyui",
                        "Build or refresh the ComfyUI application path.",
                    ),
                    _action(
                        "./rocmplete content install comfyui image",
                        "Install Qwen Image FP8 with the Lightning workflow.",
                    ),
                    _action(
                        "./rocmplete run comfyui",
                        "Start ComfyUI on the default loopback address.",
                    ),
                ),
                role="success",
            ),
            GuideSection(
                "Choose content",
                (
                    "Install another small runnable recipe when you want a "
                    "different workload.",
                ),
                _recipe_actions("comfyui"),
                role="info",
            ),
            _web_access_section("comfyui"),
            GuideSection(
                "Manager and custom nodes",
                (
                    "The image already includes pinned ComfyUI-GGUF and "
                    "rgthree-comfy.",
                    "Manager can install custom nodes on a loopback run. "
                    "Review third-party code before installing it.",
                ),
                (
                    _action(
                        "./rocmplete run comfyui -- --enable-manager",
                        "Enable the bundled Manager on the safe loopback run.",
                    ),
                ),
                role="warning",
            ),
            GuideSection(
                "Day-to-day",
                actions=(
                    _action(
                        "./rocmplete run comfyui --detach",
                        "Start ComfyUI in the background.",
                    ),
                    _action(
                        "./rocmplete logs comfyui --follow",
                        "Follow output from a detached ComfyUI run.",
                    ),
                    _action(
                        "./rocmplete stop comfyui",
                        "Stop and remove the managed ComfyUI container.",
                    ),
                    _action(
                        "./rocmplete run comfyui --help",
                        "Show every ComfyUI launcher option.",
                    ),
                ),
            ),
        ),
        reference="guide/applications.md#comfyui",
    )


def _llama_guide() -> ApplicationGuide:
    assistant = "qwen3.6-27b-mtp-q8-0"
    benchmark = "qwen3.6-27b-q8-0"
    return ApplicationGuide(
        application="llama-cpp",
        title="llama.cpp",
        summary="GGUF terminal chat and an OpenAI-compatible API.",
        sections=(
            GuideSection(
                "Start here",
                (
                    "Build llama.cpp and install the dense and sparse "
                    "Qwen3.6 MTP pair. The installer prints the command for "
                    "starting dense 27B MTP.",
                ),
                (
                    _action(
                        "./rocmplete doctor",
                        "Check GPU access and run the small diagnostic probe.",
                    ),
                    _action(
                        "./rocmplete build llama-cpp",
                        "Build or refresh the llama.cpp application path.",
                    ),
                    _action(
                        "./rocmplete content install llama-cpp qwen3.6",
                        "Install both practical Qwen3.6 MTP choices.",
                    ),
                    _action(
                        "./rocmplete content list --models --details",
                        "List managed models, installation state, and "
                        "runtime policy.",
                    ),
                ),
                role="success",
            ),
            GuideSection(
                "Choose content",
                (
                    "The installer shows model terms and regional "
                    "restrictions before downloading.",
                ),
                actions=_recipe_actions("llama-cpp"),
                role="info",
            ),
            GuideSection(
                "Preset versus model",
                (
                    "A model is the GGUF weight file. A ROCmplete preset "
                    "selects that model and adds its reviewed starting "
                    "context plus any required chat template, Jinja, Flash "
                    "Attention, or MTP policy.",
                    "Presets do not store a general system prompt, sampling "
                    "settings, GPU backend, or hardware profile. Put task "
                    "instructions and temperature in each API request or in "
                    "the client that owns the conversation.",
                    "ROCmplete's managed OpenCode, Pi, and OMP configurations "
                    "are coding clients, so they apply reviewed per-model "
                    "sampling defaults. Direct API requests, terminal mode, "
                    "and Maki keep their own request behavior.",
                    "The qwen3.6 recipe installs dense 27B MTP Q8_0 and sparse "
                    "35B-A3B MTP Dynamic Q8_K_XL together. Its printed next "
                    "step starts dense 27B MTP as the general baseline.",
                    "The separate ornith and kat-coder recipes install the "
                    "official Ornith 1.0 35B Q8_0 GGUF or the plain "
                    "KAT-Coder V2.5 Dev Q8_0 conversion. They are comparison "
                    "candidates, not replacement defaults.",
                    "For local agent work on a high-memory host, compare "
                    "alternative presets on real repository tasks before "
                    "choosing a default.",
                    "On Strix Halo, the sparse Qwen3.6 35B-A3B MTP "
                    "Q8_K_XL preset is the recommended OpenCode starting "
                    "point. Dense 27B MTP remains the smaller general "
                    "assistant and comparison point.",
                    "Matching dense and sparse non-MTP controls remain "
                    "available as exact advanced bundles.",
                    "MTP proposes and verifies extra tokens during decoding. "
                    "It may improve generation speed, but it is not a "
                    "reasoning mode and does not accelerate prompt ingestion.",
                    "Shisa V2.1 is the high-memory Japanese and English "
                    "translation choice. Its Llama 3.3 70B Q8_0 preset starts "
                    "at 16K so a bounded translation job leaves runtime "
                    "headroom.",
                    "TranslateGemma has one preset. It adds the required Gemma "
                    "turn markers but leaves the language direction and "
                    "translation rules in your user message.",
                ),
                role="info",
            ),
            GuideSection(
                "Install every managed model",
                (
                    "The explicit application aggregate installs every "
                    "cataloged llama.cpp bundle. It is intentionally absent "
                    "from the guided menu because the complete set is "
                    "hundreds of gigabytes.",
                    "The installer resumes partial downloads, skips verified "
                    "models, and collects terms and license acknowledgments "
                    "once for the complete plan.",
                ),
                (
                    _action(
                        "./rocmplete content install llama-cpp all --dry-run",
                        "Inspect the complete model and disk-space plan.",
                    ),
                    _action(
                        "./rocmplete content install llama-cpp all",
                        "Install every managed llama.cpp model.",
                    ),
                ),
                role="warning",
            ),
            GuideSection(
                "Talk in the terminal",
                (
                    "CLI mode keeps the conversation in one foreground "
                    "terminal session.",
                ),
                (
                    _action(
                        "./rocmplete run llama-cpp cli --preset {}".format(
                            assistant
                        ),
                        "Open an interactive conversation with the preset.",
                    ),
                ),
                role="info",
            ),
            GuideSection(
                "Serve one model",
                (
                    "A single-model server has already selected its preset, "
                    "so API requests may omit the model name.",
                    "Use an OpenAI-compatible client for automation or the "
                    "direct CLI mode above for a quick terminal check.",
                ),
                (
                    _action(
                        "./rocmplete run llama-cpp server --preset {}".format(
                            assistant
                        ),
                        "Start the API on http://127.0.0.1:8080.",
                    ),
                ),
                role="info",
            ),
            GuideSection(
                "Serve several presets",
                (
                    "Router mode exposes every complete managed preset. "
                    "The API model name is the ROCmplete preset name.",
                    "Qwen3.6, Ornith, KAT-Coder, Gemma 4, and Laguna start "
                    "at their native 256K context. Use --context 131072 "
                    "or --context 65536 for a smaller working set. In "
                    "router mode that option "
                    "overrides every loaded preset, so omit it for a "
                    "mixed-model router.",
                ),
                (
                    _action(
                        "./rocmplete run llama-cpp server "
                        "--router --models-max 2",
                        "Start the managed multi-model router.",
                    ),
                ),
                role="info",
            ),
            GuideSection(
                "Tool-using clients",
                (
                    "Managed Qwen, Ornith, KAT-Coder, and Gemma 4 agent "
                    "presets enable their "
                    "embedded Jinja templates for llama.cpp's structured "
                    "tool-call path. The Qwen templates include reviewed "
                    "developer-role fixes. Gemma uses Google's canonical "
                    "tool-calling template.",
                    "Use the ROCmplete preset ID as the router model ID and "
                    "configure the client with that preset's actual context "
                    "limit. Qwen3 0.6B can smoke-test the protocol but is "
                    "not a dependable repository agent.",
                    "ROCmplete's bin/opencode, bin/pi, bin/omp, and bin/maki "
                    "wrappers render the current server and model config at "
                    "launch without editing any client's normal settings. "
                    "Add the checkout's bin directory to PATH once, then "
                    "invoke a client normally. All use ordinary Chat "
                    "Completions function tools.",
                    "Pi package commands such as install, list, and update "
                    "keep their upstream shape and use Pi's private "
                    "ROCmplete state. Explicitly installed user packages "
                    "load in later sandboxed Pi sessions; review them as "
                    "trusted executable inputs.",
                    "OMP is a separate Pi fork with its own private state. "
                    "Its managed local roles, reviewed per-model sampling, "
                    "and yolo approval default can be overridden by normal "
                    "OMP session arguments. Named OMP profiles bypass that "
                    "state boundary and are rejected by the wrapper.",
                    "The PATH launchers use bubblewrap by default. Only the "
                    "launch directory and private ROCmplete-owned client "
                    "state are writable; the real home, credentials, SSH "
                    "agent, Podman state, and GPU devices are hidden. Host "
                    "networking remains available for the local router.",
                    "Build and Plan ask before edits, shell commands, and "
                    "subagent launches. OpenCode auto-approve bypasses those "
                    "prompts, so leave it off unless that is intended.",
                    "New sessions start in the Investigate agent. Press Tab "
                    "to cycle through Investigate, Plan, then Build; "
                    "Shift+Tab goes the other way. Investigate is hard "
                    "read-only but may "
                    "delegate bounded work only to hidden read-only local and "
                    "web workers. Their source material stays in separate "
                    "child sessions. Investigate also avoids OpenCode's "
                    "synthetic maximum-step continuation prompt.",
                    "Managed reasoning presets expose off or instant, low, "
                    "medium, and high thinking budgets. The disabled choice "
                    "turns thinking off; medium is the llama.cpp fallback. "
                    "OpenCode uses ctrl+t or /variants. Pi uses Shift+Tab "
                    "or /settings. OMP accepts --thinking. Maki uses "
                    "/thinking and Tab toggles its Plan and Build modes.",
                ),
                (
                    _action(
                        "./rocmplete content install llama-cpp qwen3.6",
                        "Install the dense and sparse MTP models.",
                    ),
                    _action(
                        "./rocmplete run llama-cpp server "
                        "--router --models-max 1",
                        "Start one managed model for an agent client.",
                    ),
                    _action(
                        "./rocmplete agent opencode",
                        "Start OpenCode directly; bin/opencode is the "
                        "PATH-friendly equivalent.",
                    ),
                    _action(
                        "./rocmplete agent pi",
                        "Start Pi directly; bin/pi is the PATH-friendly "
                        "equivalent.",
                    ),
                    _action(
                        "./rocmplete agent omp",
                        "Start OMP directly; bin/omp is the PATH-friendly "
                        "equivalent.",
                    ),
                    _action(
                        "./rocmplete agent maki",
                        "Start Maki directly; bin/maki is the PATH-friendly "
                        "equivalent.",
                    ),
                ),
                role="info",
            ),
            GuideSection(
                "Use your own GGUF",
                (
                    "External model directories are scanned only when you "
                    "name them.",
                ),
                (
                    _action(
                        "./rocmplete content list --models "
                        "--scan /path/to/ggufs",
                        "Find loose and split GGUF models in a known location.",
                    ),
                    _action(
                        "./rocmplete run llama-cpp cli "
                        "--model /path/to/model.gguf",
                        "Run one exact local model without creating a preset.",
                    ),
                ),
                role="info",
            ),
            GuideSection(
                "Choose a GPU backend",
                (
                    "ROCm is the default. Vulkan can be faster for some "
                    "models and slower for others.",
                ),
                (
                    _action(
                        "./rocmplete benchmark llama-cpp "
                        "--preset {} --compare-backends".format(benchmark),
                        "Measure both backends with the non-MTP control.",
                    ),
                ),
                role="info",
            ),
            GuideSection(
                "Network access",
                (
                    "The server defaults to loopback. A non-loopback address "
                    "needs an API key or another trusted access layer.",
                ),
                (
                    _action(
                        "./rocmplete run llama-cpp server "
                        "--preset {} --listen 192.168.1.50 "
                        "--api-key-file /path/to/key".format(assistant),
                        "Publish on one host address with a bearer API key.",
                    ),
                    _action(
                        "./rocmplete run llama-cpp server "
                        "--preset {} --port 8081".format(assistant),
                        "Change the host and server port together.",
                    ),
                ),
                role="warning",
            ),
            GuideSection(
                "Day-to-day",
                actions=(
                    _action(
                        "./rocmplete run llama-cpp server "
                        "--preset {} --detach".format(assistant),
                        "Start the single-model server in the background.",
                    ),
                    _action(
                        "./rocmplete logs llama-cpp --follow",
                        "Follow output from a detached server.",
                    ),
                    _action(
                        "./rocmplete stop llama-cpp",
                        "Stop and remove the managed llama.cpp container.",
                    ),
                    _action(
                        "./rocmplete run llama-cpp --help",
                        "Show server and CLI modes.",
                    ),
                ),
            ),
        ),
        reference="guide/applications.md#llamacpp",
    )


def _dwarfstar_guide() -> ApplicationGuide:
    return ApplicationGuide(
        application="dwarfstar",
        title="DwarfStar",
        summary=(
            "DeepSeek V4 Flash 0731 on a high-memory AMD GPU host, served "
            "by a narrow engine built for this model."
        ),
        sections=(
            GuideSection(
                "Start here",
                (
                    "DwarfStar is an experimental high-memory path. Its "
                    "image targets every ROCmplete GPU architecture, but "
                    "formal hardware acceptance remains pending. The 128 GB "
                    "Strix Halo path has been exercised manually. A 128 GB "
                    "Strix Point host also completed a manual run at roughly "
                    "3.9 generated tokens per second; that is a feasibility "
                    "observation, not formal acceptance. Build it from "
                    "pinned source and install the verified "
                    "80.76 GiB Q2 imatrix model. Its routed gate/up "
                    "weights use IQ2_XXS, routed down weights use Q2_K, "
                    "and attention projections, shared experts, and output "
                    "use Q8. The 112 GiB "
                    "shared-memory starting point supports the managed 128K "
                    "context in that manual run. The bounded smoke "
                    "still uses only a 4K context.",
                ),
                (
                    _action(
                        "./rocmplete doctor",
                        "Check GPU access and, on an APU, the available "
                        "GPU-mapped memory.",
                    ),
                    _action(
                        "./rocmplete build dwarfstar",
                        "Build the source-pinned multi-architecture image.",
                    ),
                    _action(
                        "./rocmplete content install dwarfstar "
                        "flash-0731-q2-imatrix",
                        "Install the verified DeepSeek V4 Flash 0731 model.",
                    ),
                    _action(
                        "./rocmplete run dwarfstar server",
                        "Start the API on http://127.0.0.1:8000.",
                    ),
                ),
                role="success",
            ),
            GuideSection(
                "Build and content policy",
                (
                    "ROCmplete does not use DwarfStar's host setup or model "
                    "download scripts. The engine is compiled locally from "
                    "one source commit and the GGUF uses the normal verified "
                    "content installer.",
                ),
                actions=(
                    _action(
                        "./rocmplete build dwarfstar",
                        "Build or refresh the source-built AMD GPU image.",
                    ),
                )
                + _recipe_actions("dwarfstar"),
                role="info",
            ),
            GuideSection(
                "Context and thinking",
                (
                    "Without --model, ROCmplete selects the installed, "
                    "verified flash-0731-q2-imatrix model. Pass one exact local "
                    "DwarfStar-compatible GGUF path to test another model; "
                    "its containing directory is mounted read-only.",
                    "The managed default is 131072 context tokens with a "
                    "16000-token output ceiling. Lower --context to reduce "
                    "memory pressure. The official model supports more, but "
                    "large contexts still need hardware acceptance.",
                    "Server clients select thinking behavior in each request. "
                    "CLI mode uses normal thinking by default; --no-thinking "
                    "is useful for deterministic direct-answer checks.",
                    "DSpark and optional MTP support are intentionally not "
                    "enabled while their ROCm paths remain unsettled.",
                ),
                (
                    _action(
                        "./rocmplete run dwarfstar server --model "
                        "/path/to/deepseek-v4.gguf",
                        "Serve one explicit compatible local GGUF.",
                    ),
                    _action(
                        "./rocmplete run dwarfstar cli --no-thinking "
                        "--prompt \"Reply with exactly: DwarfStar ready\"",
                        "Run a bounded deterministic first inference check.",
                    ),
                    _action(
                        "./rocmplete run dwarfstar server --context 32768",
                        "Use a smaller context and working set.",
                    ),
                ),
                role="info",
            ),
            GuideSection(
                "Network access",
                (
                    "The server defaults to loopback and does not provide "
                    "authentication. Put a trusted authenticated proxy in "
                    "front of it before publishing beyond a trusted LAN.",
                    "DwarfStar binds to a wildcard inside its private "
                    "container namespace so Podman can forward the port. "
                    "The host publication remains 127.0.0.1 unless --listen "
                    "selects another host address.",
                ),
                (
                    _action(
                        "./rocmplete run dwarfstar server "
                        "--listen 192.168.1.50",
                        "Publish on one exact host address without built-in "
                        "authentication.",
                    ),
                    _action(
                        "./rocmplete run dwarfstar server --port 8001",
                        "Change the host and application port together.",
                    ),
                ),
                role="warning",
            ),
            GuideSection(
                "Day-to-day",
                actions=(
                    _action(
                        "./rocmplete run dwarfstar server --detach",
                        "Start the API in the background.",
                    ),
                    _action(
                        "./rocmplete agent opencode -- "
                        "-m dwarfstar/deepseek-v4-flash",
                        "Use the running server through ROCmplete's guarded "
                        "OpenCode launcher.",
                    ),
                    _action(
                        "./rocmplete agent pi -- --provider dwarfstar "
                        "--model deepseek-v4-flash --thinking high",
                        "Use the same server through ROCmplete's guarded "
                        "Pi launcher.",
                    ),
                    _action(
                        "./rocmplete agent omp -- --model "
                        "dwarfstar/deepseek-v4-flash --thinking high",
                        "Use the same server through ROCmplete's guarded "
                        "OMP launcher.",
                    ),
                    _action(
                        "./rocmplete agent maki -- "
                        "-m dwarfstar/deepseek-v4-flash",
                        "Use the same server through ROCmplete's guarded "
                        "Maki launcher.",
                    ),
                    _action(
                        "./rocmplete logs dwarfstar --follow",
                        "Follow output from a detached server.",
                    ),
                    _action(
                        "./rocmplete stop dwarfstar",
                        "Stop and remove the managed container.",
                    ),
                    _action(
                        "./rocmplete run dwarfstar --help",
                        "Show the managed server and CLI modes.",
                    ),
                ),
            ),
        ),
        reference="guide/applications.md#dwarfstar",
    )


APPLICATION_GUIDES: Mapping[str, ApplicationGuide] = {
    guide.application: guide
    for guide in (
        _comfyui_guide(),
        _llama_guide(),
        _dwarfstar_guide(),
    )
}


def guide_commands(guide: ApplicationGuide) -> Tuple[str, ...]:
    """Return every copyable ROCmplete command in a guide."""
    return tuple(
        action.command
        for section in guide.sections
        for action in section.actions
    )


def _print_wrapped(value: str, indent: str, role: Optional[str] = None) -> None:
    width = max(40, 88 - len(indent))
    lines = textwrap.wrap(
        value,
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
    ) or [""]
    for line in lines:
        rendered = style(line, role) if role is not None else line
        print("{}{}".format(indent, rendered))


def _print_actions(actions: Sequence[GuideAction]) -> None:
    for action in actions:
        print("    {}".format(style(action.command, "command")))
        _print_wrapped(action.description, "        ", role="muted")


def print_application_guide(application: Optional[str]) -> int:
    """Print the application index or one focused mini-guide."""
    if application is None:
        print(style("Application guides", "heading"))
        print_columns(
            tuple(
                (identifier, guide.title, guide.summary)
                for identifier, guide in APPLICATION_GUIDES.items()
            ),
            columns=(
                ColumnSpec(role="command"),
                ColumnSpec(role="label"),
                ColumnSpec(role="muted"),
            ),
            indent="  ",
        )
        print()
        print(
            "Run {} for one focused walkthrough.".format(
                style(
                    "./rocmplete guide APPLICATION",
                    "command",
                )
            )
        )
        return 0

    try:
        guide = APPLICATION_GUIDES[application]
    except KeyError:
        raise LauncherError(
            "unknown application guide {!r}; choose {}".format(
                application, ", ".join(APPLICATION_GUIDES)
            )
        )

    print(style(guide.title, "heading"))
    _print_wrapped(guide.summary, "  ", role="muted")
    for section in guide.sections:
        print()
        print(style(section.title, section.role))
        for paragraph in section.paragraphs:
            _print_wrapped(paragraph, "  ")
        if section.paragraphs and section.actions:
            print()
        _print_actions(section.actions)
    print()
    print(style("Full reference", "label"))
    print("  {}".format(style(guide.reference, "command")))
    return 0
