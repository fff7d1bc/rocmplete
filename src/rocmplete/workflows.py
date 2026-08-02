"""Curated workflow installation for the ComfyUI application."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

from . import podman
from .catalog import WorkflowPack
from .errors import LauncherError
from .layout import StorageLayout, validate_managed_parent
from .ui import style

_MIT_NOTICE = """MIT License

Copyright (c) 2023-present Comfy Org

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""

_MODEL_SOURCES: Mapping[str, Tuple[str, str, str]] = {
    "qwen_image_2512_fp8_e4m3fn.safetensors": (
        "Comfy-Org/Qwen-Image_ComfyUI",
        "46839d338df81ce625d5fae27d7e370314c0fbc9",
        "split_files/diffusion_models/qwen_image_2512_fp8_e4m3fn.safetensors",
    ),
    "qwen_image_2512_bf16.safetensors": (
        "Comfy-Org/Qwen-Image_ComfyUI",
        "46839d338df81ce625d5fae27d7e370314c0fbc9",
        "split_files/diffusion_models/qwen_image_2512_bf16.safetensors",
    ),
    "qwen_2.5_vl_7b_fp8_scaled.safetensors": (
        "Comfy-Org/Qwen-Image_ComfyUI",
        "46839d338df81ce625d5fae27d7e370314c0fbc9",
        "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
    ),
    "qwen_image_vae.safetensors": (
        "Comfy-Org/Qwen-Image_ComfyUI",
        "46839d338df81ce625d5fae27d7e370314c0fbc9",
        "split_files/vae/qwen_image_vae.safetensors",
    ),
    "Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors": (
        "lightx2v/Qwen-Image-2512-Lightning",
        "a52649c9d0f6e1a248bff13f0df33bb8a2abdb52",
        "Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors",
    ),
    "qwen_image_edit_2511_fp8mixed.safetensors": (
        "Comfy-Org/Qwen-Image-Edit_ComfyUI",
        "e9e85de74a8f48c1e3e2656617626348675a2f21",
        "split_files/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors",
    ),
    "qwen_image_edit_2511_bf16.safetensors": (
        "Comfy-Org/Qwen-Image-Edit_ComfyUI",
        "e9e85de74a8f48c1e3e2656617626348675a2f21",
        "split_files/diffusion_models/qwen_image_edit_2511_bf16.safetensors",
    ),
    "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors": (
        "lightx2v/Qwen-Image-Edit-2511-Lightning",
        "d74eba145674fd7e31b949324e148e21e7118abd",
        "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
    ),
    "umt5_xxl_fp8_e4m3fn_scaled.safetensors": (
        "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        "fb1388adc906ab39ffc26ee40e96b22886b56bc4",
        "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
    ),
    "umt5_xxl_fp16.safetensors": (
        "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        "fb1388adc906ab39ffc26ee40e96b22886b56bc4",
        "split_files/text_encoders/umt5_xxl_fp16.safetensors",
    ),
    "wan_2.1_vae.safetensors": (
        "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        "fb1388adc906ab39ffc26ee40e96b22886b56bc4",
        "split_files/vae/wan_2.1_vae.safetensors",
    ),
    "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors": (
        "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        "fb1388adc906ab39ffc26ee40e96b22886b56bc4",
        "split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors",
    ),
    "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors": (
        "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        "fb1388adc906ab39ffc26ee40e96b22886b56bc4",
        "split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors",
    ),
    "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors": (
        "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        "fb1388adc906ab39ffc26ee40e96b22886b56bc4",
        "split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
    ),
    "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors": (
        "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        "fb1388adc906ab39ffc26ee40e96b22886b56bc4",
        "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
    ),
    "wan2.2_t2v_high_noise_14B_fp16.safetensors": (
        "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        "fb1388adc906ab39ffc26ee40e96b22886b56bc4",
        "split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp16.safetensors",
    ),
    "wan2.2_t2v_low_noise_14B_fp16.safetensors": (
        "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        "fb1388adc906ab39ffc26ee40e96b22886b56bc4",
        "split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp16.safetensors",
    ),
    "wan2.2_i2v_high_noise_14B_fp16.safetensors": (
        "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        "fb1388adc906ab39ffc26ee40e96b22886b56bc4",
        "split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors",
    ),
    "wan2.2_i2v_low_noise_14B_fp16.safetensors": (
        "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        "fb1388adc906ab39ffc26ee40e96b22886b56bc4",
        "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors",
    ),
    "wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors": (
        "lightx2v/Wan2.2-Lightning",
        "18bccf8884ec0a078eed79785eb4ef13ea16ce1e",
        "Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/high_noise_model.safetensors",
    ),
    "wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors": (
        "lightx2v/Wan2.2-Lightning",
        "18bccf8884ec0a078eed79785eb4ef13ea16ce1e",
        "Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/low_noise_model.safetensors",
    ),
    "wan2.2_t2v_lightx2v_4steps_lora_v2_high_noise.safetensors": (
        "lightx2v/Wan2.2-Lightning",
        "18bccf8884ec0a078eed79785eb4ef13ea16ce1e",
        "Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V2.0/high_noise_model.safetensors",
    ),
    "wan2.2_t2v_lightx2v_4steps_lora_v2_low_noise.safetensors": (
        "lightx2v/Wan2.2-Lightning",
        "18bccf8884ec0a078eed79785eb4ef13ea16ce1e",
        "Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V2.0/low_noise_model.safetensors",
    ),
    "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors": (
        "lightx2v/Wan2.2-Lightning",
        "18bccf8884ec0a078eed79785eb4ef13ea16ce1e",
        "Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/high_noise_model.safetensors",
    ),
    "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors": (
        "lightx2v/Wan2.2-Lightning",
        "18bccf8884ec0a078eed79785eb4ef13ea16ce1e",
        "Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/low_noise_model.safetensors",
    ),
    "ltx-2-19b-dev-fp8.safetensors": (
        "Lightricks/LTX-2",
        "47da56e2ad66ce4125a9922b4a8826bf407f9d0a",
        "ltx-2-19b-dev-fp8.safetensors",
    ),
    "ltx-2-19b-dev.safetensors": (
        "Lightricks/LTX-2",
        "47da56e2ad66ce4125a9922b4a8826bf407f9d0a",
        "ltx-2-19b-dev.safetensors",
    ),
    "ltx-2-19b-distilled-fp8.safetensors": (
        "Lightricks/LTX-2",
        "47da56e2ad66ce4125a9922b4a8826bf407f9d0a",
        "ltx-2-19b-distilled-fp8.safetensors",
    ),
    "ltx-2-19b-distilled-lora-384.safetensors": (
        "Lightricks/LTX-2",
        "47da56e2ad66ce4125a9922b4a8826bf407f9d0a",
        "ltx-2-19b-distilled-lora-384.safetensors",
    ),
    "ltx-2-spatial-upscaler-x2-1.0.safetensors": (
        "Lightricks/LTX-2",
        "47da56e2ad66ce4125a9922b4a8826bf407f9d0a",
        "ltx-2-spatial-upscaler-x2-1.0.safetensors",
    ),
    "ltx-2-19b-lora-camera-control-dolly-left.safetensors": (
        "Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Left",
        "75cdba2244db6e2934d06095dd8bb6efa33006f6",
        "ltx-2-19b-lora-camera-control-dolly-left.safetensors",
    ),
    "gemma_3_12B_it_fp4_mixed.safetensors": (
        "Comfy-Org/ltx-2",
        "bd5f9c87fcb0360ae7112f9784562670894d9492",
        "split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors",
    ),
    "byt5_small_glyphxl_fp16.safetensors": (
        "Comfy-Org/HunyuanVideo_1.5_repackaged",
        "2fa8a1a2d48fe2837368eb64b400217e595ab117",
        "split_files/text_encoders/byt5_small_glyphxl_fp16.safetensors",
    ),
    "hunyuanvideo15_vae_fp16.safetensors": (
        "Comfy-Org/HunyuanVideo_1.5_repackaged",
        "2fa8a1a2d48fe2837368eb64b400217e595ab117",
        "split_files/vae/hunyuanvideo15_vae_fp16.safetensors",
    ),
    "hunyuanvideo1.5_720p_t2v_fp16.safetensors": (
        "Comfy-Org/HunyuanVideo_1.5_repackaged",
        "2fa8a1a2d48fe2837368eb64b400217e595ab117",
        "split_files/diffusion_models/hunyuanvideo1.5_720p_t2v_fp16.safetensors",
    ),
    "hunyuanvideo1.5_720p_i2v_fp16.safetensors": (
        "Comfy-Org/HunyuanVideo_1.5_repackaged",
        "2fa8a1a2d48fe2837368eb64b400217e595ab117",
        "split_files/diffusion_models/hunyuanvideo1.5_720p_i2v_fp16.safetensors",
    ),
    "hunyuanvideo1.5_480p_i2v_step_distilled_fp16.safetensors": (
        "Comfy-Org/HunyuanVideo_1.5_repackaged",
        "2fa8a1a2d48fe2837368eb64b400217e595ab117",
        (
            "split_files/diffusion_models/"
            "hunyuanvideo1.5_480p_i2v_step_distilled_fp16.safetensors"
        ),
    ),
    "hunyuanvideo1.5_480p_t2v_cfg_distilled_fp16.safetensors": (
        "Comfy-Org/HunyuanVideo_1.5_repackaged",
        "2fa8a1a2d48fe2837368eb64b400217e595ab117",
        (
            "split_files/diffusion_models/"
            "hunyuanvideo1.5_480p_t2v_cfg_distilled_fp16.safetensors"
        ),
    ),
    "hunyuanvideo1.5_1080p_sr_distilled_fp16.safetensors": (
        "Comfy-Org/HunyuanVideo_1.5_repackaged",
        "2fa8a1a2d48fe2837368eb64b400217e595ab117",
        "split_files/diffusion_models/hunyuanvideo1.5_1080p_sr_distilled_fp16.safetensors",
    ),
    "hunyuanvideo15_latent_upsampler_1080p.safetensors": (
        "Comfy-Org/HunyuanVideo_1.5_repackaged",
        "2fa8a1a2d48fe2837368eb64b400217e595ab117",
        "split_files/latent_upscale_models/hunyuanvideo15_latent_upsampler_1080p.safetensors",
    ),
    "sigclip_vision_patch14_384.safetensors": (
        "Comfy-Org/HunyuanVideo_1.5_repackaged",
        "2fa8a1a2d48fe2837368eb64b400217e595ab117",
        "split_files/clip_vision/sigclip_vision_patch14_384.safetensors",
    ),
}

_MODEL_ALIASES: Mapping[str, str] = {
    "ltx-2-19b-distilled.safetensors": (
        "ltx-2-19b-distilled-fp8.safetensors"
    ),
}


def workflow_destination(data_dir: Path, pack: WorkflowPack) -> Path:
    return StorageLayout(data_dir).curated_workflows / pack.destination


def workflow_state(data_dir: Path, pack: WorkflowPack) -> str:
    destination = workflow_destination(data_dir, pack)
    try:
        status = destination.lstat()
    except FileNotFoundError:
        return "missing"
    except OSError as error:
        raise LauncherError("cannot inspect {}: {}".format(destination, error))
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISREG(status.st_mode):
        raise LauncherError(
            "workflow path is not a regular file: {}".format(destination)
        )
    try:
        contents = destination.read_bytes()
    except OSError as error:
        raise LauncherError("cannot read {}: {}".format(destination, error))
    digest = hashlib.sha256(contents).hexdigest()
    return "installed" if digest == pack.rendered_sha256 else "modified"


def source_command(image: str, pack: WorkflowPack) -> List[str]:
    script = (
        "import importlib.metadata, importlib.resources, sys; "
        "actual=importlib.metadata.version(sys.argv[1]); "
        "expected=sys.argv[2]; "
        "actual == expected or sys.exit("
        "'package version mismatch: expected %s, got %s' % (expected, actual)); "
        "root=importlib.resources.files(sys.argv[1]); "
        "sys.stdout.buffer.write(root.joinpath(sys.argv[3]).read_bytes())"
    )
    return [
        "podman",
        "run",
        "--rm",
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "all",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        "128",
        "--entrypoint",
        "/opt/venv/bin/python",
        image,
        "-c",
        script,
        pack.source_package,
        pack.source_version,
        pack.source_resource,
    ]


def fetch_source(pack: WorkflowPack, image: str) -> Mapping[str, object]:
    podman.require_rootless()
    if not podman.image_exists(image):
        raise LauncherError(
            "image not found: {} (run './rocmplete build comfyui')".format(image)
        )
    source = podman.capture_bytes(
        source_command(image, pack),
        "cannot read workflow template from image {}".format(image),
    )
    digest = hashlib.sha256(source).hexdigest()
    if digest != pack.source_sha256:
        raise LauncherError(
            "workflow template does not match the pinned catalog "
            "(expected {}, got {})".format(pack.source_sha256, digest)
        )
    try:
        result = json.loads(source)
    except json.JSONDecodeError as error:
        raise LauncherError("workflow template contains invalid JSON: {}".format(error))
    if not isinstance(result, dict):
        raise LauncherError("workflow template root must be an object")
    return result


def _node_groups(
    workflow: MutableMapping[str, object],
) -> Iterable[MutableMapping[str, object]]:
    yield workflow
    definitions = workflow.get("definitions")
    if not isinstance(definitions, dict):
        return
    subgraphs = definitions.get("subgraphs")
    if not isinstance(subgraphs, list):
        return
    for item in subgraphs:
        if isinstance(item, dict):
            yield item


def _nodes(group: MutableMapping[str, object]) -> List[MutableMapping[str, object]]:
    value = group.get("nodes")
    if not isinstance(value, list):
        raise LauncherError("official workflow node collection is invalid")
    return [item for item in value if isinstance(item, dict)]


def _all_nodes(
    workflow: MutableMapping[str, object],
) -> Iterable[MutableMapping[str, object]]:
    for group in _node_groups(workflow):
        yield from _nodes(group)


def _widgets(node: MutableMapping[str, object]) -> List[object]:
    value = node.get("widgets_values")
    if not isinstance(value, list):
        raise LauncherError(
            "official {} node has invalid widgets".format(node.get("type"))
        )
    return value


def _find_nodes(
    workflow: MutableMapping[str, object], node_type: str
) -> List[MutableMapping[str, object]]:
    return [node for node in _all_nodes(workflow) if node.get("type") == node_type]


def _set_first_widget(
    workflow: MutableMapping[str, object], node_type: str, filename: str
) -> None:
    nodes = _find_nodes(workflow, node_type)
    if not nodes:
        raise LauncherError(
            "official workflow is missing {} node".format(node_type)
        )
    for node in nodes:
        widgets = _widgets(node)
        if not widgets:
            raise LauncherError(
                "official {} node has no filename".format(node_type)
            )
        widgets[0] = filename


def _remove_markdown_nodes(workflow: MutableMapping[str, object]) -> None:
    for group in _node_groups(workflow):
        nodes = group.get("nodes")
        if isinstance(nodes, list):
            group["nodes"] = [
                node
                for node in nodes
                if not isinstance(node, dict)
                or node.get("type") != "MarkdownNote"
            ]


def _source_node(
    group: MutableMapping[str, object], identifier: object
) -> MutableMapping[str, object]:
    for node in _nodes(group):
        if node.get("id") == identifier:
            return node
    raise LauncherError("official workflow link references a missing node")


def _link_value(link: object, index: int, key: str) -> object:
    if isinstance(link, list) and len(link) >= 6:
        return link[index]
    if isinstance(link, dict):
        return link.get(key)
    return None


def _set_link_value(link: object, index: int, key: str, value: object) -> None:
    if isinstance(link, list) and len(link) >= 6:
        link[index] = value
    elif isinstance(link, dict):
        link[key] = value
    else:
        raise LauncherError("official workflow link is invalid")


def _remove_link_references(
    group: MutableMapping[str, object], link_ids: Sequence[object]
) -> None:
    identifiers = set(link_ids)
    for node in _nodes(group):
        inputs = node.get("inputs")
        if isinstance(inputs, list):
            for item in inputs:
                if isinstance(item, dict) and item.get("link") in identifiers:
                    item["link"] = None
        outputs = node.get("outputs")
        if isinstance(outputs, list):
            for item in outputs:
                if not isinstance(item, dict):
                    continue
                links = item.get("links")
                if isinstance(links, list):
                    item["links"] = [
                        link for link in links if link not in identifiers
                    ]
    for field in ("inputs", "outputs"):
        values = group.get(field)
        if not isinstance(values, list):
            continue
        for item in values:
            if not isinstance(item, dict):
                continue
            links = item.get("linkIds")
            if isinstance(links, list):
                item["linkIds"] = [
                    link for link in links if link not in identifiers
                ]


def _remove_passthrough_node(
    group: MutableMapping[str, object],
    node: MutableMapping[str, object],
) -> None:
    links_value = group.get("links")
    if not isinstance(links_value, list):
        raise LauncherError("official workflow links are invalid")
    identifier = node.get("id")
    inbound = [
        link
        for link in links_value
        if _link_value(link, 3, "target_id") == identifier
    ]
    outbound = [
        link
        for link in links_value
        if _link_value(link, 1, "origin_id") == identifier
    ]
    model_inbound = [
        link
        for link in inbound
        if _link_value(link, 5, "type") == "MODEL"
        and _link_value(link, 4, "target_slot") == 0
    ]
    if len(model_inbound) != 1 or not outbound:
        raise LauncherError(
            "official LoRA node is not a simple model passthrough"
        )
    incoming = model_inbound[0]
    source_id = _link_value(incoming, 1, "origin_id")
    source_slot = _link_value(incoming, 2, "origin_slot")
    source = _source_node(group, source_id)
    outputs = source.get("outputs")
    if not isinstance(outputs, list) or not isinstance(source_slot, int):
        raise LauncherError("official workflow model output is invalid")
    if source_slot >= len(outputs) or not isinstance(outputs[source_slot], dict):
        raise LauncherError("official workflow model output is missing")
    output_links = outputs[source_slot].get("links")
    if output_links is None:
        output_links = []
    if not isinstance(output_links, list):
        raise LauncherError("official workflow model links are invalid")
    inbound_ids = [_link_value(link, 0, "id") for link in inbound]
    output_links = [item for item in output_links if item not in inbound_ids]
    for link in outbound:
        _set_link_value(link, 1, "origin_id", source_id)
        _set_link_value(link, 2, "origin_slot", source_slot)
        link_id = _link_value(link, 0, "id")
        if link_id not in output_links:
            output_links.append(link_id)
    outputs[source_slot]["links"] = output_links
    _remove_link_references(group, inbound_ids)
    group["links"] = [link for link in links_value if link not in inbound]
    nodes = group.get("nodes")
    if isinstance(nodes, list):
        group["nodes"] = [item for item in nodes if item is not node]


def _remove_loras(workflow: MutableMapping[str, object]) -> None:
    found = 0
    for group in list(_node_groups(workflow)):
        for node in list(_nodes(group)):
            if node.get("type") == "LoraLoaderModelOnly":
                _remove_passthrough_node(group, node)
                found += 1
    if found == 0:
        raise LauncherError("official workflow has no expected LoRA nodes")


def _remove_loras_named(
    workflow: MutableMapping[str, object], fragment: str
) -> None:
    found = 0
    for group in list(_node_groups(workflow)):
        for node in list(_nodes(group)):
            if node.get("type") != "LoraLoaderModelOnly":
                continue
            widgets = _widgets(node)
            if widgets and fragment in str(widgets[0]):
                _remove_passthrough_node(group, node)
                found += 1
    if found == 0:
        raise LauncherError(
            "official workflow has no expected LoRA matching {!r}".format(
                fragment
            )
        )


def _clear_input_media(workflow: MutableMapping[str, object]) -> None:
    for node_type in ("LoadImage", "LoadVideo"):
        for node in _find_nodes(workflow, node_type):
            widgets = _widgets(node)
            if widgets:
                widgets[0] = ""


def _configure_qwen(
    workflow: MutableMapping[str, object],
    edit: bool,
    lightning: bool,
    bf16: bool = False,
) -> None:
    if edit:
        _set_first_widget(
            workflow,
            "UNETLoader",
            (
                "qwen_image_edit_2511_bf16.safetensors"
                if bf16
                else "qwen_image_edit_2511_fp8mixed.safetensors"
            ),
        )
        steps, cfg = ((4, 1.0) if lightning else (20, 4.0))
        lora = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
    else:
        _set_first_widget(
            workflow,
            "UNETLoader",
            (
                "qwen_image_2512_bf16.safetensors"
                if bf16
                else "qwen_image_2512_fp8_e4m3fn.safetensors"
            ),
        )
        steps, cfg = ((4, 1.0) if lightning else (20, 2.5))
        lora = "Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors"
    _set_first_widget(
        workflow, "CLIPLoader", "qwen_2.5_vl_7b_fp8_scaled.safetensors"
    )
    _set_first_widget(workflow, "VAELoader", "qwen_image_vae.safetensors")
    samplers = _find_nodes(workflow, "KSampler")
    if len(samplers) != 1:
        raise LauncherError("official Qwen workflow must contain one KSampler")
    widgets = _widgets(samplers[0])
    if len(widgets) < 4:
        raise LauncherError("official Qwen sampler widgets are invalid")
    widgets[2], widgets[3] = steps, cfg
    if lightning:
        _set_first_widget(workflow, "LoraLoaderModelOnly", lora)
    else:
        _remove_loras(workflow)


def _configure_wan(
    workflow: MutableMapping[str, object],
    mode: str,
    lightning: bool,
    fp16: bool = False,
    lightning_version: str = "v1.1",
) -> None:
    prefix = "wan2.2_{}_".format(mode)
    model_nodes = _find_nodes(workflow, "UNETLoader")
    names = {
        "high": "{}high_noise_14B_{}.safetensors".format(
            prefix, "fp16" if fp16 else "fp8_scaled"
        ),
        "low": "{}low_noise_14B_{}.safetensors".format(
            prefix, "fp16" if fp16 else "fp8_scaled"
        ),
    }
    if len(model_nodes) != 2:
        raise LauncherError("official Wan workflow must contain two UNET loaders")
    for node in model_nodes:
        widgets = _widgets(node)
        current = str(widgets[0]) if widgets else ""
        key = "high" if "high_noise" in current else "low"
        widgets[0] = names[key]
    _set_first_widget(
        workflow,
        "CLIPLoader",
        (
            "umt5_xxl_fp16.safetensors"
            if fp16
            else "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
        ),
    )
    _set_first_widget(workflow, "VAELoader", "wan_2.1_vae.safetensors")

    samplers = _find_nodes(workflow, "KSamplerAdvanced")
    if len(samplers) != 2:
        raise LauncherError(
            "official Wan workflow must contain two advanced samplers"
        )
    for sampler in samplers:
        widgets = _widgets(sampler)
        if len(widgets) < 9:
            raise LauncherError("official Wan sampler widgets are invalid")
        high_noise = widgets[7] == 0
        if lightning:
            widgets[3], widgets[4] = 4, 1.0
            widgets[7], widgets[8] = ((0, 2) if high_noise else (2, 4))
        else:
            widgets[3], widgets[4] = 20, 3.5
            widgets[7], widgets[8] = ((0, 10) if high_noise else (10, 10000))
    if lightning:
        loras = _find_nodes(workflow, "LoraLoaderModelOnly")
        if len(loras) != 2:
            raise LauncherError(
                "official Wan workflow must contain two LoRA loaders"
            )
        for node in loras:
            widgets = _widgets(node)
            current = str(widgets[0]) if widgets else ""
            noise = "high_noise" if "high_noise" in current else "low_noise"
            if mode == "t2v":
                widgets[0] = (
                    "wan2.2_t2v_lightx2v_4steps_lora_{}_{}.safetensors".format(
                        lightning_version, noise
                    )
                )
            else:
                widgets[0] = (
                    "wan2.2_i2v_lightx2v_4steps_lora_v1_{}.safetensors".format(
                        noise
                    )
                )
    else:
        _remove_loras(workflow)


def _configure_ltx(
    workflow: MutableMapping[str, object],
    distilled: bool,
    bf16: bool = False,
) -> None:
    model_name = (
        "ltx-2-19b-distilled-fp8.safetensors"
        if distilled
        else (
            "ltx-2-19b-dev.safetensors"
            if bf16
            else "ltx-2-19b-dev-fp8.safetensors"
        )
    )
    for node_type in ("CheckpointLoaderSimple", "LTXVAudioVAELoader"):
        _set_first_widget(workflow, node_type, model_name)
    text_loaders = _find_nodes(workflow, "LTXAVTextEncoderLoader")
    if not text_loaders:
        raise LauncherError(
            "official LTX workflow is missing LTXAVTextEncoderLoader"
        )
    for node in text_loaders:
        widgets = _widgets(node)
        if len(widgets) < 2:
            raise LauncherError(
                "official LTX text encoder loader widgets are invalid"
            )
        widgets[0] = "gemma_3_12B_it_fp4_mixed.safetensors"
        widgets[1] = model_name
    _set_first_widget(
        workflow,
        "LatentUpscaleModelLoader",
        "ltx-2-spatial-upscaler-x2-1.0.safetensors",
    )
    for node in _all_nodes(workflow):
        widgets = node.get("widgets_values")
        if not isinstance(widgets, list):
            continue
        for index, value in enumerate(widgets):
            if value == "ltx-2-19b-distilled.safetensors":
                widgets[index] = "ltx-2-19b-distilled-fp8.safetensors"
    if distilled:
        _remove_loras_named(workflow, "camera-control")
    else:
        camera_loras = [
            node
            for node in _all_nodes(workflow)
            if node.get("type") == "LoraLoaderModelOnly"
            and _widgets(node)
            and "camera-control" in str(_widgets(node)[0])
        ]
        if not camera_loras:
            raise LauncherError(
                "official workflow has no expected camera-control LoRA"
            )
        if any(node.get("mode") != 4 for node in camera_loras):
            raise LauncherError(
                "official camera-control LoRA must remain bypassed by default"
            )
        for node in camera_loras:
            _widgets(node)[0] = (
                "ltx-2-19b-lora-camera-control-dolly-left.safetensors"
            )


def _configure_hunyuan(workflow: MutableMapping[str, object]) -> None:
    _set_first_widget(
        workflow, "VAELoader", "hunyuanvideo15_vae_fp16.safetensors"
    )
    _set_first_widget(
        workflow,
        "LatentUpscaleModelLoader",
        "hunyuanvideo15_latent_upsampler_1080p.safetensors",
    )
    enabled = 0
    for node in _all_nodes(workflow):
        if node.get("mode") == 4 and node.get("type") != "Note":
            node["mode"] = 0
            enabled += 1
    if enabled == 0:
        raise LauncherError(
            "official Hunyuan workflow has no bypassed super-resolution stage"
        )


def _configure_hunyuan_480p(
    workflow: MutableMapping[str, object], mode: str
) -> None:
    _set_first_widget(
        workflow, "VAELoader", "hunyuanvideo15_vae_fp16.safetensors"
    )
    if mode == "i2v":
        model = "hunyuanvideo1.5_480p_i2v_step_distilled_fp16.safetensors"
        dimension_type = "HunyuanVideo15ImageToVideo"
        steps, shift = 12, 7
    elif mode == "t2v":
        model = "hunyuanvideo1.5_480p_t2v_cfg_distilled_fp16.safetensors"
        dimension_type = "EmptyHunyuanVideo15Latent"
        steps, shift = 50, 5
    else:
        raise LauncherError("unknown Hunyuan workflow mode: {}".format(mode))

    counts = {
        "UNETLoader": 0,
        dimension_type: 0,
        "BasicScheduler": 0,
        "CFGGuider": 0,
        "ModelSamplingSD3": 0,
    }
    for node in _all_nodes(workflow):
        if node.get("mode", 0) == 4:
            continue
        node_type = node.get("type")
        widgets = node.get("widgets_values")
        if not isinstance(widgets, list):
            continue
        if node_type == "UNETLoader":
            if not widgets:
                raise LauncherError("official Hunyuan UNET loader is invalid")
            widgets[0] = model
            properties = node.get("properties")
            models = (
                properties.get("models")
                if isinstance(properties, dict)
                else None
            )
            if (
                not isinstance(models, list)
                or len(models) != 1
                or not isinstance(models[0], dict)
            ):
                raise LauncherError(
                    "official Hunyuan UNET metadata is invalid"
                )
            models[0]["name"] = model
            counts[node_type] += 1
        elif node_type == dimension_type:
            if len(widgets) < 2:
                raise LauncherError("official Hunyuan dimensions are invalid")
            widgets[0], widgets[1] = 832, 480
            counts[node_type] += 1
        elif node_type == "BasicScheduler":
            if len(widgets) < 2:
                raise LauncherError("official Hunyuan scheduler is invalid")
            widgets[1] = steps
            counts[node_type] += 1
        elif node_type == "CFGGuider":
            if not widgets:
                raise LauncherError("official Hunyuan guider is invalid")
            widgets[0] = 1
            counts[node_type] += 1
        elif node_type == "ModelSamplingSD3":
            if not widgets:
                raise LauncherError("official Hunyuan sampling node is invalid")
            widgets[0] = shift
            counts[node_type] += 1
    if any(count != 1 for count in counts.values()):
        raise LauncherError(
            "official Hunyuan base workflow changed unexpectedly: {}".format(
                counts
            )
        )


def _rewrite_model_metadata(workflow: MutableMapping[str, object]) -> None:
    for node in _all_nodes(workflow):
        properties = node.get("properties")
        if not isinstance(properties, dict):
            continue
        models = properties.get("models")
        if not isinstance(models, list):
            continue
        for model in models:
            if not isinstance(model, dict):
                continue
            name = model.get("name")
            if name in _MODEL_ALIASES:
                name = _MODEL_ALIASES[name]
                model["name"] = name
            if name not in _MODEL_SOURCES:
                continue
            repository, revision, path = _MODEL_SOURCES[name]
            model["url"] = "https://huggingface.co/{}/resolve/{}/{}".format(
                repository, revision, path
            )


def _assert_core_nodes(workflow: MutableMapping[str, object]) -> None:
    for node in _all_nodes(workflow):
        properties = node.get("properties")
        if not isinstance(properties, dict):
            continue
        registry = properties.get("cnr_id")
        if registry not in (None, "comfy-core"):
            raise LauncherError(
                "curated workflow unexpectedly requires custom node {}".format(
                    registry
                )
            )


def _add_provenance(
    workflow: MutableMapping[str, object], pack: WorkflowPack
) -> None:
    extra = workflow.get("extra")
    if not isinstance(extra, dict):
        extra = {}
        workflow["extra"] = extra
    extra.pop("prompt", None)
    extra["rocmplete"] = {
        "bundle": pack.identifier,
        "modified": True,
        "modified_by": "ROCmplete",
        "source": (
            "https://github.com/Comfy-Org/workflow_templates/blob/"
            + pack.source_revision
            + "/"
            + pack.source_resource
        ),
        "source_package": pack.source_package,
        "source_version": pack.source_version,
        "source_revision": pack.source_revision,
        "source_sha256": pack.source_sha256,
        "license": pack.license,
        "license_url": pack.license_url,
        "license_notice": _MIT_NOTICE,
    }


def render_workflow(
    pack: WorkflowPack, source: Mapping[str, object]
) -> bytes:
    workflow = copy.deepcopy(source)
    if not isinstance(workflow, dict):
        raise LauncherError("official workflow root is invalid")
    _remove_markdown_nodes(workflow)
    _clear_input_media(workflow)
    renderer = pack.renderer
    if renderer == "qwen-image-base":
        _configure_qwen(workflow, edit=False, lightning=False)
    elif renderer == "qwen-image-base-bf16":
        _configure_qwen(workflow, edit=False, lightning=False, bf16=True)
    elif renderer == "qwen-image-lightning":
        _configure_qwen(workflow, edit=False, lightning=True)
    elif renderer == "qwen-image-lightning-bf16":
        _configure_qwen(workflow, edit=False, lightning=True, bf16=True)
    elif renderer == "qwen-edit-base":
        _configure_qwen(workflow, edit=True, lightning=False)
    elif renderer == "qwen-edit-base-bf16":
        _configure_qwen(workflow, edit=True, lightning=False, bf16=True)
    elif renderer == "qwen-edit-lightning":
        _configure_qwen(workflow, edit=True, lightning=True)
    elif renderer == "qwen-edit-lightning-bf16":
        _configure_qwen(workflow, edit=True, lightning=True, bf16=True)
    elif renderer == "wan-t2v-base":
        _configure_wan(workflow, mode="t2v", lightning=False)
    elif renderer == "wan-t2v-base-fp16":
        _configure_wan(workflow, mode="t2v", lightning=False, fp16=True)
    elif renderer == "wan-t2v-lightning":
        _configure_wan(workflow, mode="t2v", lightning=True)
    elif renderer == "wan-t2v-lightning-fp16":
        _configure_wan(workflow, mode="t2v", lightning=True, fp16=True)
    elif renderer == "wan-t2v-lightning-v2":
        _configure_wan(
            workflow, mode="t2v", lightning=True, lightning_version="v2"
        )
    elif renderer == "wan-t2v-lightning-v2-fp16":
        _configure_wan(
            workflow,
            mode="t2v",
            lightning=True,
            fp16=True,
            lightning_version="v2",
        )
    elif renderer == "wan-i2v-base":
        _configure_wan(workflow, mode="i2v", lightning=False)
    elif renderer == "wan-i2v-base-fp16":
        _configure_wan(workflow, mode="i2v", lightning=False, fp16=True)
    elif renderer == "wan-i2v-lightning":
        _configure_wan(workflow, mode="i2v", lightning=True)
    elif renderer == "wan-i2v-lightning-fp16":
        _configure_wan(workflow, mode="i2v", lightning=True, fp16=True)
    elif renderer == "ltx-full":
        _configure_ltx(workflow, distilled=False)
    elif renderer == "ltx-full-bf16":
        _configure_ltx(workflow, distilled=False, bf16=True)
    elif renderer == "ltx-distilled":
        _configure_ltx(workflow, distilled=True)
    elif renderer == "hunyuan-1.5":
        _configure_hunyuan(workflow)
    elif renderer == "hunyuan-i2v-480p-step-distilled":
        _configure_hunyuan_480p(workflow, mode="i2v")
    elif renderer == "hunyuan-t2v-480p-cfg-distilled":
        _configure_hunyuan_480p(workflow, mode="t2v")
    else:
        raise LauncherError("unknown workflow renderer: {}".format(renderer))
    _rewrite_model_metadata(workflow)
    _assert_core_nodes(workflow)
    _add_provenance(workflow, pack)
    rendered = json.dumps(
        workflow, ensure_ascii=False, indent=2, sort_keys=True
    ) + "\n"
    return rendered.encode("utf-8")


def install_workflow(
    pack: WorkflowPack,
    data_dir: Path,
    image: str,
    force: bool = False,
) -> Path:
    destination = workflow_destination(data_dir, pack)
    managed_root = StorageLayout(data_dir).curated_workflows
    validate_managed_parent(
        destination, managed_root, data_dir, "curated workflow"
    )
    # The catalog-pinned rendered hash is the idempotency boundary. Re-fetching
    # the pinned source here would start one container per already-valid file.
    state = workflow_state(data_dir, pack)
    if state == "installed":
        print(
            "{} {}".format(
                style("Workflow already installed:", "success"),
                destination,
            )
        )
        return destination
    if state == "modified" and not force:
        raise LauncherError(
            "workflow already exists and differs from the curated version: {}; "
            "use --force to replace it".format(destination)
        )

    source = fetch_source(pack, image)
    rendered = render_workflow(pack, source)
    rendered_digest = hashlib.sha256(rendered).hexdigest()
    if rendered_digest != pack.rendered_sha256:
        raise LauncherError(
            "rendered workflow does not match the pinned catalog "
            "(expected {}, got {})".format(
                pack.rendered_sha256, rendered_digest
            )
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    validate_managed_parent(
        destination, managed_root, data_dir, "curated workflow"
    )
    if destination.exists():
        try:
            current = destination.read_bytes()
        except OSError as error:
            raise LauncherError("cannot read {}: {}".format(destination, error))
        if current == rendered:
            print(
                "{} {}".format(
                    style("Workflow already installed:", "success"),
                    destination,
                )
            )
            return destination
        if not force:
            raise LauncherError(
                "workflow already exists and differs from the curated version: {}; "
                "use --force to replace it".format(destination)
            )

    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=str(destination.parent),
            prefix=".{}.".format(destination.name),
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(rendered)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, destination)
    except OSError as error:
        if temporary_name:
            try:
                os.unlink(temporary_name)
            except OSError:
                pass
        raise LauncherError("cannot install workflow {}: {}".format(destination, error))
    print(
        "{} {}".format(
            style("Installed workflow:", "success"), destination
        )
    )
    return destination
