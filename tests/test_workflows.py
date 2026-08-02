import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from rocmplete.catalog import WorkflowPack
from rocmplete.errors import LauncherError
from rocmplete.workflows import (
    install_workflow,
    render_workflow,
    source_command,
)


def node(identifier, node_type, widgets, inputs=None, outputs=None):
    return {
        "id": identifier,
        "type": node_type,
        "mode": 0,
        "inputs": inputs or [],
        "outputs": outputs or [],
        "properties": {"cnr_id": "comfy-core"},
        "widgets_values": widgets,
    }


def loader_chain(start, model_name, lora_name, sampler_type, sampler_widgets):
    model_link = start + 100
    widget_link = start + 101
    output_link = start + 102
    model = node(
        start,
        "UNETLoader",
        [model_name, "default"],
        outputs=[{"name": "MODEL", "type": "MODEL", "links": [model_link]}],
    )
    lora = node(
        start + 1,
        "LoraLoaderModelOnly",
        [lora_name, 1.0],
        inputs=[
            {"name": "model", "type": "MODEL", "link": model_link},
            {"name": "lora_name", "type": "COMBO", "link": widget_link},
        ],
        outputs=[{"name": "MODEL", "type": "MODEL", "links": [output_link]}],
    )
    sampler = node(
        start + 2,
        sampler_type,
        sampler_widgets,
        inputs=[{"name": "model", "type": "MODEL", "link": output_link}],
    )
    links = [
        {
            "id": model_link,
            "origin_id": start,
            "origin_slot": 0,
            "target_id": start + 1,
            "target_slot": 0,
            "type": "MODEL",
        },
        {
            "id": widget_link,
            "origin_id": -10,
            "origin_slot": 0,
            "target_id": start + 1,
            "target_slot": 1,
            "type": "COMBO",
        },
        {
            "id": output_link,
            "origin_id": start + 1,
            "origin_slot": 0,
            "target_id": start + 2,
            "target_slot": 0,
            "type": "MODEL",
        },
    ]
    return [model, lora, sampler], links, widget_link


def qwen_source(edit=False):
    nodes, links, widget_link = loader_chain(
        10,
        "old-model.safetensors",
        "old-lightning.safetensors",
        "KSampler",
        [1, "randomize", 50, 4, "euler", "simple", 1],
    )
    nodes.extend(
        [
            node(20, "CLIPLoader", ["old-clip.safetensors", "qwen_image"]),
            node(21, "VAELoader", ["old-vae.safetensors"]),
        ]
    )
    top = [
        node(30, "MarkdownNote", ["upstream note"]),
        node(31, "LoadImage", ["sample.png", "image"]) if edit else node(
            31, "SaveImage", ["Qwen"]
        ),
    ]
    return {
        "nodes": top,
        "links": [],
        "definitions": {
            "subgraphs": [
                {
                    "nodes": nodes,
                    "links": links,
                    "inputs": [{"name": "lora", "linkIds": [widget_link]}],
                    "outputs": [],
                }
            ]
        },
        "extra": {"frontendVersion": "1.45.21"},
    }


def wan_source(mode="t2v"):
    high_nodes, high_links, high_widget = loader_chain(
        10,
        "wan2.2_{}_high_noise_14B_fp8_scaled.safetensors".format(mode),
        "old_high_noise.safetensors",
        "KSamplerAdvanced",
        ["enable", 1, "randomize", 4, 1, "euler", "simple", 0, 2, "enable"],
    )
    low_nodes, low_links, low_widget = loader_chain(
        20,
        "wan2.2_{}_low_noise_14B_fp8_scaled.safetensors".format(mode),
        "old_low_noise.safetensors",
        "KSamplerAdvanced",
        ["disable", 0, "fixed", 4, 1, "euler", "simple", 2, 4, "disable"],
    )
    nodes = high_nodes + low_nodes + [
        node(30, "CLIPLoader", ["old-clip.safetensors", "wan"]),
        node(31, "VAELoader", ["old-vae.safetensors"]),
    ]
    top = [node(40, "LoadImage", ["sample.png", "image"])] if mode == "i2v" else []
    return {
        "nodes": top,
        "links": [],
        "definitions": {
            "subgraphs": [
                {
                    "nodes": nodes,
                    "links": high_links + low_links,
                    "inputs": [
                        {"name": "high_lora", "linkIds": [high_widget]},
                        {"name": "low_lora", "linkIds": [low_widget]},
                    ],
                    "outputs": [],
                }
            ]
        },
        "extra": {},
    }


def hunyuan_source(mode="i2v"):
    dimension_type = (
        "HunyuanVideo15ImageToVideo"
        if mode == "i2v"
        else "EmptyHunyuanVideo15Latent"
    )
    model = "hunyuanvideo1.5_720p_{}_fp16.safetensors".format(mode)
    loader = node(2, "UNETLoader", [model, "default"])
    loader["properties"]["models"] = [
        {
            "directory": "diffusion_models",
            "name": model,
            "url": "https://example.invalid/old-model",
        }
    ]
    bypassed = node(
        7,
        "UNETLoader",
        ["hunyuanvideo1.5_1080p_sr_distilled_fp16.safetensors", "default"],
    )
    bypassed["mode"] = 4
    return {
        "nodes": [
            node(1, "VAELoader", ["old-vae.safetensors"]),
            loader,
            node(3, dimension_type, [1280, 720, 121, 1]),
            node(4, "BasicScheduler", ["simple", 20, 1]),
            node(5, "CFGGuider", [6]),
            node(6, "ModelSamplingSD3", [7]),
            bypassed,
        ],
        "links": [],
        "extra": {},
    }


def ltx_source():
    camera = node(
        5,
        "LoraLoaderModelOnly",
        ["old-camera-control.safetensors", 1.0],
    )
    camera["mode"] = 4
    camera["properties"]["models"] = [
        {
            "directory": "loras",
            "name": "ltx-2-19b-lora-camera-control-dolly-left.safetensors",
            "url": "https://example.invalid/old-camera",
        }
    ]
    return {
        "nodes": [
            node(1, "CheckpointLoaderSimple", ["old-checkpoint.safetensors"]),
            node(2, "LTXVAudioVAELoader", ["old-checkpoint.safetensors"]),
            node(
                3,
                "LTXAVTextEncoderLoader",
                ["old-text-encoder.safetensors", "old-checkpoint.safetensors"],
            ),
            node(4, "LatentUpscaleModelLoader", ["old-upscaler.safetensors"]),
            camera,
        ],
        "links": [],
        "extra": {},
    }


def workflow_pack(identifier, renderer, rendered_sha256="a" * 64):
    resource = (
        "templates/image_qwen_image_edit_2511.json"
        if renderer.startswith("qwen-edit")
        else "templates/image_qwen_Image_2512.json"
    )
    return WorkflowPack(
        identifier=identifier,
        description="Test workflow",
        destination=identifier + ".json",
        source_package="comfyui_workflow_templates_json",
        source_version="0.1.6",
        source_revision="c" * 40,
        source_resource=resource,
        source_sha256="b" * 64,
        rendered_sha256=rendered_sha256,
        renderer=renderer,
        license="MIT",
        license_url="https://example.invalid/license",
    )


def all_nodes(workflow):
    result = list(workflow.get("nodes", []))
    for graph in workflow.get("definitions", {}).get("subgraphs", []):
        result.extend(graph.get("nodes", []))
    return result


class WorkflowTests(unittest.TestCase):
    def test_qwen_base_and_lightning_are_distinct_and_attributed(self):
        base_pack = workflow_pack("qwen-base", "qwen-image-base")
        base = json.loads(render_workflow(base_pack, qwen_source()))
        nodes = all_nodes(base)
        self.assertFalse(any(n["type"] == "LoraLoaderModelOnly" for n in nodes))
        sampler = next(n for n in nodes if n["type"] == "KSampler")
        self.assertEqual(sampler["widgets_values"][2:4], [20, 2.5])
        self.assertEqual(base["extra"]["rocmplete"]["license"], "MIT")
        self.assertIn(
            "Copyright (c) 2023-present Comfy Org",
            base["extra"]["rocmplete"]["license_notice"],
        )

        lightning_pack = workflow_pack(
            "qwen-lightning", "qwen-image-lightning"
        )
        lightning = json.loads(
            render_workflow(lightning_pack, qwen_source())
        )
        nodes = all_nodes(lightning)
        lora = next(n for n in nodes if n["type"] == "LoraLoaderModelOnly")
        self.assertEqual(
            lora["widgets_values"][0],
            "Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors",
        )
        sampler = next(n for n in nodes if n["type"] == "KSampler")
        self.assertEqual(sampler["widgets_values"][2:4], [4, 1.0])

    def test_edit_and_i2v_workflows_do_not_reference_sample_media(self):
        edit = json.loads(
            render_workflow(
                workflow_pack("qwen-edit", "qwen-edit-base"),
                qwen_source(edit=True),
            )
        )
        load = next(n for n in all_nodes(edit) if n["type"] == "LoadImage")
        self.assertEqual(load["widgets_values"][0], "")

        i2v = json.loads(
            render_workflow(
                workflow_pack("wan-i2v", "wan-i2v-lightning"),
                wan_source("i2v"),
            )
        )
        load = next(n for n in all_nodes(i2v) if n["type"] == "LoadImage")
        self.assertEqual(load["widgets_values"][0], "")

    def test_wan_base_and_lightning_sampler_settings(self):
        base = json.loads(
            render_workflow(
                workflow_pack("wan-base", "wan-t2v-base"),
                wan_source("t2v"),
            )
        )
        nodes = all_nodes(base)
        self.assertFalse(any(n["type"] == "LoraLoaderModelOnly" for n in nodes))
        samplers = [n for n in nodes if n["type"] == "KSamplerAdvanced"]
        self.assertEqual(
            sorted(n["widgets_values"][7] for n in samplers), [0, 10]
        )
        self.assertTrue(
            all(n["widgets_values"][3:5] == [20, 3.5] for n in samplers)
        )

        lightning = json.loads(
            render_workflow(
                workflow_pack("wan-lightning", "wan-t2v-lightning"),
                wan_source("t2v"),
            )
        )
        nodes = all_nodes(lightning)
        samplers = [n for n in nodes if n["type"] == "KSamplerAdvanced"]
        self.assertTrue(
            all(n["widgets_values"][3:5] == [4, 1.0] for n in samplers)
        )
        text = json.dumps(lightning)
        self.assertIn("lora_v1.1_high_noise.safetensors", text)
        self.assertIn("lora_v1.1_low_noise.safetensors", text)

        lightning_v2 = json.loads(
            render_workflow(
                workflow_pack("wan-lightning-v2", "wan-t2v-lightning-v2"),
                wan_source("t2v"),
            )
        )
        text = json.dumps(lightning_v2)
        self.assertIn("lora_v2_high_noise.safetensors", text)
        self.assertIn("lora_v2_low_noise.safetensors", text)
        self.assertNotIn("lora_v1.1_", text)

    def test_ltx_full_workflow_preserves_bypassed_camera_loader(self):
        rendered = json.loads(
            render_workflow(
                workflow_pack("ltx-full", "ltx-full"),
                ltx_source(),
            )
        )
        camera = next(
            item
            for item in all_nodes(rendered)
            if item["type"] == "LoraLoaderModelOnly"
        )
        self.assertEqual(camera["mode"], 4)
        self.assertEqual(
            camera["widgets_values"][0],
            "ltx-2-19b-lora-camera-control-dolly-left.safetensors",
        )
        self.assertIn(
            "/resolve/75cdba2244db6e2934d06095dd8bb6efa33006f6/",
            camera["properties"]["models"][0]["url"],
        )

    def test_hunyuan_accelerated_workflows_rewrite_model_and_metadata(self):
        cases = (
            (
                "hunyuan-i2v-480p-step-distilled",
                "i2v",
                "hunyuanvideo1.5_480p_i2v_step_distilled_fp16.safetensors",
                12,
                7,
            ),
            (
                "hunyuan-t2v-480p-cfg-distilled",
                "t2v",
                "hunyuanvideo1.5_480p_t2v_cfg_distilled_fp16.safetensors",
                50,
                5,
            ),
        )
        for renderer, mode, model, steps, shift in cases:
            with self.subTest(renderer=renderer):
                rendered = json.loads(
                    render_workflow(
                        workflow_pack("hunyuan-test", renderer),
                        hunyuan_source(mode),
                    )
                )
                nodes = all_nodes(rendered)
                loader = next(
                    item
                    for item in nodes
                    if item["type"] == "UNETLoader"
                    and item.get("mode", 0) != 4
                )
                self.assertEqual(loader["widgets_values"][0], model)
                self.assertEqual(
                    loader["properties"]["models"][0]["name"], model
                )
                self.assertIn(
                    "/resolve/2fa8a1a2d48fe2837368eb64b400217e595ab117/",
                    loader["properties"]["models"][0]["url"],
                )
                dimensions = next(
                    item
                    for item in nodes
                    if item["type"]
                    in (
                        "HunyuanVideo15ImageToVideo",
                        "EmptyHunyuanVideo15Latent",
                    )
                )
                self.assertEqual(dimensions["widgets_values"][:2], [832, 480])
                scheduler = next(
                    item for item in nodes if item["type"] == "BasicScheduler"
                )
                guider = next(
                    item for item in nodes if item["type"] == "CFGGuider"
                )
                sampling = next(
                    item for item in nodes if item["type"] == "ModelSamplingSD3"
                )
                self.assertEqual(scheduler["widgets_values"][1], steps)
                self.assertEqual(guider["widgets_values"][0], 1)
                self.assertEqual(sampling["widgets_values"][0], shift)

    def test_install_is_idempotent_and_protects_modified_workflow(self):
        source = qwen_source()
        initial_pack = workflow_pack("qwen-base", "qwen-image-base")
        digest = hashlib.sha256(render_workflow(initial_pack, source)).hexdigest()
        pack = workflow_pack("qwen-base", "qwen-image-base", digest)
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            with patch(
                "rocmplete.workflows.fetch_source", return_value=source
            ) as fetch_source:
                with redirect_stdout(io.StringIO()):
                    destination = install_workflow(pack, data_dir, "image")
                    self.assertEqual(fetch_source.call_count, 1)
                    original = destination.read_bytes()
                    self.assertEqual(
                        install_workflow(pack, data_dir, "image"), destination
                    )
                    self.assertEqual(fetch_source.call_count, 1)
                    destination.write_text("user modification")
                    with self.assertRaisesRegex(LauncherError, "--force"):
                        install_workflow(pack, data_dir, "image")
                    self.assertEqual(fetch_source.call_count, 1)
                    install_workflow(pack, data_dir, "image", force=True)
                    self.assertEqual(fetch_source.call_count, 2)
                self.assertEqual(destination.read_bytes(), original)

    def test_install_rejects_symlinked_curated_workflow_parent(self):
        source = qwen_source()
        initial_pack = workflow_pack("qwen-base", "qwen-image-base")
        digest = hashlib.sha256(render_workflow(initial_pack, source)).hexdigest()
        pack = workflow_pack("qwen-base", "qwen-image-base", digest)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "data"
            workflows = (
                data_dir
                / "apps"
                / "comfyui"
                / "user"
                / "default"
                / "workflows"
            )
            workflows.mkdir(parents=True)
            external = root / "external"
            external.mkdir()
            (workflows / "curated").symlink_to(
                external, target_is_directory=True
            )

            with patch(
                "rocmplete.workflows.fetch_source", return_value=source
            ) as fetch_source:
                with self.assertRaisesRegex(
                    LauncherError, "symlinked curated workflow path component"
                ):
                    install_workflow(pack, data_dir, "image")

            fetch_source.assert_not_called()
            self.assertEqual(tuple(external.iterdir()), ())

    def test_source_command_has_no_network_or_data_mount(self):
        command = source_command(
            "localhost/image",
            workflow_pack("qwen-base", "qwen-image-base"),
        )
        self.assertIn("none", command)
        self.assertIn("--read-only", command)
        self.assertIn("no-new-privileges", command)
        self.assertNotIn("--volume", command)
        self.assertIn("0.1.6", command)


if __name__ == "__main__":
    unittest.main()
