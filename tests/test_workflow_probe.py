import json
import tempfile
import unittest
from pathlib import Path

from tools.workflow_probe import probe


class WorkflowProbeTests(unittest.TestCase):
    def test_ui_workflow_summarizes_packages_modes_and_assets(self):
        workflow = {
            "nodes": [
                {
                    "id": 1,
                    "type": "CoreLoader",
                    "properties": {"cnr_id": "comfy-core", "ver": "1.0"},
                    "widgets_values": ["models/core.safetensors"],
                },
                {
                    "id": 2,
                    "type": "RegistryNode",
                    "mode": 4,
                    "properties": {"cnr_id": "custom-pack", "ver": "2.0"},
                    "widgets_values": ["loras/style.safetensors"],
                },
                {
                    "id": 3,
                    "type": "RepositoryNode",
                    "properties": {
                        "aux_id": "owner/repository",
                        "ver": "a" * 40,
                    },
                    "widgets_values": [],
                },
                {
                    "id": 4,
                    "type": "UnknownNode",
                    "widgets_values": ["not-a-model.txt"],
                },
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "workflow.json"
            path.write_text(json.dumps(workflow))
            result = probe(path)

        self.assertEqual(result["format"], "ui")
        self.assertEqual(result["node_modes"]["active"], 3)
        self.assertEqual(result["node_modes"]["bypassed"], 1)
        self.assertEqual(
            [item["identifier"] for item in result["declared_packages"]],
            ["custom-pack", "owner/repository"],
        )
        self.assertEqual(result["core_node_types"], ["CoreLoader"])
        self.assertEqual(result["unattributed_node_types"], ["UnknownNode"])
        self.assertEqual(
            result["asset_references"],
            ["loras/style.safetensors", "models/core.safetensors"],
        )

    def test_ui_workflow_includes_subgraph_nodes(self):
        workflow = {
            "nodes": [],
            "definitions": {
                "subgraphs": [
                    {
                        "nodes": [
                            {
                                "type": "Nested",
                                "properties": {"cnr_id": "nested-pack"},
                            }
                        ]
                    }
                ]
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "workflow.json"
            path.write_text(json.dumps(workflow))
            result = probe(path)

        self.assertEqual(result["node_count"], 1)
        self.assertEqual(
            result["declared_packages"][0]["identifier"], "nested-pack"
        )

    def test_api_workflow_summarizes_types_and_assets(self):
        workflow = {
            "1": {
                "class_type": "UNETLoader",
                "inputs": {"unet_name": "model.safetensors"},
            },
            "2": {
                "class_type": "KSampler",
                "inputs": {"seed": 1},
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "workflow.json"
            path.write_text(json.dumps(workflow))
            result = probe(path)

        self.assertEqual(result["format"], "api")
        self.assertEqual(result["node_types"], ["KSampler", "UNETLoader"])
        self.assertEqual(result["asset_references"], ["model.safetensors"])


if __name__ == "__main__":
    unittest.main()
