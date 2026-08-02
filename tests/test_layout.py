import tempfile
import unittest
from pathlib import Path

from rocmplete.layout import StorageLayout


class StorageLayoutTests(unittest.TestCase):
    def test_application_and_content_partitions_are_distinct(self):
        layout = StorageLayout(Path("/srv/rocmplete"))

        self.assertEqual(
            layout.application("comfyui"),
            Path("/srv/rocmplete/apps/comfyui"),
        )
        self.assertEqual(
            layout.comfy_models,
            Path("/srv/rocmplete/content/comfyui/models"),
        )
        self.assertEqual(
            layout.llama_models,
            Path("/srv/rocmplete/content/llama-cpp/models"),
        )
        self.assertEqual(
            layout.llama_benchmarks,
            Path("/srv/rocmplete/apps/llama-cpp/benchmarks"),
        )
        self.assertEqual(
            layout.acceptance_results,
            Path("/srv/rocmplete/apps/acceptance/results"),
        )
        self.assertEqual(
            layout.curated_workflows,
            Path(
                "/srv/rocmplete/apps/comfyui/user/default/"
                "workflows/curated"
            ),
        )
        self.assertEqual(
            layout.imported_workflows,
            Path(
                "/srv/rocmplete/apps/comfyui/user/default/"
                "workflows/imported"
            ),
        )
        self.assertEqual(
            layout.staging_for("comfyui"),
            Path("/srv/rocmplete/staging/comfyui"),
        )

    def test_runtime_preparation_creates_only_required_content(self):
        with tempfile.TemporaryDirectory() as directory:
            layout = StorageLayout(Path(directory))
            layout.prepare_runtime("comfyui")

            self.assertTrue(layout.application("comfyui").is_dir())
            self.assertTrue(layout.comfy_models.is_dir())
            self.assertFalse(layout.llama_models.exists())

            layout.prepare_runtime("llama-cpp")
            self.assertTrue(layout.llama_models.is_dir())


if __name__ == "__main__":
    unittest.main()
