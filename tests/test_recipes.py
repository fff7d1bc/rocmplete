import unittest

from rocmplete.catalog import load_catalog
from rocmplete.errors import LauncherError
from rocmplete.recipes import (
    ContentRecipe,
    RecipeLaunch,
    application_recipes,
    content_recipe,
    recipe_bundles,
)


class ContentRecipeTests(unittest.TestCase):
    def test_recipe_launch_is_the_single_command_source(self):
        comfy = content_recipe("comfyui", "image")
        llama = content_recipe("llama-cpp", "qwen3.6")
        qwen38 = content_recipe("llama-cpp", "qwen3.8")
        kat_coder = content_recipe("llama-cpp", "kat-coder")
        muse = content_recipe("llama-cpp", "muse-glimmer")
        hy = content_recipe("llama-cpp", "translation-hy")
        gemma = content_recipe("llama-cpp", "translation-gemma")
        shisa = content_recipe("llama-cpp", "shisa-v2.1")
        dwarfstar = content_recipe("dwarfstar", "flash-0731-q2-imatrix")

        self.assertEqual(comfy.next_command, "./rocmplete run comfyui")
        self.assertEqual(
            llama.next_command,
            "./rocmplete run llama-cpp server "
            "--preset qwen3.6-27b-mtp-q8-0",
        )
        self.assertEqual(
            llama.bundles,
            ("llama-qwen3.6-27b-mtp-q8-0",),
        )
        self.assertEqual(
            qwen38.next_command,
            "./rocmplete run llama-cpp server "
            "--preset qwen3.8-27b-mtp-ud-q8-k-xl",
        )
        self.assertEqual(
            qwen38.bundles,
            ("llama-qwen3.8-27b-ud-q8-k-xl",),
        )
        self.assertEqual(
            kat_coder.next_command,
            "./rocmplete run llama-cpp server "
            "--preset kat-coder-v2.5-dev-q8-0",
        )
        self.assertEqual(
            kat_coder.bundles,
            ("llama-kat-coder-v2.5-dev-q8-0",),
        )
        self.assertEqual(
            muse.next_command,
            "./rocmplete run llama-cpp server "
            "--preset "
            "muse-glimmer-30b-kquant-dynamic-q4-k-xl-dflash-256k",
        )
        self.assertEqual(
            muse.bundles,
            ("llama-muse-glimmer-30b-kquant-dynamic-q4-k-xl-dflash",),
        )
        self.assertEqual(
            hy.next_command,
            "./rocmplete run llama-cpp server --preset hy-mt1.5-7b-q8-0",
        )
        self.assertEqual(
            gemma.next_command,
            "./rocmplete run llama-cpp server "
            "--preset translategemma-27b-it-q8-0",
        )
        self.assertEqual(
            shisa.next_command,
            "./rocmplete run llama-cpp server "
            "--preset shisa-v2.1-llama3.3-70b-q8-0",
        )
        self.assertEqual(
            shisa.bundles,
            ("llama-shisa-v2.1-llama3.3-70b-q8-0",),
        )
        self.assertEqual(
            dwarfstar.next_command,
            "./rocmplete run dwarfstar server",
        )

    def test_every_recipe_resolves_only_its_application_bundles(self):
        catalog = load_catalog()
        for application in ("comfyui", "llama-cpp", "dwarfstar"):
            for recipe in application_recipes(application):
                with self.subTest(
                    application=application,
                    recipe=recipe.identifier,
                ):
                    bundles = recipe_bundles(catalog, recipe)
                    self.assertTrue(bundles)
                    self.assertTrue(
                        all(
                            bundle.application == application
                            for bundle in bundles
                        )
                    )

    def test_recipe_bundle_validation_rejects_cross_application_content(self):
        recipe = ContentRecipe(
            identifier="invalid",
            application="comfyui",
            description="invalid test recipe",
            bundles=("llama-qwen3-0.6b-q8-0",),
            launch=RecipeLaunch("comfyui"),
        )
        with self.assertRaisesRegex(
            LauncherError, "owned by another application"
        ):
            recipe_bundles(load_catalog(), recipe)


if __name__ == "__main__":
    unittest.main()
