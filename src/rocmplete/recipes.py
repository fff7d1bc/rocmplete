"""Small, runnable content recipes for guided installation and help."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

from .catalog import Bundle, Catalog
from .errors import LauncherError


@dataclass(frozen=True)
class RecipeLaunch:
    application: str
    mode: Optional[str] = None
    preset: Optional[str] = None

    @property
    def command(self) -> str:
        parts = ["./rocmplete", "run", self.application]
        if self.mode:
            parts.append(self.mode)
        if self.preset:
            parts.extend(("--preset", self.preset))
        return " ".join(parts)


@dataclass(frozen=True)
class ContentRecipe:
    identifier: str
    application: str
    description: str
    bundles: Tuple[str, ...]
    launch: RecipeLaunch

    @property
    def next_command(self) -> str:
        return self.launch.command


APPLICATION_RECIPES: Mapping[str, Tuple[ContentRecipe, ...]] = {
    "comfyui": (
        ContentRecipe(
            identifier="image",
            application="comfyui",
            description="Qwen Image FP8, 4-step Lightning",
            bundles=("qwen-image-2512-fp8-lightning",),
            launch=RecipeLaunch("comfyui"),
        ),
        ContentRecipe(
            identifier="edit",
            application="comfyui",
            description="Qwen Image Edit FP8, 4-step Lightning",
            bundles=("qwen-image-edit-2511-fp8-lightning",),
            launch=RecipeLaunch("comfyui"),
        ),
        ContentRecipe(
            identifier="t2v",
            application="comfyui",
            description="Wan 2.2 T2V FP8, 4-step Lightning",
            bundles=("wan-2.2-t2v-14b-fp8-lightning",),
            launch=RecipeLaunch("comfyui"),
        ),
        ContentRecipe(
            identifier="i2v",
            application="comfyui",
            description="Wan 2.2 I2V FP8, 4-step Lightning",
            bundles=("wan-2.2-i2v-14b-fp8-lightning",),
            launch=RecipeLaunch("comfyui"),
        ),
    ),
    "llama-cpp": (
        ContentRecipe(
            identifier="qwen3.6",
            application="llama-cpp",
            description=(
                "Qwen3.6 dense 27B MTP Q8_0 and sparse 35B-A3B MTP "
                "Dynamic Q8_K_XL"
            ),
            bundles=(
                "llama-qwen3.6-27b-mtp-q8-0",
                "llama-qwen3.6-35b-a3b-mtp-ud-q8-k-xl",
            ),
            launch=RecipeLaunch(
                "llama-cpp",
                mode="server",
                preset="qwen3.6-27b-mtp-q8-0",
            ),
        ),
        ContentRecipe(
            identifier="ornith",
            application="llama-cpp",
            description="Ornith 1.0 35B Q8_0 coding and agent model",
            bundles=("llama-ornith-1.0-35b-q8-0",),
            launch=RecipeLaunch(
                "llama-cpp",
                mode="server",
                preset="ornith-1.0-35b-q8-0",
            ),
        ),
        ContentRecipe(
            identifier="kat-coder",
            application="llama-cpp",
            description="KAT-Coder V2.5 Dev 35B Q8_0 coding and agent model",
            bundles=("llama-kat-coder-v2.5-dev-q8-0",),
            launch=RecipeLaunch(
                "llama-cpp",
                mode="server",
                preset="kat-coder-v2.5-dev-q8-0",
            ),
        ),
        ContentRecipe(
            identifier="laguna-s-2.1",
            application="llama-cpp",
            description="Laguna S 2.1 Q4_K_M coding and agent model",
            bundles=("llama-laguna-s-2.1-q4-k-m",),
            launch=RecipeLaunch(
                "llama-cpp",
                mode="server",
                preset="laguna-s-2.1-q4-k-m",
            ),
        ),
        ContentRecipe(
            identifier="laguna-xs-2.1",
            application="llama-cpp",
            description="Laguna XS 2.1 Q4_K_M coding and agent model",
            bundles=("llama-laguna-xs-2.1-q4-k-m",),
            launch=RecipeLaunch(
                "llama-cpp",
                mode="server",
                preset="laguna-xs-2.1-q4-k-m",
            ),
        ),
        ContentRecipe(
            identifier="muse-glimmer",
            application="llama-cpp",
            description=(
                "Muse Glimmer 30B official Dynamic Q4_K_XL and 17GB "
                "Q4_K_M with DFlash"
            ),
            bundles=(
                "llama-muse-glimmer-30b-kquant-dynamic-q4-k-xl-dflash",
                "llama-muse-glimmer-30b-kquant-17gb-q4-k-m-dflash",
            ),
            launch=RecipeLaunch(
                "llama-cpp",
                mode="server",
                preset="muse-glimmer-30b-kquant-dynamic-q4-k-xl-dflash",
            ),
        ),
        ContentRecipe(
            identifier="translation-hy",
            application="llama-cpp",
            description="Tencent HY-MT1.5 7B Q8_0 multilingual translator",
            bundles=("llama-hy-mt1.5-7b-q8-0",),
            launch=RecipeLaunch(
                "llama-cpp",
                mode="server",
                preset="hy-mt1.5-7b-q8-0",
            ),
        ),
        ContentRecipe(
            identifier="translation-gemma",
            application="llama-cpp",
            description=(
                "TranslateGemma 27B IT Q8_0 manually prompted translator"
            ),
            bundles=("llama-translategemma-27b-it-q8-0",),
            launch=RecipeLaunch(
                "llama-cpp",
                mode="server",
                preset="translategemma-27b-it-q8-0",
            ),
        ),
        ContentRecipe(
            identifier="shisa-v2.1",
            application="llama-cpp",
            description=(
                "Shisa V2.1 Llama 3.3 70B Q8_0 Japanese/English translator"
            ),
            bundles=("llama-shisa-v2.1-llama3.3-70b-q8-0",),
            launch=RecipeLaunch(
                "llama-cpp",
                mode="server",
                preset="shisa-v2.1-llama3.3-70b-q8-0",
            ),
        ),
    ),
    "dwarfstar": (
        ContentRecipe(
            identifier="flash-0731-q2-imatrix",
            application="dwarfstar",
            description=(
                "DeepSeek V4 Flash 0731 chat-v2 imatrix; routed "
                "IQ2_XXS/Q2_K with Q8 attention/shared/output"
            ),
            bundles=("dwarfstar-deepseek-v4-flash-0731-q2-imatrix",),
            launch=RecipeLaunch("dwarfstar", mode="server"),
        ),
    ),
}


def application_recipes(application: str) -> Tuple[ContentRecipe, ...]:
    try:
        return APPLICATION_RECIPES[application]
    except KeyError:
        raise LauncherError(
            "unknown content application {!r}".format(application)
        )


def content_recipe(application: str, identifier: str) -> ContentRecipe:
    for recipe in application_recipes(application):
        if recipe.identifier == identifier:
            return recipe
    raise LauncherError(
        "unknown {} recipe {!r}; choose {}".format(
            application,
            identifier,
            ", ".join(
                recipe.identifier for recipe in application_recipes(application)
            ),
        )
    )


def recipe_bundles(
    catalog: Catalog, recipe: ContentRecipe
) -> Tuple[Bundle, ...]:
    bundles = tuple(catalog.bundle(identifier) for identifier in recipe.bundles)
    wrong_application = tuple(
        bundle.identifier
        for bundle in bundles
        if bundle.application != recipe.application
    )
    if wrong_application:
        raise LauncherError(
            "{} recipe {} contains bundles owned by another application: "
            "{}".format(
                recipe.application,
                recipe.identifier,
                ", ".join(wrong_application),
            )
        )
    return bundles
