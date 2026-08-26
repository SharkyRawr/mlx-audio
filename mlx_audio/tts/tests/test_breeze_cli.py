"""CPU-safe CLI and registry coverage for Breeze TTS 2."""

import pytest

from mlx_audio.registry import SUPPORTED_MODEL_TYPES
from mlx_audio.tts.generate import parse_args
from mlx_audio.tts.models.breeze_tts import Model
from mlx_audio.tts.utils import MODEL_REMAPPING, get_model_and_args
from mlx_audio.utils import get_model_category, get_model_name_parts


def _parse(*options: str):
    return parse_args(
        [
            "--model",
            "BreezeBlue/breeze-tts-2",
            "--text",
            "Hello from Breeze.",
            *options,
        ]
    )


@pytest.mark.parametrize(
    ("established", "upstream", "destination", "value"),
    [
        ("--instruct", "--instruction", "instruct", "A calm, warm voice."),
        ("--cfg_scale", "--cfg-scale", "cfg_scale", "4"),
        ("--ref_audio", "--ref-audio", "ref_audio", "reference.wav"),
        ("--ref_text", "--ref-text", "ref_text", "The reference transcript."),
    ],
)
def test_upstream_flag_aliases_share_existing_destinations(
    established, upstream, destination, value
):
    established_args = _parse(established, value)
    upstream_args = _parse(upstream, value)

    assert getattr(established_args, destination) == getattr(upstream_args, destination)


def test_reference_aliases_remain_repeatable_and_can_be_mixed():
    args = _parse(
        "--ref-audio",
        "first.wav",
        "--ref_audio",
        "second.wav",
        "--ref-text",
        "First transcript.",
        "--ref_text",
        "Second transcript.",
    )

    assert args.ref_audio == ["first.wav", "second.wav"]
    assert args.ref_text == ["First transcript.", "Second transcript."]


@pytest.mark.parametrize("alias", ["breeze", "breeze-tts", "breeze_tts"])
def test_breeze_registry_aliases_resolve_to_one_module(alias):
    module, model_type = get_model_and_args(alias, [])

    assert model_type == "breeze_tts"
    assert module.Model is Model
    assert MODEL_REMAPPING[alias] == "breeze_tts"


def test_breeze_registry_advertises_all_aliases():
    assert {"breeze", "breeze-tts", "breeze_tts"} <= SUPPORTED_MODEL_TYPES["tts"]


def test_breeze_publisher_repository_is_classified_as_tts():
    parts = get_model_name_parts("BreezeBlue/breeze-tts-2")

    assert get_model_category("breeze", parts) == "tts"
