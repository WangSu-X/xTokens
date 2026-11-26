from __future__ import annotations

import pytest

from x_tokens.modeling.registry import DEFAULT_MODEL_REGISTRY


@pytest.mark.parametrize(
    ("architecture", "model_type"),
    [
        ("Qwen3_5ForCausalLM", "qwen3_5_text"),
        ("Qwen3_5ForConditionalGeneration", "qwen3_5"),
    ],
)
def test_registry_resolves_declared_architecture(
    architecture: str, model_type: str
) -> None:
    registration = DEFAULT_MODEL_REGISTRY.resolve(
        {
            "architectures": [architecture],
            "model_type": model_type,
        }
    )

    assert registration.architecture == architecture
    assert callable(registration.runner_factory)


def test_registry_falls_back_to_unique_model_type() -> None:
    registration = DEFAULT_MODEL_REGISTRY.resolve({"model_type": "qwen3_5"})

    assert registration.architecture == "Qwen3_5ForConditionalGeneration"


def test_registry_rejects_architecture_model_type_mismatch() -> None:
    with pytest.raises(ValueError, match="expects model_type qwen3_5_text"):
        DEFAULT_MODEL_REGISTRY.resolve(
            {
                "architectures": ["Qwen3_5ForCausalLM"],
                "model_type": "qwen3_5",
            }
        )


@pytest.mark.parametrize(
    ("architecture", "model_type"),
    [("UnknownForCausalLM", "unknown"), ("Qwen3MoeForCausalLM", "qwen3_moe")],
)
def test_registry_rejects_unsupported_model(architecture: str, model_type: str) -> None:
    with pytest.raises(ValueError, match="unsupported or ambiguous"):
        DEFAULT_MODEL_REGISTRY.resolve(
            {
                "architectures": [architecture],
                "model_type": model_type,
            }
        )
