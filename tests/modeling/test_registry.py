from __future__ import annotations

import pytest

from x_tokens.modeling.registry import DEFAULT_MODEL_REGISTRY


def test_registry_prefers_declared_architecture() -> None:
    registration = DEFAULT_MODEL_REGISTRY.resolve(
        {
            "architectures": ["Qwen3MoeForCausalLM"],
            "model_type": "qwen3_moe",
        }
    )

    assert registration.architecture == "Qwen3MoeForCausalLM"
    assert callable(registration.runner_factory_builder)


def test_registry_falls_back_to_unique_model_type() -> None:
    registration = DEFAULT_MODEL_REGISTRY.resolve({"model_type": "qwen3_moe"})

    assert registration.architecture == "Qwen3MoeForCausalLM"


def test_registry_rejects_architecture_model_type_mismatch() -> None:
    with pytest.raises(ValueError, match="expects model_type qwen3_moe"):
        DEFAULT_MODEL_REGISTRY.resolve(
            {
                "architectures": ["Qwen3MoeForCausalLM"],
                "model_type": "qwen3",
            }
        )


def test_registry_rejects_unknown_model() -> None:
    with pytest.raises(ValueError, match="unsupported or ambiguous"):
        DEFAULT_MODEL_REGISTRY.resolve(
            {
                "architectures": ["UnknownForCausalLM"],
                "model_type": "unknown",
            }
        )
