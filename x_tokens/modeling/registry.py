"""Explicit architecture and model-type registration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from x_tokens.distributed import ParallelContext

from .config import ModelLoadConfig
from .protocol import RunnerFactory

if TYPE_CHECKING:
    from x_tokens.worker.model_runner.protocol import ModelRunner


@dataclass(frozen=True, slots=True)
class ModelRegistration:
    architecture: str
    model_type: str
    runner_factory: RunnerFactory


class ModelRegistry:
    def __init__(self) -> None:
        self.architectures: dict[str, ModelRegistration] = {}
        self.model_types: dict[str, list[ModelRegistration]] = {}

    def register(self, registration: ModelRegistration) -> None:
        if registration.architecture in self.architectures:
            raise ValueError(
                f"duplicate model architecture: {registration.architecture}"
            )
        self.architectures[registration.architecture] = registration
        self.model_types.setdefault(registration.model_type, []).append(registration)

    def resolve(self, hf_config: Mapping[str, Any]) -> ModelRegistration:
        architectures = tuple(str(item) for item in hf_config.get("architectures", ()))
        for architecture in architectures:
            registration = self.architectures.get(architecture)
            if registration is None:
                continue
            model_type = str(hf_config.get("model_type", ""))
            if model_type != registration.model_type:
                raise ValueError(
                    f"architecture {architecture} expects model_type "
                    f"{registration.model_type}, got {model_type}"
                )
            return registration
        model_type = str(hf_config.get("model_type", ""))
        candidates = self.model_types.get(model_type, ())
        if len(candidates) == 1:
            return candidates[0]
        supported = ", ".join(sorted(self.architectures)) or "<none>"
        raise ValueError(
            f"unsupported or ambiguous model architecture; declared={architectures}, "
            f"model_type={model_type!r}, supported={supported}"
        )


DEFAULT_MODEL_REGISTRY = ModelRegistry()


def _create_qwen3_5_runner(
    load_config: ModelLoadConfig,
    context: ParallelContext,
    *,
    pad_token_id: int,
    is_output_rank: bool,
) -> ModelRunner:
    from .models.qwen3_5 import create_qwen3_5_runner

    return create_qwen3_5_runner(
        load_config,
        context,
        pad_token_id=pad_token_id,
        is_output_rank=is_output_rank,
    )


for architecture, model_type in (
    ("Qwen3_5ForConditionalGeneration", "qwen3_5"),
    ("Qwen3_5ForCausalLM", "qwen3_5_text"),
):
    DEFAULT_MODEL_REGISTRY.register(
        ModelRegistration(
            architecture,
            model_type,
            _create_qwen3_5_runner,
        )
    )
