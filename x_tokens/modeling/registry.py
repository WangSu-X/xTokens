"""Explicit architecture and model-type registration."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from x_tokens.config import ParallelConfig

from .config import ModelLoadConfig, Qwen3MoeModelConfig
from .loader.source import ResolvedModelSource

ConfigAdapter = Callable[[Mapping[str, Any]], Qwen3MoeModelConfig]


class RunnerFactoryBuilder(Protocol):
    def __call__(
        self,
        source: ResolvedModelSource,
        model_config: Qwen3MoeModelConfig,
        load_config: ModelLoadConfig,
        pad_token_id: int,
    ) -> object: ...


@dataclass(frozen=True, slots=True)
class ModelRegistration:
    architecture: str
    model_type: str
    config_adapter: ConfigAdapter
    runner_factory_builder: RunnerFactoryBuilder


class ModelRegistry:
    def __init__(self) -> None:
        self._architectures: dict[str, ModelRegistration] = {}
        self._model_types: dict[str, list[ModelRegistration]] = {}

    def register(self, registration: ModelRegistration) -> None:
        if registration.architecture in self._architectures:
            raise ValueError(
                f"duplicate model architecture: {registration.architecture}"
            )
        self._architectures[registration.architecture] = registration
        self._model_types.setdefault(registration.model_type, []).append(registration)

    def resolve(self, hf_config: Mapping[str, Any]) -> ModelRegistration:
        architectures = tuple(str(item) for item in hf_config.get("architectures", ()))
        for architecture in architectures:
            registration = self._architectures.get(architecture)
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
        candidates = self._model_types.get(model_type, ())
        if len(candidates) == 1:
            return candidates[0]
        supported = ", ".join(sorted(self._architectures)) or "<none>"
        raise ValueError(
            f"unsupported or ambiguous model architecture; declared={architectures}, "
            f"model_type={model_type!r}, supported={supported}"
        )


@dataclass(frozen=True, slots=True)
class ResolvedModelRegistration:
    source: ResolvedModelSource
    registration: ModelRegistration
    config: Qwen3MoeModelConfig


def resolve_model_registration(
    source: ResolvedModelSource,
    parallel_config: ParallelConfig,
    *,
    registry: ModelRegistry | None = None,
) -> ResolvedModelRegistration:
    with Path(source.config_path).open(encoding="utf-8") as config_file:
        hf_config = json.load(config_file)
    if not isinstance(hf_config, dict):
        raise TypeError("config.json must contain a JSON object")
    selected_registry = registry or DEFAULT_MODEL_REGISTRY
    registration = selected_registry.resolve(hf_config)
    model_config = registration.config_adapter(hf_config)
    model_config.validate_parallel(parallel_config)
    return ResolvedModelRegistration(source, registration, model_config)


DEFAULT_MODEL_REGISTRY = ModelRegistry()


def _create_qwen3_moe_runner_factory(
    source: ResolvedModelSource,
    model_config: Qwen3MoeModelConfig,
    load_config: ModelLoadConfig,
    pad_token_id: int,
) -> object:
    from .models.qwen3_moe import Qwen3MoeRunnerFactory

    return Qwen3MoeRunnerFactory(
        source,
        model_config,
        load_config,
        pad_token_id,
    )


DEFAULT_MODEL_REGISTRY.register(
    ModelRegistration(
        "Qwen3MoeForCausalLM",
        "qwen3_moe",
        Qwen3MoeModelConfig.from_hf_config,
        _create_qwen3_moe_runner_factory,
    )
)
