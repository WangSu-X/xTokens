"""Model registration and tensor contracts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from x_tokens.distributed import ParallelGroupKind
from x_tokens.kv_cache import KVCacheSlots

from .config import ModelLoadConfig
from .loader.source import ResolvedModelSource
from .qwen3_5_config import Qwen3_5TextConfig


@dataclass(frozen=True, slots=True)
class ModelInput:
    input_ids: Any | None
    attention_mask: Any
    position_ids: Any
    kv_cache_slots: tuple[KVCacheSlots, ...]


@dataclass(frozen=True, slots=True)
class IntermediateTensors:
    tensors: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ModelOutput:
    logits: Any | None
    intermediate_tensors: IntermediateTensors | None = None


@dataclass(frozen=True, slots=True)
class TensorShard:
    group: ParallelGroupKind
    dimension: int


@dataclass(frozen=True, slots=True)
class ParameterShardSpec:
    pipeline_stage: int | None = None
    expert_ids: tuple[int, ...] | None = None
    tensor_shards: tuple[TensorShard, ...] = ()
    replicated: bool = False


class XTokensModel(Protocol):
    def __call__(
        self,
        inputs: ModelInput,
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> ModelOutput: ...


ConfigAdapter = Callable[[Mapping[str, Any]], Qwen3_5TextConfig]


class RunnerFactoryBuilder(Protocol):
    def __call__(
        self,
        source: ResolvedModelSource,
        model_config: Qwen3_5TextConfig,
        load_config: ModelLoadConfig,
        pad_token_id: int,
    ) -> object: ...
