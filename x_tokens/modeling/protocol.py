"""Model registration and tensor contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from x_tokens.distributed import ParallelContext, ParallelGroupKind
from x_tokens.kv_cache import KVCacheSlots

from .config import ModelLoadConfig

if TYPE_CHECKING:
    from x_tokens.worker.model_runner.protocol import ModelRunner


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


class RunnerFactory(Protocol):
    def __call__(
        self,
        load_config: ModelLoadConfig,
        context: ParallelContext,
        *,
        pad_token_id: int,
        is_output_rank: bool,
    ) -> ModelRunner: ...
