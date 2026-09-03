"""Stable tensor contracts between model runners and model implementations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from x_tokens.distributed import ParallelGroupKind
from x_tokens.kv_cache import KVCacheMetadata


@dataclass(frozen=True, slots=True)
class ModelInput:
    input_ids: Any | None
    attention_mask: Any
    position_ids: Any
    kv_cache_metadata: tuple[KVCacheMetadata, ...]


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
