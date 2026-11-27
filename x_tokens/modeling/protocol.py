"""Model registration and tensor contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from torch import Tensor

from x_tokens.distributed import ParallelContext, ParallelGroupKind

from .config import ModelLoadConfig
from .kv_cache import KVCacheSpec

if TYPE_CHECKING:
    from x_tokens.worker.model_runner.protocol import ModelRunner


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
    def get_kv_cache_specs(self, block_size: int) -> Mapping[str, KVCacheSpec]: ...

    def __call__(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
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
