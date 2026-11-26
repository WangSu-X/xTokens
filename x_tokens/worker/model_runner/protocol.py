"""Model-runner control-plane contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from x_tokens.config import KVCacheConfig


@dataclass(frozen=True, slots=True)
class SamplingMetadata:
    temperature: float
    top_p: float
    top_k: int | None


@dataclass(frozen=True, slots=True)
class ModelRunnerRequest:
    token_ids: tuple[int, ...]
    num_computed_tokens: int
    block_ids: tuple[int, ...]
    sampling_metadata: SamplingMetadata | None


@dataclass(frozen=True, slots=True)
class ModelRunnerInput:
    step_id: int
    requests: tuple[ModelRunnerRequest, ...]


@dataclass(frozen=True, slots=True)
class ModelRunnerOutput:
    step_id: int


class ModelRunner(Protocol):
    def load_model(self) -> None: ...

    def get_kv_cache_capacity(self, config: KVCacheConfig) -> int: ...

    def initialize_kv_cache(self, num_blocks: int, block_size: int) -> None: ...

    def execute_model(self, inputs: ModelRunnerInput) -> ModelRunnerOutput | None: ...

    def sample_tokens(self, step_id: int) -> tuple[int, ...] | None: ...

    def shutdown(self) -> None: ...
