"""Model-runner control-plane contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from x_tokens.config import KVCacheConfig


@dataclass(frozen=True, slots=True)
class SamplingMetadata:
    temperature: float
    top_p: float
    top_k: int | None


@dataclass(frozen=True, slots=True)
class ModelRunnerRequest:
    request_id: str
    scheduled_token_ids: tuple[int, ...]
    num_computed_tokens: int
    block_ids: tuple[int, ...]
    sampling_metadata: SamplingMetadata | None


@dataclass(frozen=True, slots=True)
class ModelRunnerOutput:
    """Sampled tokens keyed by the request that produced each token."""

    sampled_token_ids: dict[str, int] = field(default_factory=dict)


class ModelRunner(Protocol):
    def load_model(self) -> None: ...

    def get_kv_cache_capacity(self, config: KVCacheConfig) -> int: ...

    def initialize_kv_cache(self, num_blocks: int, block_size: int) -> None: ...

    def execute_model(self, requests: tuple[ModelRunnerRequest, ...]) -> None: ...

    def sample_tokens(self) -> ModelRunnerOutput: ...

    def shutdown(self) -> None: ...
