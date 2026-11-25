"""Hardware-worker contract used by executor backends."""

from __future__ import annotations

from typing import Protocol

from x_tokens.config import KVCacheConfig

from .model_runner import ModelRunnerInput, ModelRunnerOutput


class ModelWorker(Protocol):
    def init_device(self) -> None: ...

    def load_model(self) -> None: ...

    def get_kv_cache_capacity(self, config: KVCacheConfig) -> int: ...

    def initialize_kv_cache(self, num_blocks: int, block_size: int) -> None: ...

    def execute_model(self, inputs: ModelRunnerInput) -> ModelRunnerOutput | None: ...

    def execute_and_sample(
        self, inputs: ModelRunnerInput
    ) -> tuple[int, ...] | None: ...

    def sample_tokens(self, step_id: int) -> tuple[int, ...] | None: ...

    def check_health(self) -> None: ...

    def shutdown(self) -> None: ...
