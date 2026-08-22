"""Hardware-worker contract used by executor backends."""

from __future__ import annotations

from typing import Protocol

from .model_runner import ModelRunnerInput, ModelRunnerOutput


class ModelWorker(Protocol):
    def init_device(self) -> None: ...

    def load_model(self) -> None: ...

    def execute_model(self, inputs: ModelRunnerInput) -> ModelRunnerOutput | None: ...

    def sample_tokens(self, step_id: int) -> tuple[int, ...] | None: ...

    def check_health(self) -> None: ...

    def shutdown(self) -> None: ...
