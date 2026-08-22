"""Model-runner control-plane contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from x_tokens.distributed import ParallelContext


@dataclass(frozen=True, slots=True)
class SamplingMetadata:
    temperature: float
    top_p: float
    top_k: int | None


@dataclass(frozen=True, slots=True)
class ModelRunnerInput:
    step_id: int
    token_ids: tuple[tuple[int, ...], ...]
    sampling_metadata: tuple[SamplingMetadata, ...]


@dataclass(frozen=True, slots=True)
class ModelRunnerOutput:
    step_id: int


class ModelRunner(Protocol):
    def load_model(self) -> None: ...

    def execute_model(self, inputs: ModelRunnerInput) -> ModelRunnerOutput | None: ...

    def sample_tokens(self, step_id: int) -> tuple[int, ...] | None: ...

    def shutdown(self) -> None: ...


class ModelRunnerFactory(Protocol):
    def __call__(
        self, context: ParallelContext, *, is_output_rank: bool
    ) -> ModelRunner: ...
