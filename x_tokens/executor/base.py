"""Backend interface consumed by the backend-independent EngineCore."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from x_tokens.config import XTokensConfig

if TYPE_CHECKING:
    from x_tokens.core.scheduler import SchedulerOutput

    from .model_executor import RunnerStepId


class Executor(Protocol):
    """Model execution backend consumed by EngineCore."""

    num_gpu_blocks: int

    @property
    def eos_token_ids(self) -> frozenset[int]: ...

    def execute_model(self, batch: SchedulerOutput) -> ModelForwardOutput: ...

    def sample_tokens(
        self, output: ModelForwardOutput, batch: SchedulerOutput
    ) -> tuple[int, ...]: ...


ExecutorFactory = Callable[[XTokensConfig, int, frozenset[int]], Executor]


@dataclass(frozen=True, slots=True)
class ModelForwardOutput:
    """Handle to rank-local state awaiting device-side sampling."""

    runner_step_id: RunnerStepId | None = None
