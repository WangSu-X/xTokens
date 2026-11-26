"""Backend interface consumed by the backend-independent EngineCore."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from x_tokens.core.protocol import SchedulerOutput


class Executor(Protocol):
    """Model execution backend consumed by EngineCore."""

    num_gpu_blocks: int

    @property
    def eos_token_ids(self) -> frozenset[int]: ...

    def execute_model(self, batch: SchedulerOutput) -> ModelForwardOutput: ...

    def sample_tokens(
        self, output: ModelForwardOutput, batch: SchedulerOutput
    ) -> tuple[int, ...]: ...


@dataclass(frozen=True, slots=True)
class RunnerStepId:
    runner_group: int
    step_id: int


@dataclass(frozen=True, slots=True)
class ModelForwardOutput:
    """Handle to rank-local state awaiting device-side sampling."""

    runner_step_id: RunnerStepId | None = None


class WorkerHandle(Protocol):
    rank: int

    def send(self, method: str, args: tuple[Any, ...]) -> None: ...

    def recv(self, timeout_s: float) -> Any: ...

    def check_health(self) -> None: ...
