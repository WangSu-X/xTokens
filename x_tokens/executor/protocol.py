"""Backend interface consumed by the backend-independent EngineCore."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from x_tokens.core.protocol import SchedulerOutput
    from x_tokens.worker.model_runner.protocol import ModelRunnerOutput


class Executor(Protocol):
    """Model execution backend consumed by EngineCore."""

    num_gpu_blocks: int

    @property
    def eos_token_ids(self) -> frozenset[int]: ...

    def execute_step(self, batch: SchedulerOutput) -> ModelRunnerOutput: ...


class WorkerHandle(Protocol):
    rank: int

    def send(self, method: str, args: tuple[Any, ...]) -> None: ...

    def recv(self, timeout_s: float) -> Any: ...

    def check_health(self) -> None: ...
