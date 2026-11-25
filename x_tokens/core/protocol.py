"""Request slices shared by the scheduler and executor."""

from __future__ import annotations

from dataclasses import dataclass

from ..engine.protocol import EngineCoreRequest, FinishReason


@dataclass(frozen=True, slots=True)
class SchedulerRequest:
    request: EngineCoreRequest
    block_ids: tuple[int, ...]

    @property
    def request_id(self) -> str:
        return self.request.request_id


@dataclass(frozen=True, slots=True)
class SchedulerOutput:
    running_requests: tuple[SchedulerRequest, ...]
    num_scheduled_tokens: dict[str, int]

    @property
    def num_samples(self) -> int:
        return sum(
            self.num_scheduled_tokens[item.request_id]
            == item.request.num_uncomputed_tokens
            for item in self.running_requests
        )


@dataclass(frozen=True, slots=True)
class SchedulerUpdate:
    request: EngineCoreRequest
    token_id: int
    finish_reason: FinishReason | None
