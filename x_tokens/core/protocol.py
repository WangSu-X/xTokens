"""Request slices shared by the scheduler and executor."""

from __future__ import annotations

from dataclasses import dataclass

from ..engine.protocol import EngineCoreRequest, FinishReason, SamplingParams


@dataclass(frozen=True, slots=True)
class SchedulerRequest:
    request_id: str
    scheduled_token_ids: tuple[int, ...]
    num_computed_tokens: int
    block_ids: tuple[int, ...]
    sampling_params: SamplingParams | None

    @classmethod
    def from_engine_core_request(
        cls,
        request: EngineCoreRequest,
        block_ids: tuple[int, ...],
        num_scheduled_tokens: int,
    ) -> SchedulerRequest:
        start = request.num_computed_tokens
        return cls(
            request_id=request.request_id,
            scheduled_token_ids=request.all_token_ids[
                start : start + num_scheduled_tokens
            ],
            num_computed_tokens=start,
            block_ids=block_ids,
            sampling_params=(
                request.sampling_params
                if num_scheduled_tokens == request.num_uncomputed_tokens
                else None
            ),
        )


@dataclass(frozen=True, slots=True)
class SchedulerOutput:
    running_requests: tuple[SchedulerRequest, ...]
    num_scheduled_tokens: dict[str, int]

    @property
    def num_samples(self) -> int:
        return sum(item.sampling_params is not None for item in self.running_requests)


@dataclass(frozen=True, slots=True)
class SchedulerUpdate:
    request: EngineCoreRequest
    token_id: int
    finish_reason: FinishReason | None
