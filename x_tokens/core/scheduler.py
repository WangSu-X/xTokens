"""Request schedulers for the reference and KV-cache execution paths."""

from __future__ import annotations

import logging
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from x_tokens.config import KVCacheConfig
from x_tokens.kv_cache import KVCacheMetadata

from ..engine.types import FinishReason, GenerateRequest
from ..logger import init_logger
from .kv_cache_manager import KVCacheCapacityError, KVCacheManager

logger = init_logger(__name__)


class RequestStatus(StrEnum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"
    ABORTED = "aborted"


class SchedulerPhase(StrEnum):
    FULL = "full"
    PREFILL = "prefill"
    DECODE = "decode"


@dataclass(slots=True)
class ScheduledRequest:
    request: GenerateRequest
    prompt_token_ids: tuple[int, ...]
    output_token_ids: list[int] = field(default_factory=list)
    decoded_text: str = ""
    status: RequestStatus = RequestStatus.WAITING
    needs_prefill: bool = True

    @property
    def request_id(self) -> str:
        return self.request.request_id

    @property
    def context_token_ids(self) -> tuple[int, ...]:
        return self.prompt_token_ids + tuple(self.output_token_ids)

    @property
    def prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def completion_tokens(self) -> int:
        return len(self.output_token_ids)


@dataclass(frozen=True, slots=True)
class SchedulerOutput:
    requests: tuple[ScheduledRequest, ...]
    phase: SchedulerPhase = SchedulerPhase.FULL
    kv_cache_metadata: tuple[KVCacheMetadata, ...] = ()

    def __post_init__(self) -> None:
        if self.kv_cache_metadata and len(self.kv_cache_metadata) != len(self.requests):
            raise ValueError("KV cache metadata batch size mismatch")


@dataclass(frozen=True, slots=True)
class SchedulerUpdate:
    request: ScheduledRequest
    token_id: int
    finish_reason: FinishReason | None


class Scheduler(Protocol):
    """Scheduling contract consumed by ``EngineCore``."""

    @property
    def has_work(self) -> bool: ...

    def add_request(
        self, request: GenerateRequest, prompt_token_ids: tuple[int, ...]
    ) -> None: ...

    def schedule(self) -> SchedulerOutput: ...

    def update_from_output(
        self,
        batch: SchedulerOutput,
        token_ids: tuple[int, ...],
        *,
        eos_token_ids: frozenset[int],
    ) -> tuple[SchedulerUpdate, ...]: ...

    def fail_batch(self, batch: SchedulerOutput) -> tuple[ScheduledRequest, ...]: ...

    def abort(self, request_id: str) -> bool: ...

    def abort_all(self) -> tuple[ScheduledRequest, ...]: ...


class NaiveScheduler(Scheduler):
    """Full-context scheduler retained only for the ``naive_hf`` oracle."""

    def __init__(self, *, max_num_seqs: int, max_model_len: int) -> None:
        self._max_num_seqs = max_num_seqs
        self._max_model_len = max_model_len
        self.waiting: deque[ScheduledRequest] = deque()
        self.running: OrderedDict[str, ScheduledRequest] = OrderedDict()
        self._request_ids: set[str] = set()

    @property
    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    def add_request(
        self, request: GenerateRequest, prompt_token_ids: tuple[int, ...]
    ) -> None:
        if request.request_id in self._request_ids:
            raise ValueError("duplicate request ID")
        if not prompt_token_ids:
            raise ValueError("prompt must contain at least one token")
        if len(prompt_token_ids) >= self._max_model_len:
            raise ValueError("prompt exceeds max_model_len")
        self.waiting.append(ScheduledRequest(request, prompt_token_ids))
        self._request_ids.add(request.request_id)
        logger.debug(
            "scheduler request queued: request_id=%s prompt_tokens=%d "
            "max_tokens=%d waiting=%d running=%d",
            request.request_id,
            len(prompt_token_ids),
            request.sampling.max_tokens,
            len(self.waiting),
            len(self.running),
        )

    def abort(self, request_id: str) -> bool:
        request = self.running.pop(request_id, None)
        if request is not None:
            request.status = RequestStatus.ABORTED
            self._request_ids.discard(request_id)
            logger.debug(
                "scheduler request aborted: request_id=%s state=running", request_id
            )
            return True
        for request in tuple(self.waiting):
            if request.request_id == request_id:
                self.waiting.remove(request)
                request.status = RequestStatus.ABORTED
                self._request_ids.discard(request_id)
                logger.debug(
                    "scheduler request aborted: request_id=%s state=waiting",
                    request_id,
                )
                return True
        return False

    def schedule(self) -> SchedulerOutput:
        admitted = 0
        while self.waiting and len(self.running) < self._max_num_seqs:
            request = self.waiting.popleft()
            request.status = RequestStatus.RUNNING
            self.running[request.request_id] = request
            admitted += 1
        batch = SchedulerOutput(tuple(self.running.values()))
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "scheduler step: batch_size=%d admitted=%d waiting=%d running=%d "
                "request_ids=%s",
                len(batch.requests),
                admitted,
                len(self.waiting),
                len(self.running),
                tuple(request.request_id for request in batch.requests),
            )
        return batch

    def update_from_output(
        self,
        batch: SchedulerOutput,
        token_ids: tuple[int, ...],
        *,
        eos_token_ids: frozenset[int],
    ) -> tuple[SchedulerUpdate, ...]:
        if len(batch.requests) != len(token_ids):
            raise ValueError("batch size and token output size differ")
        updates: list[SchedulerUpdate] = []
        for request, token_id in zip(batch.requests, token_ids, strict=True):
            if self.running.get(request.request_id) is not request:
                continue
            request.output_token_ids.append(token_id)
            finish_reason: FinishReason | None = None
            if token_id in eos_token_ids and not request.request.sampling.ignore_eos:
                finish_reason = FinishReason.STOP
            elif (
                request.completion_tokens >= request.request.sampling.max_tokens
                or len(request.context_token_ids) >= self._max_model_len
            ):
                finish_reason = FinishReason.LENGTH
            if finish_reason is not None:
                request.status = RequestStatus.FINISHED
                self.running.pop(request.request_id)
                self._request_ids.discard(request.request_id)
                logger.debug(
                    "scheduler request finished: request_id=%s finish_reason=%s "
                    "completion_tokens=%d",
                    request.request_id,
                    finish_reason.value,
                    request.completion_tokens,
                )
            updates.append(SchedulerUpdate(request, token_id, finish_reason))
        return tuple(updates)

    def fail_batch(self, batch: SchedulerOutput) -> tuple[ScheduledRequest, ...]:
        failed: list[ScheduledRequest] = []
        for request in batch.requests:
            if self.running.get(request.request_id) is request:
                request.status = RequestStatus.FINISHED
                self.running.pop(request.request_id)
                self._request_ids.discard(request.request_id)
                failed.append(request)
        return tuple(failed)

    def abort_all(self) -> tuple[ScheduledRequest, ...]:
        requests = tuple(self.waiting) + tuple(self.running.values())
        self.waiting.clear()
        self.running.clear()
        self._request_ids.clear()
        for request in requests:
            request.status = RequestStatus.ABORTED
        return requests


class KVCacheScheduler(NaiveScheduler):
    """Continuous batching scheduler for the custom KV-cache-only executor."""

    def __init__(
        self,
        *,
        max_num_seqs: int,
        max_model_len: int,
        kv_cache_config: KVCacheConfig,
        model_fingerprint: str,
    ) -> None:
        super().__init__(max_num_seqs=max_num_seqs, max_model_len=max_model_len)
        if kv_cache_config.num_gpu_blocks is None:
            raise ValueError("KVCacheScheduler requires resolved num_gpu_blocks")
        self._kv_cache_manager = KVCacheManager(
            kv_cache_config,
            num_blocks=kv_cache_config.num_gpu_blocks,
            max_model_len=max_model_len,
            model_fingerprint=model_fingerprint,
        )
        self._last_phase: SchedulerPhase | None = None

    def schedule(self) -> SchedulerOutput:
        admitted = 0
        while self.waiting and len(self.running) < self._max_num_seqs:
            request = self.waiting[0]
            if not self._kv_cache_manager.admit(
                request.request_id, request.context_token_ids
            ):
                break
            self.waiting.popleft()
            request.status = RequestStatus.RUNNING
            request.needs_prefill = True
            self.running[request.request_id] = request
            admitted += 1

        while True:
            prefill = tuple(
                request for request in self.running.values() if request.needs_prefill
            )
            decode = tuple(
                request
                for request in self.running.values()
                if not request.needs_prefill
            )
            if prefill and (
                self._last_phase is not SchedulerPhase.PREFILL or not decode
            ):
                phase, requests = SchedulerPhase.PREFILL, prefill
            elif decode:
                phase, requests = SchedulerPhase.DECODE, decode
            elif prefill:
                phase, requests = SchedulerPhase.PREFILL, prefill
            else:
                return SchedulerOutput(())

            try:
                metadata = tuple(
                    self._kv_cache_manager.prepare_step(
                        request.request_id,
                        request.context_token_ids,
                        is_prefill=phase is SchedulerPhase.PREFILL,
                    )
                    for request in requests
                )
                break
            except KVCacheCapacityError:
                victim = next(
                    (
                        request
                        for request in reversed(tuple(self.running.values()))
                        if request not in requests or len(requests) > 1
                    ),
                    None,
                )
                if victim is None:
                    raise
                self.running.pop(victim.request_id)
                self._kv_cache_manager.release(victim.request_id)
                victim.status = RequestStatus.WAITING
                victim.needs_prefill = True
                self.waiting.appendleft(victim)

        self._last_phase = phase
        batch = SchedulerOutput(requests, phase, metadata)
        if logger.isEnabledFor(logging.DEBUG):
            stats = self._kv_cache_manager.stats()
            logger.debug(
                "scheduler KV step: phase=%s batch_size=%d admitted=%d "
                "waiting=%d running=%d free_blocks=%d cached_blocks=%d",
                phase.value,
                len(requests),
                admitted,
                len(self.waiting),
                len(self.running),
                stats.free_blocks,
                stats.cached_blocks,
            )
        return batch

    def update_from_output(
        self,
        batch: SchedulerOutput,
        token_ids: tuple[int, ...],
        *,
        eos_token_ids: frozenset[int],
    ) -> tuple[SchedulerUpdate, ...]:
        if not batch.kv_cache_metadata:
            raise ValueError("KVCacheScheduler requires KV cache metadata")
        for request in batch.requests:
            self._kv_cache_manager.commit_step(
                request.request_id, request.context_token_ids
            )
            request.needs_prefill = False
        updates = super().update_from_output(
            batch, token_ids, eos_token_ids=eos_token_ids
        )
        for update in updates:
            if update.finish_reason is not None:
                self._kv_cache_manager.release(update.request.request_id)
        return updates

    def abort(self, request_id: str) -> bool:
        aborted = super().abort(request_id)
        if aborted:
            self._kv_cache_manager.release(request_id)
        return aborted

    def fail_batch(self, batch: SchedulerOutput) -> tuple[ScheduledRequest, ...]:
        failed = super().fail_batch(batch)
        for request in failed:
            self._kv_cache_manager.release(request.request_id)
        return failed

    def abort_all(self) -> tuple[ScheduledRequest, ...]:
        requests = super().abort_all()
        for request in requests:
            self._kv_cache_manager.release(request.request_id)
        return requests

    @property
    def kv_cache_manager(self) -> KVCacheManager:
        return self._kv_cache_manager
