"""Request scheduling with block-based KV cache and prefix reuse."""

from __future__ import annotations

import logging
from collections import OrderedDict, deque

from x_tokens.config import KVCacheConfig
from x_tokens.core.protocol import SchedulerOutput, SchedulerRequest, SchedulerUpdate

from ..engine.protocol import EngineCoreRequest, FinishReason, RequestStatus
from ..logger import init_logger
from .kv_cache_manager import KVCacheManager

logger = init_logger(__name__)


class Scheduler:
    """Own request scheduling and logical KV-cache state for EngineCore."""

    def __init__(
        self,
        *,
        max_num_seqs: int,
        max_model_len: int,
        kv_cache_config: KVCacheConfig,
        model_fingerprint: str,
        max_prefill_batch_tokens: int | None = None,
        enable_chunked_prefill: bool = False,
        prefill_chunk_alignment: int | None = None,
    ) -> None:
        if max_num_seqs < 1:
            raise ValueError("max_num_seqs must be positive")
        if max_model_len < 2:
            raise ValueError("max_model_len must be at least 2")
        if max_prefill_batch_tokens is not None and max_prefill_batch_tokens < 1:
            raise ValueError("max_prefill_batch_tokens must be positive")
        if enable_chunked_prefill and (
            max_prefill_batch_tokens is None
            or max_prefill_batch_tokens < kv_cache_config.block_size
        ):
            raise ValueError(
                "chunked prefill requires a budget of at least one KV block"
            )
        self.max_num_seqs = max_num_seqs
        self.max_model_len = max_model_len
        self.token_budget = (
            max_prefill_batch_tokens
            if max_prefill_batch_tokens is not None
            else max_num_seqs * max_model_len
        )
        self.enable_chunked_prefill = enable_chunked_prefill
        if prefill_chunk_alignment is not None and (
            not enable_chunked_prefill
            or max_prefill_batch_tokens is None
            or not 0 < prefill_chunk_alignment <= max_prefill_batch_tokens
            or prefill_chunk_alignment % kv_cache_config.block_size
        ):
            raise ValueError(
                "prefill chunk alignment must be a KV block multiple within the chunked budget"
            )
        self.prefill_chunk_alignment = prefill_chunk_alignment
        self.waiting: deque[EngineCoreRequest] = deque()
        self.running: OrderedDict[str, EngineCoreRequest] = OrderedDict()
        self.request_ids: set[str] = set()
        if kv_cache_config.num_gpu_blocks is None:
            raise ValueError("Scheduler requires resolved num_gpu_blocks")
        self.cache_manager = KVCacheManager(
            kv_cache_config,
            num_blocks=kv_cache_config.num_gpu_blocks,
            max_model_len=max_model_len,
            model_fingerprint=model_fingerprint,
        )

    @property
    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    def add_request(self, request: EngineCoreRequest) -> None:
        if request.request_id in self.request_ids:
            raise ValueError("duplicate request ID")
        if not request.prompt_token_ids:
            raise ValueError("prompt must contain at least one token")
        if request.prompt_tokens >= self.max_model_len:
            raise ValueError("prompt exceeds max_model_len")
        self.waiting.append(request)
        self.request_ids.add(request.request_id)
        logger.debug(
            "scheduler request queued: request_id=%s prompt_tokens=%d "
            "max_tokens=%d waiting=%d running=%d",
            request.request_id,
            request.prompt_tokens,
            request.sampling_params.max_tokens,
            len(self.waiting),
            len(self.running),
        )

    def abort(self, request_id: str) -> bool:
        request = self.running.pop(request_id, None)
        if request is not None:
            request.status = RequestStatus.ABORTED
            request.stop_reason = FinishReason.CANCELLED
            self.request_ids.discard(request_id)
            self.cache_manager.free(request_id)
            logger.debug(
                "scheduler request aborted: request_id=%s state=running", request_id
            )
            return True
        for request in tuple(self.waiting):
            if request.request_id == request_id:
                self.waiting.remove(request)
                request.status = RequestStatus.ABORTED
                request.stop_reason = FinishReason.CANCELLED
                self.request_ids.discard(request_id)
                logger.debug(
                    "scheduler request aborted: request_id=%s state=waiting",
                    request_id,
                )
                return True
        return False

    def schedule(self) -> SchedulerOutput:
        budget = self.token_budget
        scheduled_requests: list[SchedulerRequest] = []
        num_scheduled_tokens: dict[str, int] = {}
        preempted = False
        competing = len(self.running) + len(self.waiting) > 1

        for request in tuple(self.running.values()):
            if budget <= 0 or request.request_id not in self.running:
                break
            count = self._tokens_to_schedule(
                request.num_uncomputed_tokens, budget, competing
            )
            if not count:
                continue
            new_blocks = self.cache_manager.allocate_slots(request, count)
            while new_blocks is None:
                _, victim = self.running.popitem(last=True)
                self.cache_manager.free(victim.request_id)
                victim.status = RequestStatus.WAITING
                victim.num_computed_tokens = 0
                victim.block_hashes.clear()
                self.waiting.appendleft(victim)
                preempted = True
                logger.info(
                    "scheduler preemption: request_id=%s running=%d waiting=%d",
                    victim.request_id,
                    len(self.running),
                    len(self.waiting),
                )
                if victim is request:
                    break
                new_blocks = self.cache_manager.allocate_slots(request, count)
            if new_blocks is None:
                break
            block_ids = self.cache_manager.get_block_ids(request.request_id)
            scheduled_requests.append(SchedulerRequest(request, block_ids))
            num_scheduled_tokens[request.request_id] = count
            budget -= count

        admitted = 0
        while (
            not preempted
            and budget > 0
            and self.waiting
            and len(self.running) < self.max_num_seqs
        ):
            request = self.waiting[0]
            computed_blocks = self.cache_manager.get_computed_blocks(request)
            cached_tokens = len(computed_blocks) * self.cache_manager.block_size
            remaining = request.num_uncomputed_tokens - cached_tokens
            count = self._tokens_to_schedule(remaining, budget, competing)
            if not count:
                break
            new_blocks = self.cache_manager.allocate_slots(
                request, count, computed_blocks=computed_blocks
            )
            if new_blocks is None:
                break
            self.waiting.popleft()
            request.status = RequestStatus.RUNNING
            self.running[request.request_id] = request
            block_ids = self.cache_manager.get_block_ids(request.request_id)
            scheduled_requests.append(SchedulerRequest(request, block_ids))
            num_scheduled_tokens[request.request_id] = count
            budget -= count
            admitted += 1

        if logger.isEnabledFor(logging.DEBUG):
            stats = self.cache_manager.stats()
            logger.debug(
                "scheduler step: batch_size=%d scheduled_tokens=%d admitted=%d "
                "waiting=%d running=%d free_blocks=%d cached_blocks=%d",
                len(scheduled_requests),
                sum(num_scheduled_tokens.values()),
                admitted,
                len(self.waiting),
                len(self.running),
                stats.free_blocks,
                stats.cached_blocks,
            )
        return SchedulerOutput(tuple(scheduled_requests), num_scheduled_tokens)

    def _tokens_to_schedule(self, remaining: int, budget: int, competing: bool) -> int:
        if not self.enable_chunked_prefill:
            # Let an unchunked request larger than the budget run by itself.
            return (
                remaining if remaining <= budget or budget == self.token_budget else 0
            )
        count = min(remaining, budget)
        alignment = self.prefill_chunk_alignment
        if (
            competing
            and alignment is not None
            and remaining > alignment
            and count >= alignment
        ):
            count -= count % alignment
        return count

    def update_from_output(
        self,
        batch: SchedulerOutput,
        token_ids: tuple[int, ...],
        *,
        eos_token_ids: frozenset[int],
    ) -> tuple[SchedulerUpdate, ...]:
        if batch.num_samples != len(token_ids):
            raise ValueError("batch size and token output size differ")
        updates: list[SchedulerUpdate] = []
        samples = iter(token_ids)
        for item in batch.running_requests:
            request = item.request
            count = batch.num_scheduled_tokens[item.request_id]
            token_id = next(samples) if count == request.num_uncomputed_tokens else None
            if self.running.get(request.request_id) is not request:
                continue
            num_computed_tokens = request.num_computed_tokens + count
            self.cache_manager.cache_blocks(request, num_computed_tokens)
            request.num_computed_tokens = num_computed_tokens
            if token_id is None:
                continue
            request.output_token_ids.append(token_id)
            if token_id in eos_token_ids and not request.sampling_params.ignore_eos:
                request.stop_reason = FinishReason.STOP
            elif (
                request.completion_tokens >= request.sampling_params.max_tokens
                or request.prompt_tokens + request.completion_tokens
                >= self.max_model_len
            ):
                request.stop_reason = FinishReason.LENGTH
            if request.stop_reason is not None:
                request.status = RequestStatus.FINISHED
                self.running.pop(request.request_id)
                self.request_ids.discard(request.request_id)
                self.cache_manager.free(request.request_id)
                logger.debug(
                    "scheduler request finished: request_id=%s finish_reason=%s "
                    "completion_tokens=%d",
                    request.request_id,
                    request.stop_reason.value,
                    request.completion_tokens,
                )
            updates.append(SchedulerUpdate(request, token_id, request.stop_reason))
        return tuple(updates)

    def fail_batch(self, batch: SchedulerOutput) -> tuple[EngineCoreRequest, ...]:
        failed: list[EngineCoreRequest] = []
        for item in batch.running_requests:
            request = item.request
            if self.running.get(request.request_id) is request:
                request.status = RequestStatus.FINISHED
                self.running.pop(request.request_id)
                self.request_ids.discard(request.request_id)
                self.cache_manager.free(request.request_id)
                failed.append(request)
        return tuple(failed)

    def abort_all(self) -> tuple[EngineCoreRequest, ...]:
        requests = tuple(self.waiting) + tuple(self.running.values())
        self.waiting.clear()
        self.running.clear()
        self.request_ids.clear()
        for request in requests:
            request.status = RequestStatus.ABORTED
            request.stop_reason = FinishReason.CANCELLED
            self.cache_manager.free(request.request_id)
        return requests

    @property
    def kv_cache_manager(self) -> KVCacheManager:
        return self.cache_manager
