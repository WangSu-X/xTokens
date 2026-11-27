from __future__ import annotations

from unittest.mock import Mock

import pytest

from x_tokens.config import KVCacheConfig
from x_tokens.core import scheduler as scheduler_module
from x_tokens.core.scheduler import RequestStatus, Scheduler
from x_tokens.engine.protocol import EngineCoreRequest, FinishReason, SamplingParams


def request(
    request_id: str,
    prompt_token_ids: tuple[int, ...],
    *,
    max_tokens: int = 2,
    ignore_eos: bool = False,
) -> EngineCoreRequest:
    return EngineCoreRequest(
        request_id,
        SamplingParams(max_tokens=max_tokens, ignore_eos=ignore_eos),
        prompt_token_ids,
    )


def sampled_tokens(batch, token_ids: tuple[int, ...]) -> dict[str, int]:
    return {
        item.request_id: token_id
        for item, token_id in zip(
            (
                item
                for item in batch.running_requests
                if item.sampling_params is not None
            ),
            token_ids,
            strict=True,
        )
    }


def test_scheduler_admits_requests_in_fifo_batches() -> None:
    scheduler = Scheduler(
        kv_cache_config=KVCacheConfig(block_size=2, num_gpu_blocks=8),
        model_fingerprint="tiny",
        max_num_seqs=2,
        max_model_len=8,
    )
    for request_id in ("one", "two", "three"):
        scheduler.add_request(request(request_id, (1,)))

    first_batch = scheduler.schedule()
    assert [item.request_id for item in first_batch.running_requests] == ["one", "two"]

    updates = scheduler.update_from_output(
        first_batch, sampled_tokens(first_batch, (2, 3)), eos_token_ids=frozenset()
    )
    assert all(update.finish_reason is None for update in updates)

    second_batch = scheduler.schedule()
    assert [item.request_id for item in second_batch.running_requests] == ["one", "two"]
    scheduler.update_from_output(
        second_batch, sampled_tokens(second_batch, (4, 5)), eos_token_ids=frozenset()
    )

    third_batch = scheduler.schedule()
    assert [item.request_id for item in third_batch.running_requests] == ["three"]


def test_prefill_batch_budget_counts_only_uncached_tokens() -> None:
    scheduler = Scheduler(
        kv_cache_config=KVCacheConfig(block_size=2, num_gpu_blocks=16),
        model_fingerprint="tiny",
        max_num_seqs=4,
        max_model_len=10,
        max_prefill_batch_tokens=4,
    )
    prefix = (1, 2, 3, 4)
    scheduler.add_request(request("prime", (*prefix, 5), max_tokens=1))
    batch = scheduler.schedule()
    scheduler.update_from_output(
        batch, sampled_tokens(batch, (9,)), eos_token_ids=frozenset()
    )
    scheduler.add_request(request("cold", (10, 11)))
    scheduler.add_request(request("warm", (*prefix, 6, 7)))
    scheduler.add_request(request("later", (*prefix, 8)))

    batch = scheduler.schedule()
    assert [item.request_id for item in batch.running_requests] == ["cold", "warm"]
    assert batch.num_scheduled_tokens == {"cold": 2, "warm": 2}
    stats = scheduler.kv_cache_manager.stats()
    assert stats.prefix_hits == 2
    assert stats.active_blocks == 4
    assert [item.request_id for item in scheduler.waiting] == ["later"]
    assert scheduler.schedule() == batch
    scheduler.update_from_output(
        batch, sampled_tokens(batch, (8, 9)), eos_token_ids=frozenset()
    )
    next_batch = scheduler.schedule()
    assert next_batch.num_scheduled_tokens == {"cold": 1, "warm": 1, "later": 1}
    assert [item.request_id for item in next_batch.running_requests] == [
        "cold",
        "warm",
        "later",
    ]
    assert [item.is_prefill for item in next_batch.running_requests] == [
        False,
        False,
        True,
    ]


def test_prefill_batch_budget_defers_oversized_prompt_until_it_can_run_alone() -> None:
    scheduler = Scheduler(
        kv_cache_config=KVCacheConfig(block_size=2, num_gpu_blocks=16),
        model_fingerprint="tiny",
        max_num_seqs=4,
        max_model_len=10,
        max_prefill_batch_tokens=4,
    )
    scheduler.add_request(request("short", (1, 2), max_tokens=1))
    scheduler.add_request(request("long", (3, 4, 5, 6, 7), max_tokens=1))
    scheduler.add_request(request("after", (8,), max_tokens=1))

    first = scheduler.schedule()
    assert [item.request_id for item in first.running_requests] == ["short"]
    assert scheduler.kv_cache_manager.stats().active_blocks == 1
    scheduler.update_from_output(
        first, sampled_tokens(first, (9,)), eos_token_ids=frozenset()
    )
    second = scheduler.schedule()
    assert [item.request_id for item in second.running_requests] == ["long"]
    assert second.num_scheduled_tokens == {"long": 5}
    scheduler.update_from_output(
        second, sampled_tokens(second, (9,)), eos_token_ids=frozenset()
    )
    assert scheduler.schedule().running_requests[0].request_id == "after"


def test_chunked_prefill_advances_decode_and_commits_only_the_current_chunk() -> None:
    scheduler = Scheduler(
        kv_cache_config=KVCacheConfig(block_size=2, num_gpu_blocks=16),
        model_fingerprint="tiny",
        max_num_seqs=3,
        max_model_len=12,
        max_prefill_batch_tokens=4,
        enable_chunked_prefill=True,
    )
    scheduler.add_request(request("decode", (1, 2), max_tokens=4))
    first = scheduler.schedule()
    scheduler.update_from_output(
        first, sampled_tokens(first, (3,)), eos_token_ids=frozenset()
    )
    prompt = (4, 5, 6, 7, 8, 9)
    scheduler.add_request(request("prefill", prompt))

    mixed = scheduler.schedule()
    assert mixed.num_scheduled_tokens == {"decode": 1, "prefill": 3}
    assert [
        (item.request_id, item.scheduled_token_ids, item.num_computed_tokens)
        for item in mixed.running_requests
    ] == [("decode", (3,), 2), ("prefill", (4, 5, 6), 0)]
    assert mixed.num_samples == 1
    updates = scheduler.update_from_output(
        mixed, sampled_tokens(mixed, (11,)), eos_token_ids=frozenset()
    )
    assert [item.request.request_id for item in updates] == ["decode"]
    assert scheduler.running["prefill"].needs_prefill
    assert scheduler.running["prefill"].num_computed_tokens == 3

    last = scheduler.schedule()
    assert last.num_scheduled_tokens == {"decode": 1, "prefill": 3}
    assert [
        (item.request_id, item.scheduled_token_ids, item.num_computed_tokens)
        for item in last.running_requests
    ] == [("decode", (11,), 3), ("prefill", (7, 8, 9), 3)]
    assert last.num_samples == 2
    updates = scheduler.update_from_output(
        last, sampled_tokens(last, (10, 11)), eos_token_ids=frozenset()
    )
    assert [item.request.request_id for item in updates] == ["decode", "prefill"]
    assert not scheduler.running["prefill"].needs_prefill


def test_chunk_alignment_leaves_final_tokens_pending_for_several_prefills() -> None:
    scheduler = Scheduler(
        kv_cache_config=KVCacheConfig(block_size=2, num_gpu_blocks=16),
        model_fingerprint="tiny",
        max_num_seqs=3,
        max_model_len=10,
        max_prefill_batch_tokens=8,
        enable_chunked_prefill=True,
        prefill_chunk_alignment=4,
    )
    scheduler.add_request(request("one", (1, 2, 3, 4, 5, 6)))
    scheduler.add_request(request("two", (10, 11, 12, 13, 14, 15)))
    chunks = scheduler.schedule()
    assert chunks.num_scheduled_tokens == {"one": 4, "two": 4}
    assert chunks.num_samples == 0
    assert not scheduler.update_from_output(
        chunks, sampled_tokens(chunks, ()), eos_token_ids=frozenset()
    )
    final = scheduler.schedule()
    assert final.num_scheduled_tokens == {"one": 2, "two": 2}
    assert final.num_samples == 2


def test_isolated_request_reserves_only_the_current_chunk() -> None:
    scheduler = Scheduler(
        kv_cache_config=KVCacheConfig(block_size=2, num_gpu_blocks=8),
        model_fingerprint="tiny",
        max_num_seqs=1,
        max_model_len=8,
        max_prefill_batch_tokens=3,
        enable_chunked_prefill=True,
    )
    scheduler.add_request(request("long", (1, 2, 3, 4, 5, 6)))

    first = scheduler.schedule()
    assert first.num_scheduled_tokens == {"long": 3}
    assert len(first.running_requests[0].block_ids) == 2
    assert scheduler.kv_cache_manager.stats().active_blocks == 2
    assert not scheduler.update_from_output(
        first, sampled_tokens(first, ()), eos_token_ids=frozenset()
    )

    second = scheduler.schedule()
    assert second.num_scheduled_tokens == {"long": 3}
    assert (
        second.running_requests[0].block_ids[:2] == first.running_requests[0].block_ids
    )
    assert len(second.running_requests[0].block_ids) == 3
    assert second.num_samples == 1


def test_scheduler_finishes_on_eos_and_length() -> None:
    scheduler = Scheduler(
        kv_cache_config=KVCacheConfig(block_size=2, num_gpu_blocks=8),
        model_fingerprint="tiny",
        max_num_seqs=2,
        max_model_len=3,
    )
    scheduler.add_request(request("eos", (1,), max_tokens=3))
    scheduler.add_request(request("length", (1,), max_tokens=1))
    batch = scheduler.schedule()

    updates = scheduler.update_from_output(
        batch, sampled_tokens(batch, (9, 2)), eos_token_ids=frozenset((9,))
    )

    assert updates[0].finish_reason is FinishReason.STOP
    assert updates[1].finish_reason is FinishReason.LENGTH
    assert not scheduler.has_work


def test_scheduler_applies_ignore_eos_per_request_until_max_tokens() -> None:
    scheduler = Scheduler(
        kv_cache_config=KVCacheConfig(block_size=2, num_gpu_blocks=8),
        model_fingerprint="tiny",
        max_num_seqs=2,
        max_model_len=8,
    )
    scheduler.add_request(request("stop", (1,), max_tokens=2))
    scheduler.add_request(request("ignore", (1,), max_tokens=2, ignore_eos=True))

    first_batch = scheduler.schedule()
    first = scheduler.update_from_output(
        first_batch,
        sampled_tokens(first_batch, (9, 9)),
        eos_token_ids=frozenset((9,)),
    )
    second_batch = scheduler.schedule()
    second = scheduler.update_from_output(
        second_batch,
        sampled_tokens(second_batch, (9,)),
        eos_token_ids=frozenset((9,)),
    )

    assert first[0].finish_reason is FinishReason.STOP
    assert first[1].finish_reason is None
    assert second[0].finish_reason is FinishReason.LENGTH
    assert second[0].request.output_token_ids == [9, 9]
    assert not scheduler.has_work


def test_scheduler_aborts_waiting_and_running_requests() -> None:
    scheduler = Scheduler(
        kv_cache_config=KVCacheConfig(block_size=2, num_gpu_blocks=8),
        model_fingerprint="tiny",
        max_num_seqs=1,
        max_model_len=8,
    )
    running = request("running", (1,))
    scheduler.add_request(running)
    scheduler.add_request(request("waiting", (1,)))
    scheduler.schedule()

    assert scheduler.abort("waiting")
    assert scheduler.abort("running")
    assert running.status is RequestStatus.ABORTED
    assert not scheduler.has_work
    assert not scheduler.abort("unknown")


def test_scheduler_requires_resolved_kv_capacity() -> None:
    with pytest.raises(ValueError, match="requires resolved num_gpu_blocks"):
        Scheduler(
            max_num_seqs=1,
            max_model_len=8,
            kv_cache_config=KVCacheConfig(),
            model_fingerprint="tiny",
        )


def test_scheduler_abort_all_releases_cache_and_request_ids() -> None:
    scheduler = Scheduler(
        max_num_seqs=1,
        max_model_len=8,
        kv_cache_config=KVCacheConfig(block_size=2, num_gpu_blocks=8),
        model_fingerprint="tiny",
    )
    scheduler.add_request(request("running", (1,)))
    scheduler.add_request(request("waiting", (2,)))
    scheduler.schedule()

    aborted = scheduler.abort_all()

    assert {item.request_id for item in aborted} == {"running", "waiting"}
    assert all(item.status is RequestStatus.ABORTED for item in aborted)
    assert scheduler.kv_cache_manager.stats().active_blocks == 0
    assert not scheduler.has_work
    assert not scheduler.schedule().running_requests
    scheduler.add_request(request("running", (1,)))
    assert scheduler.schedule().running_requests[0].request_id == "running"


def test_scheduler_rejects_invalid_output_before_committing_cache() -> None:
    scheduler = Scheduler(
        max_num_seqs=1,
        max_model_len=8,
        kv_cache_config=KVCacheConfig(block_size=2, num_gpu_blocks=8),
        model_fingerprint="tiny",
    )
    active = request("one", (1,))
    scheduler.add_request(active)
    batch = scheduler.schedule()

    with pytest.raises(ValueError, match="sampled request IDs"):
        scheduler.update_from_output(
            batch, {"wrong-id": 2}, eos_token_ids=frozenset()
        )

    assert active.num_computed_tokens == 0
    assert active.needs_prefill
    assert active.output_token_ids == []
    scheduler.update_from_output(
        batch, sampled_tokens(batch, (2,)), eos_token_ids=frozenset()
    )
    assert active.num_computed_tokens == 1
    assert active.output_token_ids == [2]


def test_scheduler_releases_cache_at_context_limit() -> None:
    scheduler = Scheduler(
        max_num_seqs=1,
        max_model_len=4,
        kv_cache_config=KVCacheConfig(block_size=2, num_gpu_blocks=2),
        model_fingerprint="tiny",
    )
    scheduler.add_request(request("one", (1, 2, 3), max_tokens=10))

    batch = scheduler.schedule()
    updates = scheduler.update_from_output(
        batch, sampled_tokens(batch, (4,)), eos_token_ids=frozenset()
    )

    assert updates[0].finish_reason is FinishReason.LENGTH
    assert scheduler.kv_cache_manager.stats().active_blocks == 0
    assert not scheduler.has_work


def test_running_decode_allocates_before_waiting_prefill() -> None:
    scheduler = Scheduler(
        max_num_seqs=2,
        max_model_len=4,
        kv_cache_config=KVCacheConfig(block_size=2, num_gpu_blocks=2),
        model_fingerprint="tiny",
    )
    scheduler.add_request(request("running", (1, 2)))
    batch = scheduler.schedule()
    scheduler.update_from_output(
        batch, sampled_tokens(batch, (3,)), eos_token_ids=frozenset()
    )
    scheduler.add_request(request("waiting", (11, 12)))

    decode = scheduler.schedule()
    assert [item.request_id for item in decode.running_requests] == ["running"]
    assert [item.request_id for item in scheduler.waiting] == ["waiting"]
    assert decode.num_scheduled_tokens == {"running": 1}
    scheduler.update_from_output(
        decode, sampled_tokens(decode, (4,)), eos_token_ids=frozenset()
    )
    assert [item.request_id for item in scheduler.schedule().running_requests] == [
        "waiting"
    ]
