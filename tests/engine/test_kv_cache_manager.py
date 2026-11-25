from __future__ import annotations

import pytest

from x_tokens.config import KVCacheConfig
from x_tokens.core.kv_cache_manager import KVCacheManager
from x_tokens.core.scheduler import Scheduler
from x_tokens.engine.protocol import EngineCoreRequest, SamplingParams


def cache_config(*, blocks: int = 8, block_size: int = 2) -> KVCacheConfig:
    return KVCacheConfig(
        block_size=block_size,
        num_gpu_blocks=blocks,
    )


def request(
    request_id: str,
    prompt_token_ids: tuple[int, ...],
    max_tokens: int = 3,
    *,
    ignore_eos: bool = False,
) -> EngineCoreRequest:
    return EngineCoreRequest(
        request_id,
        SamplingParams(
            max_tokens=max_tokens,
            temperature=0,
            ignore_eos=ignore_eos,
        ),
        prompt_token_ids,
    )


def allocate_remaining(
    manager: KVCacheManager, request: EngineCoreRequest
) -> tuple[int, ...] | None:
    cached = (
        manager.get_computed_blocks(request)
        if request.request_id not in manager.req_to_blocks
        else ()
    )
    count = request.num_uncomputed_tokens - len(cached) * manager.block_size
    if manager.allocate_slots(request, count, computed_blocks=cached) is None:
        return None
    return manager.get_block_ids(request.request_id)


def cache_remaining(manager: KVCacheManager, request: EngineCoreRequest) -> None:
    end = len(request.all_token_ids)
    manager.cache_blocks(request, end)
    request.num_computed_tokens = end


def test_prefix_blocks_are_reused_and_reference_counted() -> None:
    manager = KVCacheManager(
        cache_config(),
        num_blocks=8,
        max_model_len=8,
        model_fingerprint="tiny",
    )
    tokens = (1, 2, 3)
    first_request = request("first", tokens)
    first = allocate_remaining(manager, first_request)
    assert first is not None
    cache_remaining(manager, first_request)
    manager.free("first")

    second_request = request("second", tokens)
    second = allocate_remaining(manager, second_request)
    assert second is not None

    assert second_request.num_uncomputed_tokens == 1
    assert second_request.num_computed_tokens == 2
    assert second_request.block_hashes == first_request.block_hashes
    assert len(second_request.block_hashes) == 1
    assert second[0] == first[0]
    assert manager.block_pool.blocks[second[0]].ref_cnt == 1
    assert manager.stats().prefix_hits == 1
    manager.free("second")


def test_identical_full_tail_block_is_recomputed_without_mutating_prefix() -> None:
    manager = KVCacheManager(
        cache_config(blocks=8),
        num_blocks=8,
        max_model_len=8,
        model_fingerprint="tiny",
    )
    tokens = (1, 2, 3, 4)
    first_request = request("first", tokens)
    first = allocate_remaining(manager, first_request)
    assert first is not None
    cache_remaining(manager, first_request)
    manager.free("first")

    second_request = request("second", tokens)
    second = allocate_remaining(manager, second_request)
    assert second is not None
    assert second_request.num_uncomputed_tokens == 2
    cache_remaining(manager, second_request)

    assert second[0] == first[0]
    assert second[1] != first[1]
    second_request.output_token_ids.append(5)
    assert allocate_remaining(manager, second_request) is not None
    cache_remaining(manager, second_request)
    manager.free("second")


def test_decode_commits_completed_blocks_when_prefix_caching_is_disabled() -> None:
    config = KVCacheConfig(
        block_size=2,
        num_gpu_blocks=8,
        enable_prefix_caching=False,
    )
    manager = KVCacheManager(
        config,
        num_blocks=8,
        max_model_len=8,
        model_fingerprint="tiny",
    )
    active = request("request", (1, 2, 3, 4))
    assert allocate_remaining(manager, active) is not None
    cache_remaining(manager, active)
    active.output_token_ids.append(5)
    assert allocate_remaining(manager, active) is not None
    cache_remaining(manager, active)
    manager.free("request")

    assert manager.stats().free_blocks == 8
    assert active.num_computed_tokens == 5
    assert active.block_hashes == []


def test_lru_evicts_only_unreferenced_cached_blocks() -> None:
    manager = KVCacheManager(
        cache_config(blocks=2, block_size=1),
        num_blocks=2,
        max_model_len=2,
        model_fingerprint="tiny",
    )
    cached = request("cached", (1, 2))
    assert allocate_remaining(manager, cached) is not None
    cache_remaining(manager, cached)
    manager.free("cached")

    assert allocate_remaining(manager, request("new", (9,))) is not None

    assert manager.stats().evictions == 1
    assert manager.stats().active_blocks == 1
    manager.free("new")


def test_recycling_a_released_suffix_retains_the_reusable_prefix() -> None:
    manager = KVCacheManager(
        cache_config(blocks=3), num_blocks=3, max_model_len=6, model_fingerprint="tiny"
    )
    prompt = (1, 2, 3, 4, 5, 6)
    cached = request("cached", prompt)
    block_ids = allocate_remaining(manager, cached)
    assert block_ids is not None
    cache_remaining(manager, cached)
    manager.free(cached.request_id)

    new_block_ids = allocate_remaining(manager, request("new", (9,)))
    reusable = manager.get_computed_blocks(request("reuse", prompt))

    assert new_block_ids == block_ids[-1:]
    assert tuple(block.block_id for block in reusable) == block_ids[:-1]
    assert manager.block_pool.lookup(cached.block_hashes[-1]) is None


def test_kv_continuous_batching_schedules_running_and_waiting_together() -> None:
    config = cache_config(blocks=16)
    scheduler = Scheduler(
        max_num_seqs=2,
        max_model_len=8,
        kv_cache_config=config,
        model_fingerprint="tiny",
    )
    scheduler.add_request(request("a", (1, 2)))

    first = scheduler.schedule()
    assert first.num_scheduled_tokens == {"a": 2}
    scheduler.update_from_output(first, (3,), eos_token_ids=frozenset())

    scheduler.add_request(request("b", (5, 6), max_tokens=2))
    second = scheduler.schedule()
    assert second.num_scheduled_tokens == {"a": 1, "b": 2}
    scheduler.update_from_output(second, (4, 7), eos_token_ids=frozenset())

    third = scheduler.schedule()
    assert third.num_scheduled_tokens == {"a": 1, "b": 1}
    assert [item.request_id for item in third.running_requests] == ["a", "b"]
    scheduler.update_from_output(third, (8, 9), eos_token_ids=frozenset())

    assert not scheduler.has_work
    assert scheduler.kv_cache_manager.stats().active_blocks == 0


def test_kv_scheduler_releases_blocks_on_abort_and_failure() -> None:
    scheduler = Scheduler(
        max_num_seqs=2,
        max_model_len=8,
        kv_cache_config=cache_config(blocks=16),
        model_fingerprint="tiny",
    )
    scheduler.add_request(request("abort", (1, 2)))
    scheduler.schedule()
    assert scheduler.abort("abort")
    assert scheduler.kv_cache_manager.stats().active_blocks == 0

    scheduler.add_request(request("failure", (3, 4)))
    batch = scheduler.schedule()
    assert [item.request_id for item in scheduler.fail_batch(batch)] == ["failure"]
    assert scheduler.kv_cache_manager.stats().active_blocks == 0


def test_kv_scheduler_ignore_eos_finishes_at_generation_limit() -> None:
    scheduler = Scheduler(
        max_num_seqs=1,
        max_model_len=8,
        kv_cache_config=cache_config(blocks=8),
        model_fingerprint="tiny",
    )
    scheduler.add_request(request("ignore-eos", (1,), max_tokens=2, ignore_eos=True))
    first = scheduler.schedule()
    first_update = scheduler.update_from_output(
        first, (9,), eos_token_ids=frozenset((9,))
    )
    assert first_update[0].finish_reason is None

    second = scheduler.schedule()
    second_update = scheduler.update_from_output(
        second, (9,), eos_token_ids=frozenset((9,))
    )
    assert second_update[0].finish_reason is not None
    assert scheduler.kv_cache_manager.stats().active_blocks == 0


def test_kv_scheduler_preempts_when_decode_needs_another_block() -> None:
    scheduler = Scheduler(
        max_num_seqs=2,
        max_model_len=4,
        kv_cache_config=cache_config(blocks=3),
        model_fingerprint="tiny",
    )
    scheduler.add_request(request("a", (1,)))
    scheduler.add_request(request("b", (2,)))
    prefill = scheduler.schedule()
    scheduler.update_from_output(prefill, (3, 4), eos_token_ids=frozenset())
    decode = scheduler.schedule()
    scheduler.update_from_output(decode, (5, 6), eos_token_ids=frozenset())

    after_preemption = scheduler.schedule()

    assert [item.request_id for item in after_preemption.running_requests] == ["a"]
    assert [item.request_id for item in scheduler.waiting] == ["b"]
    victim = scheduler.waiting[0]
    assert victim.num_computed_tokens == 0
    assert victim.block_hashes == []
    scheduler.update_from_output(after_preemption, (7,), eos_token_ids=frozenset())

    resumed = scheduler.schedule()
    assert [item.request_id for item in resumed.running_requests] == ["b"]
    assert resumed.running_requests[0].request is victim
    assert victim.num_computed_tokens == 2
    assert victim.all_token_ids == (2, 4, 6)
    assert resumed.num_scheduled_tokens == {"b": 1}
    final = scheduler.update_from_output(resumed, (8,), eos_token_ids=frozenset())
    assert final[0].request.output_token_ids == [4, 6, 8]
    assert not scheduler.has_work


def test_failed_reservation_keeps_cached_blocks_available_for_reuse() -> None:
    manager = KVCacheManager(
        cache_config(blocks=4), num_blocks=4, max_model_len=8, model_fingerprint="tiny"
    )
    cached = request("cached", (1, 2, 3, 4))
    assert allocate_remaining(manager, cached) is not None
    cache_remaining(manager, cached)
    manager.free("cached")

    assert allocate_remaining(manager, request("running", (7, 8, 9, 10))) is not None
    rejected = request("too-large", (1, 2, 11, 12, 13))
    assert allocate_remaining(manager, rejected) is None
    assert rejected.num_computed_tokens == 0
    assert rejected.block_hashes == []
    assert manager.stats().evictions == 0

    reused = request("reuse", (1, 2, 3))
    assert allocate_remaining(manager, reused) is not None
    assert reused.num_computed_tokens == 2


def test_scheduler_retries_allocation_after_preempting_the_running_tail() -> None:
    scheduler = Scheduler(
        max_num_seqs=3,
        max_model_len=4,
        kv_cache_config=cache_config(blocks=3),
        model_fingerprint="tiny",
    )
    scheduler.add_request(request("head", (1, 2)))
    scheduler.add_request(request("middle", (3,)))
    scheduler.add_request(request("tail", (4, 5)))
    scheduler.update_from_output(
        scheduler.schedule(), (6, 7, 8), eos_token_ids=frozenset()
    )

    batch = scheduler.schedule()

    assert batch.num_scheduled_tokens == {"head": 1, "middle": 1}
    assert len(batch.running_requests[0].block_ids) == 2
    assert scheduler.waiting[0].request_id == "tail"
    assert scheduler.waiting[0].num_computed_tokens == 0


def test_cache_only_publishes_blocks_from_the_prepared_step() -> None:
    manager = KVCacheManager(
        cache_config(), num_blocks=8, max_model_len=8, model_fingerprint="tiny"
    )
    active = request("request", (1, 2))
    assert allocate_remaining(manager, active) is not None
    active.prompt_token_ids = (1, 3)
    with pytest.raises(ValueError, match="does not match the prepared step"):
        cache_remaining(manager, active)
    assert manager.stats().cached_blocks == 0
    assert active.num_computed_tokens == 0
    assert active.block_hashes == []
    active.prompt_token_ids = (1, 2)
    cache_remaining(manager, active)
    assert manager.stats().cached_blocks == 1
