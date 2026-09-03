from __future__ import annotations

from x_tokens.config import KVCacheConfig
from x_tokens.core.kv_cache_manager import KVCacheManager
from x_tokens.core.scheduler import KVCacheScheduler, SchedulerPhase
from x_tokens.engine.types import GenerateRequest, SamplingParams


def cache_config(*, blocks: int = 8, block_size: int = 2) -> KVCacheConfig:
    return KVCacheConfig(
        block_size=block_size,
        num_gpu_blocks=blocks,
    )


def request(
    request_id: str,
    max_tokens: int = 3,
    *,
    ignore_eos: bool = False,
) -> GenerateRequest:
    return GenerateRequest(
        request_id,
        "test-model",
        "prompt",
        SamplingParams(
            max_tokens=max_tokens,
            temperature=0,
            ignore_eos=ignore_eos,
        ),
    )


def test_prefix_blocks_are_reused_and_reference_counted() -> None:
    manager = KVCacheManager(
        cache_config(),
        num_blocks=8,
        max_model_len=8,
        model_fingerprint="tiny",
    )
    tokens = (1, 2, 3)
    assert manager.admit("first", tokens)
    first = manager.prepare_step("first", tokens, is_prefill=True)
    manager.commit_step("first", tokens)
    manager.release("first")

    assert manager.admit("second", tokens)
    second = manager.prepare_step("second", tokens, is_prefill=True)

    assert second.query_len == 1
    assert second.block_table[0] == first.block_table[0]
    assert manager.blocks[second.block_table[0]].ref_count == 1
    assert manager.stats().prefix_hits == 1
    manager.release("second")


def test_identical_full_tail_block_is_recomputed_without_mutating_prefix() -> None:
    manager = KVCacheManager(
        cache_config(blocks=8),
        num_blocks=8,
        max_model_len=8,
        model_fingerprint="tiny",
    )
    tokens = (1, 2, 3, 4)
    assert manager.admit("first", tokens)
    first = manager.prepare_step("first", tokens, is_prefill=True)
    manager.commit_step("first", tokens)
    manager.release("first")

    assert manager.admit("second", tokens)
    second = manager.prepare_step("second", tokens, is_prefill=True)
    manager.commit_step("second", tokens)

    assert second.query_len == 2
    assert second.block_table[0] == first.block_table[0]
    assert second.block_table[1] != first.block_table[1]
    manager.prepare_step("second", tokens + (5,), is_prefill=False)
    manager.commit_step("second", tokens + (5,))
    manager.release("second")


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
    prompt = (1, 2, 3, 4)
    assert manager.admit("request", prompt)
    manager.prepare_step("request", prompt, is_prefill=True)
    manager.commit_step("request", prompt)
    manager.prepare_step("request", prompt + (5,), is_prefill=False)
    manager.commit_step("request", prompt + (5,))
    manager.release("request")

    assert manager.stats().free_blocks == 8


def test_lru_evicts_only_unreferenced_cached_blocks() -> None:
    manager = KVCacheManager(
        cache_config(blocks=2, block_size=1),
        num_blocks=2,
        max_model_len=2,
        model_fingerprint="tiny",
    )
    assert manager.admit("cached", (1, 2))
    manager.prepare_step("cached", (1, 2), is_prefill=True)
    manager.commit_step("cached", (1, 2))
    manager.release("cached")

    assert manager.admit("new", (9,))

    assert manager.stats().evictions == 1
    assert manager.stats().active_blocks == 1
    manager.release("new")


def test_kv_continuous_batching_interleaves_prefill_and_decode() -> None:
    config = cache_config(blocks=16)
    scheduler = KVCacheScheduler(
        max_num_seqs=2,
        max_model_len=8,
        kv_cache_config=config,
        model_fingerprint="tiny",
    )
    scheduler.add_request(request("a"), (1, 2))

    first = scheduler.schedule()
    assert first.phase is SchedulerPhase.PREFILL
    assert first.kv_cache_metadata[0].query_len == 2
    scheduler.update_from_output(first, (3,), eos_token_ids=frozenset())

    second = scheduler.schedule()
    assert second.phase is SchedulerPhase.DECODE
    assert second.kv_cache_metadata[0].query_len == 1
    scheduler.update_from_output(second, (4,), eos_token_ids=frozenset())

    scheduler.add_request(request("b", max_tokens=2), (5, 6))
    third = scheduler.schedule()
    assert third.phase is SchedulerPhase.PREFILL
    assert [item.request_id for item in third.requests] == ["b"]
    scheduler.update_from_output(third, (7,), eos_token_ids=frozenset())

    fourth = scheduler.schedule()
    assert fourth.phase is SchedulerPhase.DECODE
    assert [item.request_id for item in fourth.requests] == ["a", "b"]
    assert all(item.query_len == 1 for item in fourth.kv_cache_metadata)
    scheduler.update_from_output(fourth, (8, 9), eos_token_ids=frozenset())

    assert not scheduler.has_work
    assert scheduler.kv_cache_manager.stats().active_blocks == 0


def test_kv_scheduler_releases_blocks_on_abort_and_failure() -> None:
    scheduler = KVCacheScheduler(
        max_num_seqs=2,
        max_model_len=8,
        kv_cache_config=cache_config(blocks=16),
        model_fingerprint="tiny",
    )
    scheduler.add_request(request("abort"), (1, 2))
    scheduler.schedule()
    assert scheduler.abort("abort")
    assert scheduler.kv_cache_manager.stats().active_blocks == 0

    scheduler.add_request(request("failure"), (3, 4))
    batch = scheduler.schedule()
    assert [item.request_id for item in scheduler.fail_batch(batch)] == ["failure"]
    assert scheduler.kv_cache_manager.stats().active_blocks == 0


def test_kv_scheduler_ignore_eos_finishes_at_generation_limit() -> None:
    scheduler = KVCacheScheduler(
        max_num_seqs=1,
        max_model_len=8,
        kv_cache_config=cache_config(blocks=8),
        model_fingerprint="tiny",
    )
    scheduler.add_request(request("ignore-eos", max_tokens=2, ignore_eos=True), (1,))
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
    scheduler = KVCacheScheduler(
        max_num_seqs=2,
        max_model_len=4,
        kv_cache_config=cache_config(blocks=3),
        model_fingerprint="tiny",
    )
    scheduler.add_request(request("a"), (1,))
    scheduler.add_request(request("b"), (2,))
    prefill = scheduler.schedule()
    scheduler.update_from_output(prefill, (3, 4), eos_token_ids=frozenset())
    decode = scheduler.schedule()
    scheduler.update_from_output(decode, (5, 6), eos_token_ids=frozenset())

    after_preemption = scheduler.schedule()

    assert [item.request_id for item in after_preemption.requests] == ["a"]
    assert [item.request_id for item in scheduler.waiting] == ["b"]
