from __future__ import annotations

from x_tokens.core.kv_cache_manager import BlockPool, FreeBlockQueue, KVBlock


def test_free_queue_can_remove_and_reorder_linked_blocks() -> None:
    queue = FreeBlockQueue()
    first, middle, last = (KVBlock(index) for index in range(3))
    for block in (first, middle, last):
        queue.push_tail(block)

    queue.remove(middle)
    assert first.next_free_block is last
    assert last.prev_free_block is first
    assert middle.prev_free_block is middle.next_free_block is None
    queue.move_to_tail(first)

    assert queue.head is last and queue.tail is first
    assert queue.size == 2
    assert queue.pop_head() is last
    assert queue.pop_head() is first
    assert queue.head is queue.tail is None
    assert queue.size == 0


def test_pool_pins_shared_hits_and_evicts_only_unreferenced_blocks() -> None:
    pool = BlockPool(2)
    shared, unused = pool.allocate(), pool.allocate()
    pool.insert("prefix", shared)
    pool.release(shared)
    pool.release(unused)

    assert pool.lookup_and_acquire("prefix") is shared
    assert pool.lookup_and_acquire("prefix") is shared
    assert shared.ref_cnt == 2
    assert pool.free_queue.size == 1
    assert pool.allocate() is unused
    pool.release(shared)
    assert pool.free_queue.size == 0
    assert pool.lookup("prefix") is shared

    pool.release(shared)
    assert pool.allocate() is shared
    assert pool.lookup("prefix") is None
    assert pool.evictions == 1


def test_pool_keeps_shared_hash_index_when_releasing_a_duplicate() -> None:
    pool = BlockPool(2)
    first, duplicate = pool.allocate(), pool.allocate()
    pool.insert("prefix", first)
    pool.insert("prefix", duplicate)

    pool.release(duplicate)

    assert duplicate.block_hash is None
    assert pool.allocate() is duplicate
    assert pool.lookup("prefix") is first
    assert pool.evictions == 0
