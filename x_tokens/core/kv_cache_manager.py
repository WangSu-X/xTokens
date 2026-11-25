"""KV block allocation, ownership, and automatic prefix caching."""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field

from x_tokens.config import KVCacheConfig

from ..engine.protocol import EngineCoreRequest


class KVCacheCapacityError(RuntimeError):
    """Raised when no physical block can be allocated."""


@dataclass(slots=True, eq=False)
class KVBlock:
    block_id: int
    block_hash: str | None = None
    token_ids: tuple[int, ...] = ()
    ref_cnt: int = 0
    prev_free_block: KVBlock | None = field(default=None, repr=False)
    next_free_block: KVBlock | None = field(default=None, repr=False)


class FreeBlockQueue:
    """FIFO of unreferenced blocks with constant-time removal for prefix hits."""

    def __init__(self) -> None:
        self.head: KVBlock | None = None
        self.tail: KVBlock | None = None
        self.size = 0

    def push_tail(self, block: KVBlock) -> None:
        block.prev_free_block = self.tail
        if self.tail is None:
            self.head = block
        else:
            self.tail.next_free_block = block
        self.tail = block
        self.size += 1

    def pop_head(self) -> KVBlock:
        block = self.head
        if block is None:
            raise KVCacheCapacityError("KV cache has no evictable block")
        self.remove(block)
        return block

    def remove(self, block: KVBlock) -> None:
        previous = block.prev_free_block
        following = block.next_free_block
        if previous is None:
            self.head = following
        else:
            previous.next_free_block = following
        if following is None:
            self.tail = previous
        else:
            following.prev_free_block = previous
        block.prev_free_block = None
        block.next_free_block = None
        self.size -= 1

    def move_to_tail(self, block: KVBlock) -> None:
        if block is not self.tail:
            self.remove(block)
            self.push_tail(block)


class BlockPool:
    """Own all logical block IDs and the index of reusable full blocks."""

    def __init__(self, num_blocks: int) -> None:
        self.blocks = [KVBlock(block_id) for block_id in range(num_blocks)]
        self.hash_to_block: dict[str, KVBlock] = {}
        self.free_queue = FreeBlockQueue()
        self.evictions = 0
        for block in self.blocks:
            self.free_queue.push_tail(block)

    def allocate(self) -> KVBlock:
        block = self.free_queue.pop_head()
        if block.block_hash is not None:
            self._evict(block)
        block.ref_cnt = 1
        return block

    def release(self, block: KVBlock) -> None:
        if block.ref_cnt < 1:
            raise RuntimeError("KV cache block ref count underflow")
        block.ref_cnt -= 1
        if block.ref_cnt > 0:
            return
        if (
            block.block_hash is not None
            and self.hash_to_block.setdefault(block.block_hash, block) is not block
        ):
            # The other copy remains the reusable block.
            block.block_hash = None
            block.token_ids = ()
        self.free_queue.push_tail(block)

    def lookup(self, block_hash: str) -> KVBlock | None:
        return self.hash_to_block.get(block_hash)

    def lookup_and_acquire(self, block_hash: str) -> KVBlock | None:
        block = self.lookup(block_hash)
        if block is not None:
            if block.ref_cnt == 0:
                self.free_queue.remove(block)
            block.ref_cnt += 1
        return block

    def insert(self, block_hash: str, block: KVBlock) -> None:
        block.block_hash = block_hash
        self.hash_to_block.setdefault(block_hash, block)

    def _evict(self, block: KVBlock) -> None:
        block_hash = block.block_hash
        if block_hash is not None and self.hash_to_block.get(block_hash) is block:
            del self.hash_to_block[block_hash]
        block.block_hash = None
        block.token_ids = ()
        self.evictions += 1


@dataclass(frozen=True, slots=True)
class KVCacheStats:
    total_blocks: int
    free_blocks: int
    cached_blocks: int
    active_blocks: int
    prefix_hits: int
    prefix_misses: int
    evictions: int


class KVCacheManager:
    """Assign blocks from the pool and register completed request prefixes."""

    def __init__(
        self,
        config: KVCacheConfig,
        *,
        num_blocks: int,
        max_model_len: int,
        model_fingerprint: str,
    ) -> None:
        if num_blocks < 1:
            raise ValueError("num_blocks must be positive")
        if num_blocks * config.block_size < max_model_len:
            raise ValueError("KV cache must hold at least one max_model_len sequence")
        self.block_size = config.block_size
        self.enable_prefix_caching = config.enable_prefix_caching
        self.max_model_len = max_model_len
        self.model_fingerprint = model_fingerprint.encode("utf-8")
        self.pack_tokens = struct.Struct(f"<{self.block_size}q")
        self.block_pool = BlockPool(num_blocks)
        self.req_to_blocks: dict[str, list[KVBlock]] = {}
        self.num_cached_blocks: dict[str, int] = {}
        self.prepared_token_ids: dict[str, tuple[int, ...]] = {}
        self.prefix_hits = 0
        self.prefix_misses = 0

    def get_computed_blocks(self, request: EngineCoreRequest) -> tuple[KVBlock, ...]:
        """Probe the reusable prefix. Allocation acquires the references."""
        if not self.enable_prefix_caching:
            return ()
        token_ids = request.all_token_ids
        matched: list[KVBlock] = []
        parent_hash = ""
        # Keep at least the last token for the model to produce logits.
        for start in range(0, len(token_ids) - self.block_size, self.block_size):
            block_tokens = token_ids[start : start + self.block_size]
            block_hash = self._hash_block(parent_hash, block_tokens)
            block = self.block_pool.lookup(block_hash)
            if block is None or block.token_ids != block_tokens:
                break
            matched.append(block)
            parent_hash = block_hash
        return tuple(matched)

    def allocate_slots(
        self,
        request: EngineCoreRequest,
        num_new_tokens: int,
        *,
        computed_blocks: tuple[KVBlock, ...] = (),
    ) -> tuple[KVBlock, ...] | None:
        """Reserve this step's blocks. A capacity miss leaves the pool unchanged."""
        request_id = request.request_id
        blocks = self.req_to_blocks.get(request_id)
        computed_tokens = (
            request.num_computed_tokens
            if blocks is not None
            else len(computed_blocks) * self.block_size
        )
        end = computed_tokens + num_new_tokens
        token_ids = request.all_token_ids
        if num_new_tokens < 1 or end > min(len(token_ids), self.max_model_len):
            raise ValueError("KV cache allocation exceeds the request context")

        held_blocks = len(blocks) if blocks is not None else len(computed_blocks)
        needed = max(0, (end + self.block_size - 1) // self.block_size - held_blocks)
        idle_hits = sum(block.ref_cnt == 0 for block in computed_blocks)
        if needed + idle_hits > self.block_pool.free_queue.size:
            return None

        if blocks is None:
            hashes: list[str] = []
            for block in computed_blocks:
                block_hash = block.block_hash
                assert block_hash is not None
                acquired = self.block_pool.lookup_and_acquire(block_hash)
                assert acquired is block
                hashes.append(block_hash)
            blocks = list(computed_blocks)
            self.req_to_blocks[request_id] = blocks
            self.num_cached_blocks[request_id] = len(blocks)
            request.block_hashes[:] = hashes
            request.num_computed_tokens = computed_tokens
            self.prefix_hits += len(blocks)
            self.prefix_misses += int(
                self.enable_prefix_caching
                and len(blocks) < (len(token_ids) - 1) // self.block_size
            )

        new_blocks = tuple(self.block_pool.allocate() for _ in range(needed))
        blocks.extend(new_blocks)
        self.prepared_token_ids[request_id] = token_ids[:end]
        return new_blocks

    def get_block_ids(self, request_id: str) -> tuple[int, ...]:
        return tuple(block.block_id for block in self.req_to_blocks[request_id])

    def cache_blocks(
        self, request: EngineCoreRequest, num_computed_tokens: int
    ) -> None:
        """Register full blocks after a successful forward."""
        request_id = request.request_id
        blocks = self.req_to_blocks[request_id]
        token_ids = request.all_token_ids
        if num_computed_tokens > len(blocks) * self.block_size:
            raise ValueError("KV cache commit exceeds reserved space")
        if (
            num_computed_tokens > len(token_ids)
            or self.prepared_token_ids.get(request_id)
            != token_ids[:num_computed_tokens]
        ):
            raise ValueError("KV cache commit does not match the prepared step")

        if self.enable_prefix_caching:
            complete_blocks = num_computed_tokens // self.block_size
            parent_hash = request.block_hashes[-1] if request.block_hashes else ""
            for index in range(self.num_cached_blocks[request_id], complete_blocks):
                block = blocks[index]
                if block.ref_cnt != 1 or block.block_hash is not None:
                    raise RuntimeError("cannot cache into a shared KV block")
                start = index * self.block_size
                block.token_ids = token_ids[start : start + self.block_size]
                parent_hash = self._hash_block(parent_hash, block.token_ids)
                self.block_pool.insert(parent_hash, block)
                request.block_hashes.append(parent_hash)
            self.num_cached_blocks[request_id] = complete_blocks
        del self.prepared_token_ids[request_id]

    def free(self, request_id: str) -> None:
        blocks = self.req_to_blocks.pop(request_id, ())
        self.num_cached_blocks.pop(request_id, None)
        self.prepared_token_ids.pop(request_id, None)
        # Release the suffix first so eviction retains the longest usable prefix.
        for block in reversed(blocks):
            self.block_pool.release(block)

    def stats(self) -> KVCacheStats:
        pool = self.block_pool
        return KVCacheStats(
            total_blocks=len(pool.blocks),
            free_blocks=sum(
                block.ref_cnt == 0 and block.block_hash is None for block in pool.blocks
            ),
            cached_blocks=sum(block.block_hash is not None for block in pool.blocks),
            active_blocks=len(pool.blocks) - pool.free_queue.size,
            prefix_hits=self.prefix_hits,
            prefix_misses=self.prefix_misses,
            evictions=pool.evictions,
        )

    def _hash_block(self, parent_hash: str, token_ids: tuple[int, ...]) -> str:
        digest = hashlib.blake2b(digest_size=16)
        digest.update(self.model_fingerprint)
        digest.update(bytes.fromhex(parent_hash))
        digest.update(self.pack_tokens.pack(*token_ids))
        return digest.hexdigest()
