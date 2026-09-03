"""Logical block manager for KV cache and prefix reuse."""

from __future__ import annotations

import hashlib
import math
from collections import deque
from dataclasses import dataclass, field

from x_tokens.config import KVCacheConfig
from x_tokens.kv_cache import KVCacheMetadata


class KVCacheCapacityError(RuntimeError):
    """Raised when no physical block can be allocated."""


@dataclass(slots=True)
class KVCacheBlock:
    block_id: int
    block_hash: bytes | None = None
    token_ids: tuple[int, ...] = ()
    ref_count: int = 0
    computed: bool = False
    last_access: int = 0


@dataclass(slots=True)
class RequestKVCacheState:
    request_id: str
    block_ids: list[int] = field(default_factory=list)
    num_computed_tokens: int = 0


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
    """Driver-owned logical cache state; workers only consume block metadata."""

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
        self.config = config
        self.num_blocks = num_blocks
        self.max_model_len = max_model_len
        self.model_fingerprint = model_fingerprint.encode("utf-8")
        self.blocks = [KVCacheBlock(index) for index in range(num_blocks)]
        self._free_block_ids = deque(range(num_blocks))
        self._hash_to_block_id: dict[bytes, int] = {}
        self._requests: dict[str, RequestKVCacheState] = {}
        self._clock = 0
        self._prefix_hits = 0
        self._prefix_misses = 0
        self._evictions = 0

    def admit(self, request_id: str, token_ids: tuple[int, ...]) -> bool:
        if request_id in self._requests:
            raise ValueError(f"request already owns KV cache blocks: {request_id}")
        if not token_ids:
            raise ValueError("KV cache request must contain at least one token")
        acquired: list[int] = []
        parent_hash = b""
        max_reused_blocks = max(0, (len(token_ids) - 1) // self.config.block_size)
        if self.config.enable_prefix_caching:
            for block_index in range(max_reused_blocks):
                start = block_index * self.config.block_size
                block_tokens = token_ids[start : start + self.config.block_size]
                block_hash = self._hash_block(parent_hash, block_tokens)
                block_id = self._hash_to_block_id.get(block_hash)
                if block_id is None:
                    self._prefix_misses += 1
                    break
                block = self.blocks[block_id]
                if not block.computed or block.token_ids != block_tokens:
                    self._prefix_misses += 1
                    break
                self._acquire(block)
                acquired.append(block_id)
                self._prefix_hits += 1
                parent_hash = block_hash

        matched_tokens = len(acquired) * self.config.block_size
        needed_blocks = math.ceil(
            (len(token_ids) - matched_tokens) / self.config.block_size
        )
        try:
            acquired.extend(self._allocate_blocks(needed_blocks))
        except KVCacheCapacityError:
            self._release_block_ids(acquired)
            return False
        self._requests[request_id] = RequestKVCacheState(
            request_id,
            acquired,
            matched_tokens,
        )
        return True

    def prepare_step(
        self,
        request_id: str,
        token_ids: tuple[int, ...],
        *,
        is_prefill: bool,
    ) -> KVCacheMetadata:
        state = self._state(request_id)
        if len(token_ids) <= state.num_computed_tokens:
            raise ValueError("KV cache step has no new token to compute")
        required_blocks = math.ceil(len(token_ids) / self.config.block_size)
        additional = required_blocks - len(state.block_ids)
        if additional > 0:
            state.block_ids.extend(self._allocate_blocks(additional))
        query_len = len(token_ids) - state.num_computed_tokens
        slots = tuple(
            self._slot(state.block_ids, position)
            for position in range(state.num_computed_tokens, len(token_ids))
        )
        for block_id in state.block_ids:
            self._touch(self.blocks[block_id])
        return KVCacheMetadata(
            request_id=request_id,
            is_prefill=is_prefill,
            context_len=len(token_ids),
            query_len=query_len,
            block_table=tuple(state.block_ids),
            slot_mapping=slots,
        )

    def commit_step(self, request_id: str, token_ids: tuple[int, ...]) -> None:
        state = self._state(request_id)
        if len(token_ids) < state.num_computed_tokens:
            raise ValueError("KV cache computed token count cannot move backwards")
        parent_hash = b""
        complete_blocks = len(token_ids) // self.config.block_size
        for block_index in range(complete_blocks):
            block_id = state.block_ids[block_index]
            block = self.blocks[block_id]
            start = block_index * self.config.block_size
            block_tokens = token_ids[start : start + self.config.block_size]
            block_hash = self._hash_block(parent_hash, block_tokens)
            if block.computed:
                hash_changed = (
                    self.config.enable_prefix_caching and block.block_hash != block_hash
                )
                if hash_changed or block.token_ids != block_tokens:
                    raise RuntimeError(
                        "shared KV cache block content changed: "
                        f"request_id={request_id} block_index={block_index} "
                        f"block_id={block_id} "
                        f"hash_match={block.block_hash == block_hash} "
                        f"tokens_match={block.token_ids == block_tokens}"
                    )
            else:
                block.computed = True
                block.token_ids = block_tokens
                if self.config.enable_prefix_caching:
                    block.block_hash = block_hash
                    existing = self._hash_to_block_id.get(block_hash)
                    if existing is None or existing == block_id:
                        self._hash_to_block_id[block_hash] = block_id
            self._touch(block)
            parent_hash = block_hash
        state.num_computed_tokens = len(token_ids)

    def release(self, request_id: str) -> None:
        state = self._requests.pop(request_id, None)
        if state is not None:
            self._release_block_ids(state.block_ids)

    def num_computed_tokens(self, request_id: str) -> int:
        return self._state(request_id).num_computed_tokens

    def block_table(self, request_id: str) -> tuple[int, ...]:
        return tuple(self._state(request_id).block_ids)

    def stats(self) -> KVCacheStats:
        cached = sum(
            block.computed and block.block_hash is not None for block in self.blocks
        )
        active = sum(block.ref_count > 0 for block in self.blocks)
        return KVCacheStats(
            total_blocks=self.num_blocks,
            free_blocks=len(self._free_block_ids),
            cached_blocks=cached,
            active_blocks=active,
            prefix_hits=self._prefix_hits,
            prefix_misses=self._prefix_misses,
            evictions=self._evictions,
        )

    def _allocate_blocks(self, count: int) -> list[int]:
        allocated: list[int] = []
        try:
            for _ in range(count):
                allocated.append(self._allocate_block())
        except KVCacheCapacityError:
            self._release_block_ids(allocated)
            raise
        return allocated

    def _allocate_block(self) -> int:
        if not self._free_block_ids:
            self._evict_one()
        if not self._free_block_ids:
            raise KVCacheCapacityError("KV cache has no evictable block")
        block_id = self._free_block_ids.popleft()
        block = self.blocks[block_id]
        if block.ref_count or block.computed or block.block_hash is not None:
            raise RuntimeError("free KV cache block is not reset")
        block.ref_count = 1
        self._touch(block)
        return block_id

    def _evict_one(self) -> None:
        candidates = [
            block
            for block in self.blocks
            if block.ref_count == 0 and block.block_hash is not None
        ]
        if not candidates:
            return
        victim = min(candidates, key=lambda block: block.last_access)
        self._reset_block(victim)
        self._evictions += 1

    def _release_block_ids(self, block_ids: list[int]) -> None:
        for block_id in block_ids:
            block = self.blocks[block_id]
            if block.ref_count < 1:
                raise RuntimeError("KV cache block ref_count underflow")
            block.ref_count -= 1
            self._touch(block)
            if block.ref_count == 0 and block.block_hash is None:
                self._reset_block(block)

    def _reset_block(self, block: KVCacheBlock) -> None:
        if block.ref_count:
            raise RuntimeError("cannot reset referenced KV cache block")
        if (
            block.block_hash is not None
            and self._hash_to_block_id.get(block.block_hash) == block.block_id
        ):
            del self._hash_to_block_id[block.block_hash]
        block.block_hash = None
        block.token_ids = ()
        block.computed = False
        block.last_access = 0
        self._free_block_ids.append(block.block_id)

    def _acquire(self, block: KVCacheBlock) -> None:
        block.ref_count += 1
        self._touch(block)

    def _touch(self, block: KVCacheBlock) -> None:
        self._clock += 1
        block.last_access = self._clock

    def _slot(self, block_ids: list[int], position: int) -> int:
        logical_block = position // self.config.block_size
        offset = position % self.config.block_size
        return block_ids[logical_block] * self.config.block_size + offset

    def _hash_block(self, parent_hash: bytes, token_ids: tuple[int, ...]) -> bytes:
        digest = hashlib.blake2b(digest_size=16)
        digest.update(self.model_fingerprint)
        digest.update(parent_hash)
        for token_id in token_ids:
            digest.update(int(token_id).to_bytes(8, "little", signed=True))
        return digest.digest()

    def _state(self, request_id: str) -> RequestKVCacheState:
        try:
            return self._requests[request_id]
        except KeyError as error:
            raise KeyError(f"request has no KV cache state: {request_id}") from error
