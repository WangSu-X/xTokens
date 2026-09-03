"""Shared KV-cache data-plane contracts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class KVCacheMetadata:
    request_id: str
    is_prefill: bool
    context_len: int
    query_len: int
    block_table: tuple[int, ...]
    slot_mapping: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.context_len < 1:
            raise ValueError("KV cache context_len must be positive")
        if self.query_len < 1 or self.query_len > self.context_len:
            raise ValueError("KV cache query_len must be in [1, context_len]")
        if len(self.slot_mapping) != self.query_len:
            raise ValueError("slot_mapping must contain one slot per query token")
        if not self.block_table:
            raise ValueError("block_table must not be empty")


@dataclass(frozen=True, slots=True)
class KVCacheSpec:
    num_layers: int
    block_size: int
    num_kv_heads: int
    head_dim: int
    dtype_bytes: int

    @property
    def bytes_per_block(self) -> int:
        return (
            self.num_layers
            * 2
            * self.block_size
            * self.num_kv_heads
            * self.head_dim
            * self.dtype_bytes
        )
