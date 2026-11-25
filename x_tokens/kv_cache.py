"""Shared KV-cache data-plane contracts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class KVCacheSlots:
    context_len: int
    block_ids: tuple[int, ...]
    slot_mapping: tuple[int, ...]

    @property
    def query_len(self) -> int:
        return len(self.slot_mapping)


@dataclass(frozen=True, slots=True)
class LinearAttentionCacheSpec:
    num_layers: int
    conv_channels: int
    conv_history: int
    num_heads: int
    key_head_dim: int
    value_head_dim: int

    def bytes_per_block(self, dtype_bytes: int) -> int:
        return self.num_layers * (
            self.conv_channels * self.conv_history * dtype_bytes
            + self.num_heads * self.key_head_dim * self.value_head_dim * 4
        )


@dataclass(frozen=True, slots=True)
class KVCacheSpec:
    num_layers: int
    block_size: int
    num_kv_heads: int
    head_dim: int
    dtype_bytes: int
    linear_attention: LinearAttentionCacheSpec | None = None

    @property
    def bytes_per_block(self) -> int:
        attention_bytes = (
            self.num_layers
            * 2
            * self.block_size
            * self.num_kv_heads
            * self.head_dim
            * self.dtype_bytes
        )
        if self.linear_attention is None:
            return attention_bytes
        return attention_bytes + self.linear_attention.bytes_per_block(self.dtype_bytes)
