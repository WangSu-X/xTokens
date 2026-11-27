"""Model-side cache layout and per-step attention metadata."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class FullAttentionCacheSpec:
    block_size: int
    num_kv_heads: int
    head_dim: int
    dtype: torch.dtype

    @property
    def bytes_per_block(self) -> int:
        return (
            2
            * self.block_size
            * self.num_kv_heads
            * self.head_dim
            * self.dtype.itemsize
        )


@dataclass(frozen=True, slots=True)
class LinearAttentionCacheSpec:
    conv_channels: int
    conv_history: int
    num_heads: int
    key_head_dim: int
    value_head_dim: int
    dtype: torch.dtype

    @property
    def bytes_per_block(self) -> int:
        return (
            self.conv_channels * self.conv_history * self.dtype.itemsize
            + self.num_heads
            * self.key_head_dim
            * self.value_head_dim
            * torch.float32.itemsize
        )


type KVCacheSpec = FullAttentionCacheSpec | LinearAttentionCacheSpec
type LayerCache = Tensor | tuple[Tensor, Tensor]


@dataclass(frozen=True, slots=True)
class FlashAttentionMetadata:
    seq_lens: tuple[int, ...]
    max_query_len: int
    max_seq_len: int
    query_start_loc: Tensor
    seq_start_loc: Tensor
    query_indices: Tensor
    slot_mapping: Tensor
    context_slots: Tensor
    causal_mask: Tensor | None


@dataclass(frozen=True, slots=True)
class FlashLinearAttentionMetadata:
    prefill_rows: tuple[int, ...]
    prefill_query_lens: tuple[int, ...]
    prefill_read_indices: tuple[int | None, ...]
    prefill_spans: tuple[tuple[tuple[int, int | None], ...], ...]
    decode_rows: Tensor
    decode_read_indices: Tensor
    decode_write_indices: Tensor
    prefill_token_indices: Tensor | None = None
    prefill_round_token_indices: tuple[Tensor, ...] = ()
    prefill_round_cu_seqlens: tuple[Tensor | None, ...] = ()
    prefill_round_read_indices: tuple[Tensor, ...] = ()
    prefill_round_write_indices: tuple[Tensor, ...] = ()


@dataclass(slots=True)
class ForwardContext:
    kv_cache_dict: dict[str, LayerCache]
    full_attention: FlashAttentionMetadata | None
    linear_attention: FlashLinearAttentionMetadata | None
    rope: tuple[Tensor, Tensor] = field(init=False)


current_forward_context = ContextVar[ForwardContext]("forward_context")


def get_forward_context() -> ForwardContext:
    return current_forward_context.get()


@contextmanager
def set_forward_context(context: ForwardContext) -> Iterator[None]:
    token = current_forward_context.set(context)
    try:
        yield
    finally:
        current_forward_context.reset(token)
