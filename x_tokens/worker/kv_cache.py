"""Rank-local block-based GPU KV-cache storage."""

from __future__ import annotations

from typing import Any

from x_tokens.kv_cache import KVCacheMetadata, KVCacheSpec


class GPUKVCache:
    def __init__(
        self,
        spec: KVCacheSpec,
        *,
        num_blocks: int,
        device: Any,
        dtype: Any,
        torch_module: Any | None = None,
    ) -> None:
        if torch_module is None:
            import torch

            torch_module = torch
        if num_blocks < 1:
            raise ValueError("num_blocks must be positive")
        self.spec = spec
        self.num_blocks = num_blocks
        self._torch = torch_module
        self.storage = torch_module.empty(
            spec.num_layers,
            2,
            num_blocks,
            spec.block_size,
            spec.num_kv_heads,
            spec.head_dim,
            device=device,
            dtype=dtype,
        )

    @property
    def allocated_bytes(self) -> int:
        return self.storage.numel() * self.storage.element_size()

    def write(
        self,
        layer_index: int,
        metadata: tuple[KVCacheMetadata, ...],
        key: Any,
        value: Any,
    ) -> None:
        if key.shape != value.shape or key.ndim != 4:
            raise ValueError("KV tensors must share [batch, query, heads, dim] shape")
        if key.shape[0] != len(metadata):
            raise ValueError("KV metadata batch size mismatch")
        key_cache = self.storage[layer_index, 0]
        value_cache = self.storage[layer_index, 1]
        for batch_index, item in enumerate(metadata):
            slots = self._torch.tensor(
                item.slot_mapping,
                dtype=self._torch.long,
                device=self.storage.device,
            )
            block_ids = slots // self.spec.block_size
            offsets = slots % self.spec.block_size
            source_key = key[batch_index, -item.query_len :]
            source_value = value[batch_index, -item.query_len :]
            key_cache[block_ids, offsets] = source_key
            value_cache[block_ids, offsets] = source_value

    def gather(
        self,
        layer_index: int,
        metadata: tuple[KVCacheMetadata, ...],
    ) -> tuple[Any, Any]:
        max_context_len = max(item.context_len for item in metadata)
        shape = (
            len(metadata),
            max_context_len,
            self.spec.num_kv_heads,
            self.spec.head_dim,
        )
        key = self._torch.zeros(
            shape, dtype=self.storage.dtype, device=self.storage.device
        )
        value = self._torch.zeros_like(key)
        for batch_index, item in enumerate(metadata):
            block_ids = self._torch.tensor(
                item.block_table,
                dtype=self._torch.long,
                device=self.storage.device,
            )
            if bool((block_ids < 0).any()) or bool(
                (block_ids >= self.num_blocks).any()
            ):
                raise ValueError("KV cache block table contains an invalid block ID")
            gathered_key = self.storage[layer_index, 0].index_select(0, block_ids)
            gathered_value = self.storage[layer_index, 1].index_select(0, block_ids)
            key[batch_index, : item.context_len] = gathered_key.reshape(
                -1, self.spec.num_kv_heads, self.spec.head_dim
            )[: item.context_len]
            value[batch_index, : item.context_len] = gathered_value.reshape(
                -1, self.spec.num_kv_heads, self.spec.head_dim
            )[: item.context_len]
        return key, value

    def clear(self) -> None:
        self.storage.zero_()
