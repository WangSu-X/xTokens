"""Rank-local block-based GPU KV-cache storage."""

from __future__ import annotations

from typing import Any

from x_tokens.kv_cache import KVCacheSlots, KVCacheSpec


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
        self.torch_module = torch_module
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
        self.conv_states = None
        self.recurrent_states = None
        linear = spec.linear_attention
        if linear is not None:
            self.conv_states = torch_module.empty(
                linear.num_layers,
                num_blocks,
                linear.conv_channels,
                linear.conv_history,
                device=device,
                dtype=dtype,
            )
            self.recurrent_states = torch_module.empty(
                linear.num_layers,
                num_blocks,
                linear.num_heads,
                linear.key_head_dim,
                linear.value_head_dim,
                device=device,
                dtype=torch_module.float32,
            )

    @property
    def allocated_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (self.storage, self.conv_states, self.recurrent_states)
            if tensor is not None
        )

    def linear_state(self, layer_index: int, block_id: int | None) -> tuple[Any, Any]:
        if self.conv_states is None or self.recurrent_states is None:
            raise ValueError("KV cache does not provide linear attention state")
        if block_id is None:
            return (
                self.torch_module.zeros_like(self.conv_states[layer_index, 0]),
                self.torch_module.zeros_like(self.recurrent_states[layer_index, 0]),
            )
        return self.conv_states[layer_index, block_id], self.recurrent_states[
            layer_index, block_id
        ]

    def write_linear_state(
        self, layer_index: int, block_id: int, conv_state: Any, recurrent_state: Any
    ) -> None:
        conv, recurrent = self.linear_state(layer_index, block_id)
        conv.copy_(conv_state)
        recurrent.copy_(recurrent_state)

    def write(
        self,
        layer_index: int,
        metadata: tuple[KVCacheSlots, ...],
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
            slots = self.torch_module.tensor(
                item.slot_mapping,
                dtype=self.torch_module.long,
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
        metadata: tuple[KVCacheSlots, ...],
    ) -> tuple[Any, Any]:
        max_context_len = max(item.context_len for item in metadata)
        shape = (
            len(metadata),
            max_context_len,
            self.spec.num_kv_heads,
            self.spec.head_dim,
        )
        key = self.torch_module.zeros(
            shape, dtype=self.storage.dtype, device=self.storage.device
        )
        value = self.torch_module.zeros_like(key)
        for batch_index, item in enumerate(metadata):
            if any(not 0 <= block_id < self.num_blocks for block_id in item.block_ids):
                raise ValueError("KV cache block table contains an invalid block ID")
            block_ids = self.torch_module.tensor(
                item.block_ids,
                dtype=self.torch_module.long,
                device=self.storage.device,
            )
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
        if self.conv_states is not None:
            self.conv_states.zero_()
        if self.recurrent_states is not None:
            self.recurrent_states.zero_()
