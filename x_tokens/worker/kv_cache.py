"""Allocate rank-local KV-cache tensors."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from x_tokens.modeling.kv_cache import FullAttentionCacheSpec, KVCacheSpec, LayerCache


def allocate_kv_cache(
    specs: Mapping[str, KVCacheSpec], num_blocks: int, device: torch.device
) -> dict[str, LayerCache]:
    if num_blocks < 1:
        raise ValueError("num_blocks must be positive")

    caches: dict[str, LayerCache] = {}
    for name, spec in specs.items():
        if isinstance(spec, FullAttentionCacheSpec):
            caches[name] = torch.empty(
                (2, num_blocks, spec.block_size, spec.num_kv_heads, spec.head_dim),
                device=device,
                dtype=spec.dtype,
            )
        else:
            caches[name] = (
                torch.empty(
                    (num_blocks, spec.conv_channels, spec.conv_history),
                    device=device,
                    dtype=spec.dtype,
                ),
                torch.empty(
                    (
                        num_blocks,
                        spec.num_heads,
                        spec.key_head_dim,
                        spec.value_head_dim,
                    ),
                    device=device,
                    dtype=torch.float32,
                ),
            )
    return caches
