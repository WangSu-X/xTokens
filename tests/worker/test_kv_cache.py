from __future__ import annotations

import torch

from x_tokens.kv_cache import KVCacheMetadata, KVCacheSpec
from x_tokens.worker import GPUKVCache


def test_gpu_kv_cache_writes_slots_and_gathers_logical_block_order() -> None:
    cache = GPUKVCache(
        KVCacheSpec(1, 2, 1, 2, 4),
        num_blocks=3,
        device="cpu",
        dtype=torch.float32,
    )
    metadata = (
        KVCacheMetadata(
            "request",
            True,
            context_len=3,
            query_len=3,
            block_table=(2, 0),
            slot_mapping=(4, 5, 0),
        ),
    )
    key = torch.tensor([[[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]]]])
    value = key + 10

    cache.write(0, metadata, key, value)
    gathered_key, gathered_value = cache.gather(0, metadata)

    assert torch.equal(gathered_key[0, :, 0], key[0, :, 0])
    assert torch.equal(gathered_value[0, :, 0], value[0, :, 0])
