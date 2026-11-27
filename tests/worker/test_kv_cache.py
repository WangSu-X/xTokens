from __future__ import annotations

import pytest
import torch

from x_tokens.modeling.kv_cache import (
    ForwardContext,
    FullAttentionCacheSpec,
    LinearAttentionCacheSpec,
    get_forward_context,
    set_forward_context,
)
from x_tokens.worker.kv_cache import allocate_kv_cache


def test_cache_allocation_matches_per_layer_specs() -> None:
    specs = {
        "full": FullAttentionCacheSpec(2, 1, 4, torch.bfloat16),
        "linear": LinearAttentionCacheSpec(8, 3, 2, 4, 6, torch.bfloat16),
    }
    caches = allocate_kv_cache(specs, 3, torch.device("cpu"))
    full_cache = caches["full"]
    conv_cache, recurrent_cache = caches["linear"]
    assert full_cache.shape == (2, 3, 2, 1, 4)
    assert conv_cache.shape == (3, 8, 3)
    assert recurrent_cache.shape == (3, 2, 4, 6)
    assert full_cache.dtype == conv_cache.dtype == torch.bfloat16
    assert recurrent_cache.dtype == torch.float32
    assert sum(
        tensor.numel() * tensor.element_size()
        for tensor in (full_cache, conv_cache, recurrent_cache)
    ) == 3 * sum(spec.bytes_per_block for spec in specs.values())


def test_forward_context_is_restored_on_error() -> None:
    outer = ForwardContext({}, None, None)
    inner = ForwardContext({}, None, None)
    with set_forward_context(outer):
        with (
            pytest.raises(RuntimeError, match="failed forward"),
            set_forward_context(inner),
        ):
            assert get_forward_context() is inner
            raise RuntimeError("failed forward")
        assert get_forward_context() is outer
    with pytest.raises(LookupError):
        get_forward_context()
