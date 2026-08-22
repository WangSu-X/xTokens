"""Normalization layers used by supported model architectures."""

from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        *,
        eps: float,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(hidden_size, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        values = hidden_states.float()
        variance = values.square().mean(dim=-1, keepdim=True)
        normalized = values * torch.rsqrt(variance + self.eps)
        return self.weight * normalized.to(input_dtype)
