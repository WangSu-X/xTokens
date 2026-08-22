"""Vocabulary-parallel embedding and language-model head."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from x_tokens.distributed import CollectiveGroup


class VocabParallelEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        *,
        group: CollectiveGroup,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.group = group
        self.padded_vocab_size = (
            (vocab_size + group.world_size - 1) // group.world_size
        ) * group.world_size
        self.local_vocab_size = self.padded_vocab_size // group.world_size
        self.vocab_start = group.rank_in_group * self.local_vocab_size
        self.vocab_end = self.vocab_start + self.local_vocab_size
        self.weight = nn.Parameter(
            torch.empty(self.local_vocab_size, hidden_size, device=device, dtype=dtype)
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        outside = (input_ids < self.vocab_start) | (input_ids >= self.vocab_end)
        local_ids = (input_ids - self.vocab_start).masked_fill(outside, 0)
        output = F.embedding(local_ids, self.weight)
        output.masked_fill_(outside.unsqueeze(-1), 0)
        return self.group.all_reduce(output)

    def load_weight(self, loaded_weight: torch.Tensor) -> None:
        expected = (self.vocab_size, self.hidden_size)
        if tuple(loaded_weight.shape) != expected:
            raise ValueError(
                f"embedding shape mismatch: expected {expected}, "
                f"got {tuple(loaded_weight.shape)}"
            )
        self.weight.data.zero_()
        source_end = min(self.vocab_end, self.vocab_size)
        if source_end > self.vocab_start:
            self.weight.data[: source_end - self.vocab_start].copy_(
                loaded_weight[self.vocab_start : source_end]
            )


class ParallelLMHead(VocabParallelEmbedding):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        local_logits = F.linear(hidden_states, self.weight)
        logits = self.group.all_gather(local_logits, dim=-1)
        return logits[..., : self.vocab_size]
