"""Startup configuration for an EngineCore."""

from __future__ import annotations

from dataclasses import dataclass

from x_tokens.config import XTokensConfig


@dataclass(frozen=True, slots=True)
class EngineCoreConfig:
    """Runtime settings and tokenizer IDs for building an EngineCore."""

    xtokens_config: XTokensConfig
    pad_token_id: int
    eos_token_ids: frozenset[int]

    @property
    def served_model_name(self) -> str:
        return self.xtokens_config.model_config.served_model_name
