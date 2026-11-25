"""Immutable model loading configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelLoadConfig:
    model: str
    revision: str | None = None
    cache_dir: str | None = None
    local_files_only: bool = False
    dtype: str = "auto"
    strict_weights: bool = True

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("model must not be empty")
