"""Configuration for loading a resolved checkpoint."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ModelLoadConfig:
    model_path: Path
    hf_config: Mapping[str, Any]
    dtype: str = "auto"
    strict_weights: bool = True
