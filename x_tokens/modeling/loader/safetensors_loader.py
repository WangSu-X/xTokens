"""SafeTensors checkpoint indexes and weight-loading reports."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from safetensors import safe_open


@dataclass(frozen=True, slots=True)
class WeightLoadReport:
    rank: int
    world_size: int
    loaded_source_tensors: int
    loaded_parameters: int
    local_parameter_bytes: int
    missing_parameters: tuple[str, ...]
    unexpected_tensors: tuple[str, ...]


def _load_manifest(root: Path) -> dict[str, str]:
    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        with index_path.open(encoding="utf-8") as index_file:
            index = json.load(index_file)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("SafeTensors index must contain a non-empty weight_map")
        normalized = {str(key): str(value) for key, value in weight_map.items()}
        for filename in set(normalized.values()):
            _safe_checkpoint_path(root, filename)
        return normalized
    checkpoint_files = sorted(root.glob("*.safetensors"))
    if len(checkpoint_files) != 1:
        raise ValueError(
            "model directory without an index must contain exactly one SafeTensors file"
        )
    checkpoint = checkpoint_files[0]
    with safe_open(checkpoint, framework="pt", device="cpu") as shard:
        checkpoint_keys = shard.keys()
        return {key: checkpoint.name for key in checkpoint_keys}


def _safe_checkpoint_path(root: Path, filename: str) -> Path:
    path = (root / filename).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(
            f"checkpoint path escapes model directory: {filename}"
        ) from error
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint shard does not exist: {path}")
    return path
