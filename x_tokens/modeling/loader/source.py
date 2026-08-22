"""Resolve local directories and Hugging Face Hub IDs to immutable sources."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from x_tokens.modeling.config import ModelLoadConfig


@dataclass(frozen=True, slots=True)
class ResolvedModelSource:
    model_id: str
    path: str
    config_path: str


def resolve_model_source(config: ModelLoadConfig) -> ResolvedModelSource:
    source = Path(config.model).expanduser()
    if source.is_dir():
        resolved = source.resolve()
    elif source.exists():
        raise ValueError(f"model source must be a directory: {source}")
    else:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as error:
            raise RuntimeError(
                "Hub model resolution requires huggingface_hub"
            ) from error
        resolved = Path(
            snapshot_download(
                repo_id=config.model,
                revision=config.revision,
                cache_dir=config.cache_dir,
                local_files_only=config.local_files_only,
                allow_patterns=(
                    "config.json",
                    "*.safetensors",
                    "*.safetensors.index.json",
                ),
            )
        ).resolve()
    config_path = resolved / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"model config does not exist: {config_path}")
    return ResolvedModelSource(config.model, str(resolved), str(config_path))
