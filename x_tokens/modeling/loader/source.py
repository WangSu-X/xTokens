"""Resolve local directories and Hugging Face Hub IDs to checkpoint paths."""

from __future__ import annotations

from pathlib import Path


def resolve_model_path(model: str, *, local_files_only: bool = False) -> Path:
    source = Path(model).expanduser()
    if source.is_dir():
        return source.resolve()
    if source.exists():
        raise ValueError(f"model source must be a directory: {source}")

    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError("Hub model resolution requires huggingface_hub") from error
    return Path(
        snapshot_download(
            repo_id=model,
            local_files_only=local_files_only,
            allow_patterns=["config.json", "*.safetensors", "*.safetensors.index.json"],
        )
    ).resolve()
