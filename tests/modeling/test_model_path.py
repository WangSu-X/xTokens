from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest

from x_tokens.modeling.loader import resolve_model_path


def test_resolve_model_path_resolves_a_local_directory(tmp_path: Path) -> None:
    model_path = tmp_path / "checkpoint"
    model_path.mkdir()
    model_alias = tmp_path / "alias"
    model_alias.symlink_to(model_path, target_is_directory=True)

    assert resolve_model_path(str(model_alias)) == model_path.resolve()


def test_resolve_model_path_rejects_a_weights_file(tmp_path: Path) -> None:
    weights = tmp_path / "model.safetensors"
    weights.touch()

    with pytest.raises(ValueError, match="model source must be a directory"):
        resolve_model_path(str(weights))


def test_resolve_model_path_uses_the_hub_snapshot(tmp_path: Path, monkeypatch) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    download = Mock(return_value=str(snapshot))
    monkeypatch.setattr("huggingface_hub.snapshot_download", download)

    assert resolve_model_path("example/checkpoint", local_files_only=True) == snapshot

    download.assert_called_once_with(
        repo_id="example/checkpoint",
        local_files_only=True,
        allow_patterns=["config.json", "*.safetensors", "*.safetensors.index.json"],
    )
