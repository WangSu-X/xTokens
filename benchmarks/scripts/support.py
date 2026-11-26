"""Shared benchmark I/O and statistics."""

from __future__ import annotations

import json
import math
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
INSTRUMENTS = ROOT / "scripts" / "instruments"


def configuration(name: str) -> dict:
    path = (
        ROOT / "runtime.yaml"
        if name == "runtime"
        else ROOT / "benches" / f"{name}.yaml"
    )
    with path.open() as stream:
        return yaml.safe_load(stream)


def read_json(path: Path):
    return json.loads(path.read_text())


def result_directory(root: Path, suite: str) -> Path:
    directory = root / suite
    legacy = directory / "xtokens"
    return legacy if legacy.is_dir() else directory


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def percentile(values, quantile):
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * quantile
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def completion(url: str, model: str, prompt: str, **options) -> dict:
    request = urllib.request.Request(
        f"{url}/v1/completions",
        data=json.dumps({"model": model, "prompt": prompt, **options}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=240) as response:
        return json.load(response)
