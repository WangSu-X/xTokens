"""Fetch SPEED-Bench and GSM8K and prepare exact-length prompts."""

from __future__ import annotations

import hashlib
import json
import random
import shutil
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from .support import ROOT, read_jsonl, write_json, write_jsonl

SPEED_REVISION = "fa4e669bee2a4393c56aa8ed05a7c667897f9590"
GSM_REVISION = "3101c7d5072418e28b9008a6636bde82a006892c"
SPEED_BUCKETS = {1024: "1k", 8192: "8k", 16384: "16k"}
PLACEHOLDER = "FULL BENCHMARK DATA SHOULD BE FETCHED"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fetch(path: Path, urls: list[str]) -> None:
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    errors = []
    for url in urls:
        try:
            with urlopen(url, timeout=20) as response, partial.open("wb") as stream:
                shutil.copyfileobj(response, stream)
            partial.replace(path)
            return
        except (OSError, URLError) as error:
            errors.append(f"{url}: {error}")
            partial.unlink(missing_ok=True)
    raise RuntimeError("\n".join(errors))


def exact_text(tokenizer, text: str, length: int) -> tuple[str, list[int]]:
    ids = tokenizer.encode(text, add_special_tokens=True)
    if len(ids) < length:
        text += " river" * (length - len(ids) + 4)
        ids = tokenizer.encode(text, add_special_tokens=True)
    rendered = tokenizer.decode(ids[:length], clean_up_tokenization_spaces=False)
    actual = tokenizer.encode(rendered, add_special_tokens=True)
    if len(actual) != length:
        raise ValueError(
            f"expected {length} input tokens after round trip, got {len(actual)}"
        )
    return rendered, actual


def prepare(model: str, *, seed: int, count: int, mirror: str | None = None) -> Path:
    from pyarrow import parquet
    from transformers import AutoTokenizer

    raw = ROOT / "data/raw"
    output = ROOT / "data/prepared"
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    manifest = {
        "speed_bench_revision": SPEED_REVISION,
        "gsm8k_revision": GSM_REVISION,
        "tokenizer": str(Path(model).resolve()),
        "tokenizer_sha256": sha256(Path(model) / "tokenizer.json"),
        "seed": seed,
        "cases": {},
    }
    for length, bucket in SPEED_BUCKETS.items():
        rng = random.Random(seed + length)
        path = raw / "speed-bench" / f"{bucket}.parquet"
        suffix = (
            f"/datasets/nvidia/SPEED-Bench/resolve/{SPEED_REVISION}"
            f"/throughput_{bucket}/test-00000-of-00001.parquet"
        )
        urls = ["https://huggingface.co" + suffix]
        if mirror:
            urls.append(mirror.rstrip("/") + suffix)
        fetch(path, urls)
        table = parquet.read_table(path, columns=["question_id", "category", "turns"])
        candidates = [
            row
            for row in table.to_pylist()
            if row["turns"] and not row["turns"][0].lstrip().startswith(PLACEHOLDER)
        ]
        rng.shuffle(candidates)
        source = {
            "kind": "speed-bench",
            "bucket": bucket,
            "sha256": sha256(path),
            "source_count": len(table),
            "usable_text_count": len(candidates),
            "length_adjustment": "truncate right or pad with repeated ' river'",
        }
        rows = []
        for candidate in candidates:
            if len(rows) == count:
                break
            try:
                text, tokens = exact_text(tokenizer, candidate["turns"][0], length)
            except ValueError:
                continue
            rows.append(
                {
                    "id": candidate["question_id"],
                    "text": text,
                    "category": candidate["category"],
                    "input_tokens": len(tokens),
                    "token_ids_sha256": hashlib.sha256(
                        json.dumps(tokens).encode()
                    ).hexdigest(),
                }
            )
        if len(rows) != count:
            raise ValueError(
                f"only {len(rows)} usable exact-length prompts for {length}"
            )
        file = output / f"input-{length}.jsonl"
        write_jsonl(file, rows)
        manifest["cases"][length] = {
            **source,
            "file": file.name,
            "count": count,
            "prepared_sha256": sha256(file),
        }
    for split, size in (("train", 7473), ("test", 1319)):
        path = raw / "gsm8k" / f"{split}.jsonl"
        fetch(
            path,
            [
                f"https://raw.githubusercontent.com/openai/grade-school-math/{GSM_REVISION}/grade_school_math/data/{split}.jsonl",
            ],
        )
        if len(read_jsonl(path)) != size:
            raise ValueError(f"unexpected GSM8K {split} size")
        manifest[f"gsm8k_{split}_sha256"] = sha256(path)
    path = output / "manifest.json"
    write_json(path, manifest)
    return path
