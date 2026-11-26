"""Single-request model timings with fixed model decode inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .support import ROOT, completion, percentile, read_jsonl, write_json


def evaluate(url, runtime, case, output: Path, *, samples: int | None = None):
    from transformers import AutoTokenizer

    samples = case["samples"] if samples is None else samples
    tokenizer = AutoTokenizer.from_pretrained(runtime["model"], local_files_only=True)
    (decode_token,) = tokenizer.encode(" hello", add_special_tokens=False)
    decode_steps = case["decode_steps"]
    generated = tokenizer.decode([decode_token] * (decode_steps + 1))
    trace = output / "steps.jsonl"
    trace.unlink(missing_ok=True)
    control = output / "model-control.json"
    for length in case["input_tokens"]:
        rows = read_jsonl(ROOT / "data/prepared" / f"input-{length}.jsonl")
        for index in range(samples + case["warmup"]):
            row = rows[index]
            write_json(
                control,
                {
                    "trace": str(trace),
                    "decode_token": decode_token,
                    "prompt_tokens": length,
                    "sample": index,
                    "sample_id": row["id"],
                    "token_ids_sha256": row["token_ids_sha256"],
                    "warmup": index < case["warmup"],
                },
            )
            try:
                response = completion(
                    url,
                    runtime["served_model"],
                    row["text"],
                    max_tokens=decode_steps + 1,
                    ignore_eos=True,
                    temperature=0,
                )
            finally:
                control.unlink(missing_ok=True)
            assert response["usage"]["prompt_tokens"] == length
            assert response["usage"]["completion_tokens"] == decode_steps + 1
            assert response["choices"][0]["text"] == generated
        print(f"xTokens model i{length} {samples} samples", flush=True)
    rows = [row for row in read_jsonl(trace) if not row["warmup"]]
    summary = []
    operation = "forward_with_head"
    for length in case["input_tokens"]:
        prefills, decodes = [], []
        step_shapes = []
        for sample in range(case["warmup"], samples + case["warmup"]):
            steps = [
                row
                for row in rows
                if row["prompt_tokens"] == length and row["sample"] == sample
            ]
            forward = [row for row in steps if row["operation"] == operation]
            prefill = [row for row in forward if row["is_prefill"]]
            decode = [row for row in forward if not row["is_prefill"]]
            assert sum(row["query_tokens"] for row in prefill) == length
            assert len(decode) == decode_steps
            prompt_ids = [
                token_id for row in prefill for token_id in row["actual_prompt_tokens"]
            ]
            assert (
                hashlib.sha256(json.dumps(prompt_ids).encode()).hexdigest()
                == prefill[0]["token_ids_sha256"]
            )
            assert all(row["actual_decode_tokens"] == [decode_token] for row in decode)
            prefills.append(sum(row["gpu_ms"] for row in prefill))
            decodes.extend(row["gpu_ms"] for row in decode)
            step_shapes.append([row["query_tokens"] for row in prefill])
        p50 = percentile(prefills, 0.5)
        summary.append(
            {
                "input_tokens": length,
                "prefill_gpu_p50_ms": p50,
                "prefill_gpu_p95_ms": percentile(prefills, 0.95),
                "prefill_tokens_per_sec_p50": 1000 * length / p50,
                "prefill_with_head_p50_ms": p50,
                "decode_gpu_p50_ms": percentile(decodes, 0.5),
                "decode_gpu_p95_ms": percentile(decodes, 0.95),
                "decode_with_head_p50_ms": percentile(decodes, 0.5),
                "decode_with_head_p95_ms": percentile(decodes, 0.95),
                "prefill_query_shapes": step_shapes,
                "prefill_samples": len(prefills),
                "decode_samples": len(decodes),
                "timed_operation": operation,
            }
        )
    write_json(output / "summary.json", summary)
