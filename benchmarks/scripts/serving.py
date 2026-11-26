"""AIPerf-backed OpenAI streaming workload matrix."""

from __future__ import annotations

import itertools
import json
import os
import random
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from .support import (
    INSTRUMENTS,
    ROOT,
    completion,
    percentile,
    read_jsonl,
    write_json,
    write_jsonl,
)


def metric(data, name, field):
    return data.get(name, {}).get(field)


def validate_metrics(
    data: dict, requests: int, input_tokens: int, output_tokens: int
) -> list[str]:
    problems = []
    if (
        data.get("is_complete") is not True
        or metric(data, "request_count", "avg") != requests
    ):
        problems.append("request count or completion status differs")
    for name, expected in (
        ("input_sequence_length", input_tokens),
        ("output_sequence_length", output_tokens),
    ):
        if any(metric(data, name, field) != expected for field in ("min", "max")):
            problems.append(f"{name} outside fixed length {expected}")
    return problems


def request_lengths(input_cycle: list[int], requests: int, warmup: int) -> list[int]:
    return [
        *itertools.islice(itertools.cycle(input_cycle), warmup),
        *itertools.islice(itertools.cycle(input_cycle), requests),
    ]


def select_requests(rows, lengths: list[int], seed: int) -> list[dict]:
    samples = {}
    for length, count in Counter(lengths).items():
        if count > len(rows[length]):
            raise ValueError(f"{length}-token bucket needs {count} prepared samples")
        samples[length] = iter(random.Random(seed + length).sample(rows[length], count))
    return [next(samples[length]) for length in lengths]


def summarize_requests(records, expected: dict[str, int], output_tokens: int) -> dict:
    """Join profiling records to submitted requests, including missing responses."""
    counts = Counter(expected.values())
    latencies = defaultdict(list)
    seen = set()
    problems = []
    for record in records:
        metadata = record["metadata"]
        if metadata["phase_kind"] != "profiling":
            continue
        session = metadata["conversation_id"]
        if session not in expected or session in seen:
            problems.append(f"unexpected or duplicate profiling request {session}")
            continue
        seen.add(session)
        if record.get("error") or metadata.get("was_cancelled"):
            continue
        length = expected[session]
        metrics = record.get("metrics", {})
        if (
            metric(metrics, "usage_prompt_tokens", "value") != length
            or metric(metrics, "usage_completion_tokens", "value") != output_tokens
        ):
            problems.append(f"server token usage differs for {session}")
            continue
        ttft = metric(metrics, "time_to_first_token", "value")
        if ttft is None:
            problems.append(f"missing TTFT for {session}")
            continue
        latencies[length].append(ttft)
    by_length = {
        length: {
            "requests": count,
            "requests_ok": len(latencies[length]),
            "ttft_p95_ms": percentile(latencies[length], 0.95),
        }
        for length, count in sorted(counts.items())
    }
    if sum(row["requests_ok"] for row in by_length.values()) != len(expected):
        problems.append("missing or invalid request records")
    return {"problems": problems, "by_input_tokens": by_length}


def evaluate(
    url, runtime, case, output: Path, *, aiperf: str, requests=None, repeats=None
) -> None:
    requests = case["requests_per_repeat"] if requests is None else requests
    repeats = case["repeats"] if repeats is None else repeats
    env = os.environ.copy()
    env["XTOKENS_BENCH_AIPERF"] = "1"
    paths = [str(INSTRUMENTS), env.get("PYTHONPATH", "")]
    env["PYTHONPATH"] = os.pathsep.join(path for path in paths if path)
    matrix = case["matrix"]
    input_cycle = case.get("input_cycle")
    input_lengths = sorted(set(input_cycle or matrix["input_tokens"]))
    rows = {}
    for source in input_lengths:
        rows[source] = read_jsonl(ROOT / "data/prepared" / f"input-{source}.jsonl")
        check = completion(
            url,
            runtime["served_model"],
            rows[source][0]["text"],
            max_tokens=1,
            ignore_eos=True,
            temperature=0,
        )["usage"]
        if (check["prompt_tokens"], check["completion_tokens"]) != (source, 1):
            raise ValueError(
                f"server reported unexpected lengths for {source}: {check}"
            )
    groups = [input_cycle] if input_cycle else [[length] for length in input_lengths]
    for group, target in itertools.product(groups, matrix["output_tokens"]):
        lengths = request_lengths(group, requests, case["warmup"])
        scenario = "mix" if input_cycle else f"i{group[0]}"
        for concurrency, repeat in itertools.product(
            matrix["concurrency"], range(repeats)
        ):
            cell = f"{scenario}-o{target}-c{concurrency}"
            folder = output / cell / f"r{repeat + 1}"
            selected = select_requests(rows, lengths, case["seed"] + 100 * repeat)
            session_ids = [f"r{index:04d}" for index in range(len(selected))]
            profile_inputs = dict(
                zip(session_ids[case["warmup"] :], lengths[case["warmup"] :])
            )
            dataset = folder / "requests.jsonl"
            write_jsonl(
                dataset,
                (
                    {
                        "session_id": session,
                        "text": row["text"],
                        "output_length": target,
                    }
                    for session, row in zip(session_ids, selected)
                ),
            )
            artifact = folder / "aiperf"
            argv = [
                aiperf,
                "profile",
                "--model",
                runtime["served_model"],
                "--url",
                url,
                "--endpoint-type",
                "completions",
                "--streaming",
                "--tokenizer",
                runtime["model"],
                "--use-server-token-count",
                "--input-file",
                str(dataset),
                "--custom-dataset-type",
                "single_turn",
                "--dataset-sampling-strategy",
                "sequential",
                "--random-seed",
                str(case["seed"]),
                "--concurrency",
                str(concurrency),
                "--request-count",
                str(requests),
                "--extra-inputs",
                "temperature:0",
                "--extra-inputs",
                "ignore_eos:true",
                "--ui",
                "none",
                "--no-server-metrics",
                "--no-auto-plot",
                "--request-timeout-seconds",
                str(case["request_timeout_seconds"]),
                "--export-level",
                "records",
                "--artifact-dir",
                str(artifact),
            ]
            if case["warmup"]:
                argv.extend(["--warmup-request-count", str(case["warmup"])])
            if case.get("slo_ms"):
                argv.extend(
                    [
                        "--goodput",
                        *(f"{name}:{value}" for name, value in case["slo_ms"].items()),
                    ]
                )
            write_json(
                folder / "case.json",
                {
                    "input_tokens": input_lengths if input_cycle else group[0],
                    "output_tokens": target,
                    "concurrency": concurrency,
                    "repeat": repeat + 1,
                    "requests": requests,
                    "argv": argv,
                    "sample_ids": [row["id"] for row in selected],
                    "profile_inputs": profile_inputs,
                    "prefix_cache": "disabled",
                },
            )
            print(f"{cell} r{repeat + 1}", flush=True)
            with (folder / "client.log").open("w") as stream:
                proc = subprocess.run(
                    argv, env=env, stdout=stream, stderr=subprocess.STDOUT, check=False
                )
            export = artifact / "profile_export_aiperf.json"
            records = artifact / "profile_export.jsonl"
            validation = summarize_requests(
                read_jsonl(records) if records.is_file() else [], profile_inputs, target
            )
            issues = validation["problems"]
            if export.is_file():
                data = json.loads(export.read_text())
                if not input_cycle:
                    issues.extend(validate_metrics(data, requests, group[0], target))
                elif (
                    data.get("is_complete") is not True
                    or metric(data, "request_count", "avg") != requests
                ):
                    issues.append("request count or completion status differs")
            else:
                issues.append("AIPerf did not produce an aggregate export")
            if proc.returncode:
                issues.append(f"AIPerf exit code {proc.returncode}")
            write_json(folder / "validation.json", validation)
