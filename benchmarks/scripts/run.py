"""Prepare reproducible inputs and run the xTokens benchmark suites."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from . import accuracy, data, model, serving
from .servers import serving_engine
from .support import ROOT, configuration, read_json, result_directory, write_json

SUITES = ("model", "serving", "mixed", "accuracy")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=("prepare", *SUITES, "all"))
    parser.add_argument("--engine-python", default=sys.executable)
    parser.add_argument("--aiperf", default="aiperf")
    parser.add_argument("--model")
    parser.add_argument(
        "--gpus", type=lambda value: [int(part) for part in value.split(",")]
    )
    parser.add_argument("--port", type=int, default=8021)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--mirror", help="optional fallback Hugging Face dataset mirror"
    )
    parser.add_argument("--dataset-count", type=int, default=128)
    parser.add_argument("--requests", type=int)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--model-samples", type=int)
    parser.add_argument("--gsm-limit", type=int)
    parser.add_argument("--gsm-concurrency", type=int)
    parser.add_argument("--gsm-shard-count", type=int, default=1)
    parser.add_argument("--gsm-shard-index", type=int, default=0)
    parser.add_argument("--max-num-seqs", type=int)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--kv-cache-memory-bytes", type=int)
    args = parser.parse_args()
    runtime = configuration("runtime")
    if args.model:
        runtime["model"] = args.model
    if args.gpus:
        runtime["gpus"] = args.gpus
    for name in ("max_num_seqs", "max_model_len", "kv_cache_memory_bytes"):
        if getattr(args, name) is not None:
            runtime[name] = getattr(args, name)
    if not 0 <= args.gsm_shard_index < args.gsm_shard_count:
        parser.error("GSM8K shard index must be within the shard count")
    if len(runtime["gpus"]) != 2 or len(set(runtime["gpus"])) != 2:
        parser.error("specify exactly two distinct GPUs")
    for name in (
        "dataset_count",
        "requests",
        "repeats",
        "model_samples",
        "gsm_limit",
        "gsm_concurrency",
    ):
        if getattr(args, name) is not None and getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    workload = configuration("serving")
    if args.warmup is not None and args.warmup < 0:
        parser.error("--warmup must be nonnegative")
    if args.suite == "prepare":
        print(
            data.prepare(
                runtime["model"],
                seed=workload["seed"],
                count=args.dataset_count,
                mirror=args.mirror,
            )
        )
        return
    output = (
        args.output or ROOT / "results" / datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    ).resolve()
    suites = SUITES if args.suite == "all" else (args.suite,)
    serving_suites = set(suites) & {"serving", "mixed"}
    if serving_suites:
        version = subprocess.check_output([args.aiperf, "--version"], text=True).strip()
        if version != runtime["aiperf_version"]:
            parser.error(f"expected AIPerf {runtime['aiperf_version']}, got {version}")
    manifest_path = ROOT / "data/prepared/manifest.json"
    if not manifest_path.is_file():
        parser.error("prepare benchmark datasets before running")
    manifest = read_json(manifest_path)
    tokenizer = Path(runtime["model"]) / "tokenizer.json"
    if (
        not tokenizer.is_file()
        or data.sha256(tokenizer) != manifest["tokenizer_sha256"]
    ):
        parser.error("prepared dataset tokenizer differs from the requested model")
    cases = {}
    for suite in suites:
        if suite == "model":
            case = configuration(suite)
            if args.model_samples is not None:
                case["samples"] = args.model_samples
            needed = dict.fromkeys(
                case["input_tokens"], case["samples"] + case["warmup"]
            )
            if max(needed) + case["decode_steps"] + 1 > runtime["max_model_len"]:
                parser.error("model input plus output exceeds --max-model-len")
        elif suite in serving_suites:
            case = {**workload, "matrix": workload["matrix"].copy()}
            matrix = case["matrix"]
            if suite == "mixed":
                mixed = configuration(suite)
                case["input_cycle"] = mixed["input_cycle"]
                matrix["concurrency"] = mixed["concurrency"]
            for field in ("warmup", "repeats"):
                if getattr(args, field) is not None:
                    case[field] = getattr(args, field)
            if args.requests is not None:
                case["requests_per_repeat"] = args.requests
            requests = case["requests_per_repeat"]
            if requests < max(matrix["concurrency"]):
                parser.error(
                    f"{suite} needs at least {max(matrix['concurrency'])} requests"
                )
            if suite == "mixed":
                needed = Counter(
                    serving.request_lengths(
                        case["input_cycle"], requests, case["warmup"]
                    )
                )
                if set(case["input_cycle"][:requests]) != set(case["input_cycle"]):
                    parser.error("mixed workload must measure every input length")
            else:
                needed = dict.fromkeys(
                    matrix["input_tokens"], requests + case["warmup"]
                )
            if max(needed) + max(matrix["output_tokens"]) > runtime["max_model_len"]:
                parser.error(f"{suite} input plus output exceeds --max-model-len")
        else:
            case = configuration(suite)
            if args.gsm_concurrency is not None:
                case["concurrency"] = [args.gsm_concurrency]
            needed = {}
            for split in ("train", "test"):
                path = ROOT / "data/raw/gsm8k" / f"{split}.jsonl"
                if (
                    not path.is_file()
                    or data.sha256(path) != manifest[f"gsm8k_{split}_sha256"]
                ):
                    parser.error(f"GSM8K {split} differs from the dataset manifest")
        for length, count in needed.items():
            prepared = manifest["cases"].get(str(length), {})
            path = manifest_path.parent / f"input-{length}.jsonl"
            if prepared.get("count", 0) < count or not path.is_file():
                parser.error(
                    f"{suite} needs {count} prepared prompts of {length} tokens"
                )
            digest = prepared.get("prepared_sha256")
            if digest and data.sha256(path) != digest:
                parser.error(
                    f"prepared prompts differ from the dataset manifest: {path}"
                )
        cases[suite] = case
    for suite in suites:
        directory = result_directory(output, suite)
        if (directory / "launch.json").exists():
            parser.error(f"result directory already contains a run: {directory}")
    output.mkdir(parents=True, exist_ok=True)
    snapshot = output / "dataset-manifest.json"
    if snapshot.exists() and snapshot.read_bytes() != manifest_path.read_bytes():
        parser.error("dataset manifest differs from the existing experiment")
    if not snapshot.exists():
        shutil.copyfile(manifest_path, snapshot)
    git_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    for suite, case in cases.items():
        directory = result_directory(output, suite)
        write_json(
            directory / "run.json",
            {
                "git_head": git_head,
                "suite": suite,
                "engine": "xtokens",
                "port": args.port,
                "gpu_devices": runtime["gpus"],
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "configuration": case,
            },
        )
        print(f"starting {suite}/xtokens", flush=True)
        with serving_engine(
            runtime,
            directory,
            python=args.engine_python,
            port=args.port,
            model_only=suite == "model",
        ) as url:
            if suite == "model":
                model.evaluate(url, runtime, case, directory)
            elif suite in serving_suites:
                serving.evaluate(url, runtime, case, directory, aiperf=args.aiperf)
            else:
                for concurrency in case["concurrency"]:
                    accuracy.evaluate(
                        url,
                        runtime,
                        {**case, "concurrency": concurrency},
                        directory / f"c{concurrency}",
                        limit=args.gsm_limit,
                        shard_index=args.gsm_shard_index,
                        shard_count=args.gsm_shard_count,
                    )


if __name__ == "__main__":
    main()
