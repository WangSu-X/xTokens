"""Summarize recorded xTokens benchmark artifacts."""

from __future__ import annotations

import argparse
import collections
import statistics
from pathlib import Path

from .serving import metric
from .support import read_json, read_jsonl, result_directory, write_json

METRICS = {
    "ttft_p50_ms": ("time_to_first_token", "p50"),
    "ttft_p95_ms": ("time_to_first_token", "p95"),
    "tpot_p50_ms": ("inter_token_latency", "p50"),
    "tpot_p95_ms": ("inter_token_latency", "p95"),
    "e2e_p50_ms": ("request_latency", "p50"),
    "e2e_p95_ms": ("request_latency", "p95"),
    "output_tokens_per_sec": ("output_token_throughput", "avg"),
    "requests_ok": ("request_count", "avg"),
    "goodput": ("goodput", "avg"),
}


def serving_summary(root: Path, suite: str = "serving") -> list[dict]:
    cells = collections.defaultdict(dict)
    for path in result_directory(root, suite).glob("*-o*-c*/r*/case.json"):
        config = read_json(path)
        folder = path.parent
        path = folder / "validation.json"
        validation = read_json(path) if path.is_file() else {"problems": ["pending"]}
        issues = validation["problems"]
        export = folder / "aiperf/profile_export_aiperf.json"
        metrics = read_json(export) if export.is_file() else {}
        if not export.is_file() and not issues:
            issues = ["missing metrics"]
        lengths = config["input_tokens"]
        inputs = (lengths,) if isinstance(lengths, int) else tuple(lengths)
        key = inputs, config["output_tokens"], config["concurrency"]
        by_length = validation.get("by_input_tokens", {})
        cells[key][config["repeat"]] = {
            "metrics": {
                name: metric(metrics, *query) for name, query in METRICS.items()
            },
            "problems": issues,
            "sample_ids": config["sample_ids"],
            "requests": config["requests"],
            "by_input_tokens": by_length,
            "artifact": str(folder.relative_to(root)),
        }
    rows = []
    for (inputs, outputs, concurrency), repeats in sorted(cells.items()):
        valid = [
            repeat["metrics"] for repeat in repeats.values() if not repeat["problems"]
        ]
        aggregated = {}
        for name in METRICS:
            values = [row[name] for row in valid if row[name] is not None]
            aggregated[name] = statistics.median(values) if values else None
        by_length = {}
        for length in inputs:
            counts = [
                repeat["by_input_tokens"][str(length)]
                for repeat in repeats.values()
                if str(length) in repeat["by_input_tokens"]
            ]
            if not counts:
                continue
            total = sum(row["requests"] for row in counts)
            success = sum(row["requests_ok"] for row in counts)
            tail = [
                row["ttft_p95_ms"] for row in counts if row["ttft_p95_ms"] is not None
            ]
            by_length[length] = {
                "requests": total,
                "requests_ok": success,
                "failure_rate": (total - success) / total,
                "ttft_p95_ms": statistics.median(tail) if tail else None,
            }
        rows.append(
            {
                "input_tokens": list(inputs) if suite == "mixed" else inputs[0],
                "output_tokens": outputs,
                "concurrency": concurrency,
                "valid_repeats": len(valid),
                "metrics": aggregated,
                "by_input_tokens": by_length,
                "repeats": repeats,
            }
        )
    return rows


def accuracy_results(folders):
    rows = {}
    expected = set()
    configured_total = None
    settings = None
    for folder in folders:
        details = folder / "case.json"
        if not details.is_file():
            continue
        config = read_json(details)
        signature = (
            {
                key: value
                for key, value in config.get("configuration", {}).items()
                if key != "concurrency"
            },
            config.get("fewshot_indices"),
        )
        if settings is not None and settings != signature:
            raise ValueError("GSM8K shards disagree on evaluation settings")
        settings = signature
        ids = set(config["problem_indices"])
        total = config.get("full_problem_count")
        if total is not None:
            if configured_total not in (None, total):
                raise ValueError("GSM8K shards disagree on the number of problems")
            configured_total = total
        if ids & expected:
            raise ValueError("GSM8K shards overlap")
        expected.update(ids)
        if not (folder / "summary.json").is_file():
            continue
        batch = {item["id"]: item for item in read_jsonl(folder / "responses.jsonl")}
        if batch.keys() != ids:
            raise ValueError("GSM8K results differ from the declared sample")
        rows.update(batch)
    summary = {
        "complete": bool(rows)
        and rows.keys() == expected
        and configured_total in (None, len(rows)),
        "expected": configured_total if configured_total is not None else len(expected),
        "completed": len(rows),
        "strict_correct": sum(
            item.get("strict_correct", False) for item in rows.values()
        ),
        "flexible_correct": sum(
            item.get("flexible_correct", False) for item in rows.values()
        ),
        "truncated": sum(item.get("finish") == "length" for item in rows.values()),
        "errors": sum("error" in item for item in rows.values()),
    }
    return summary, rows, settings


def accuracy_summary(root: Path) -> dict:
    folders = [
        result_directory(path, "accuracy")
        for path in sorted(root.glob("gsm-shard-*")) or [root]
    ]
    runs = collections.defaultdict(list)
    for folder in folders:
        for path in folder.glob("c*/case.json"):
            runs[int(path.parent.name[1:])].append(path.parent)
    if not runs:
        return accuracy_results(folders)[0]

    results = {
        concurrency: accuracy_results(paths) for concurrency, paths in runs.items()
    }
    summary = {
        "runs": {
            concurrency: results[concurrency][0] for concurrency in sorted(results)
        },
        "comparisons": {},
    }
    if 1 not in results or not results[1][0]["complete"]:
        return summary
    _, baseline, settings = results[1]
    for concurrency, (stats, rows, candidate_settings) in sorted(results.items()):
        if concurrency == 1 or not stats["complete"]:
            continue
        if baseline.keys() != rows.keys() or settings != candidate_settings:
            raise ValueError(
                f"GSM8K C1 and C{concurrency} used different inputs or settings"
            )
        changes = {
            "answer_changed": [],
            "correct_to_incorrect": [],
            "new_truncations": [],
            "new_errors": [],
        }
        for index in sorted(baseline):
            before, after = baseline[index], rows[index]
            if before.get("gold") != after.get("gold"):
                raise ValueError(f"GSM8K gold answer differs for problem {index}")
            if "error" not in before and "error" not in after:
                before_answer = before.get("strict_answer") or before.get(
                    "flexible_answer"
                )
                after_answer = after.get("strict_answer") or after.get(
                    "flexible_answer"
                )
                if before_answer != after_answer:
                    changes["answer_changed"].append(index)
                if before.get("strict_correct") and not after.get("strict_correct"):
                    changes["correct_to_incorrect"].append(index)
                if before.get("finish") != "length" and after.get("finish") == "length":
                    changes["new_truncations"].append(index)
            if "error" not in before and "error" in after:
                changes["new_errors"].append(index)
        summary["comparisons"][concurrency] = changes
    return summary


def report(root: Path) -> Path:
    model_path = result_directory(root, "model") / "summary.json"
    models = read_json(model_path) if model_path.is_file() else []
    serving = serving_summary(root)
    mixed = serving_summary(root, "mixed")
    accuracy = accuracy_summary(root)
    write_json(
        root / "summary.json",
        {"model": models, "serving": serving, "mixed": mixed, "accuracy": accuracy},
    )
    lines = [
        "# xTokens Benchmark",
        "",
        "## 单请求模型",
        "",
        "| 输入 token | Prefill 含输出投影 P50 (ms) | Decode 含输出投影 P50 (ms) |",
        "|---:|---:|---:|",
    ]
    for row in models:
        lines.append(
            f"| {row['input_tokens']} | {row['prefill_with_head_p50_ms']:.2f}"
            f" | {row['decode_with_head_p50_ms']:.2f} |"
        )

    def formatted(value, spec=".2f"):
        return format(value, spec) if value is not None else "—"

    lines.extend(
        [
            "",
            "## 分档 Serving",
            "",
            "| 输入 / 输出 / 并发 | 有效轮数 | 吞吐 token/s | TTFT P95 (ms) | TPOT P95 (ms) | 失败率 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in serving:
        length = row["input_tokens"]
        scenario = f"{length} / {row['output_tokens']} / {row['concurrency']}"
        metrics = ("output_tokens_per_sec", "ttft_p95_ms", "tpot_p95_ms")
        failure_rate = row["by_input_tokens"].get(length, {}).get("failure_rate")
        cells = [
            scenario,
            str(row["valid_repeats"]),
            *(formatted(row["metrics"][key]) for key in metrics),
            formatted(failure_rate, ".1%"),
        ]
        lines.append("| " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## 混合请求 Serving",
            "",
            "| 输入 / 输出 / 并发 | 混跑 TTFT P95 (ms) | 同档 TTFT P95 (ms) | 混跑失败率 |",
            "|---|---:|---:|---:|",
        ]
    )
    baseline = {
        (row["input_tokens"], row["output_tokens"], row["concurrency"]): row
        for row in serving
    }
    for row in mixed:
        output, concurrency = row["output_tokens"], row["concurrency"]
        for length, stats in row["by_input_tokens"].items():
            fixed = baseline.get((length, output, concurrency), {})
            fixed_stats = fixed.get("by_input_tokens", {}).get(
                length, fixed.get("metrics", {})
            )
            lines.append(
                f"| {length} / {output} / {concurrency} "
                f"| {formatted(stats['ttft_p95_ms'])} "
                f"| {formatted(fixed_stats.get('ttft_p95_ms'))} "
                f"| {formatted(stats['failure_rate'], '.1%')} |"
            )

    lines.extend(
        [
            "",
            "## GSM8K",
            "",
            "| 并发 | 已完成 / 总数 | 严格匹配 | 宽松匹配 | 截断 | 请求错误 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for concurrency, stats in accuracy.get("runs", {"—": accuracy}).items():
        lines.append(
            f"| {concurrency} | {stats['completed']} / {stats['expected']} "
            f"| {stats['strict_correct']} | {stats['flexible_correct']} "
            f"| {stats['truncated']} | {stats['errors']} |"
        )
    for concurrency, changes in accuracy.get("comparisons", {}).items():
        lines.append(
            f"\nC1 → C{concurrency}，答案变化 {len(changes['answer_changed'])}，"
            f"由对变错 {len(changes['correct_to_incorrect'])}，"
            f"新增截断 {len(changes['new_truncations'])}，"
            f"新增失败 {len(changes['new_errors'])}。"
        )
    destination = root / "summary.md"
    destination.write_text("\n".join(lines) + "\n")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    print(report(parser.parse_args().results))


if __name__ == "__main__":
    main()
