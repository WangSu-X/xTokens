from contextlib import contextmanager

from benchmarks.scripts import run
from benchmarks.scripts.accuracy import answer
from benchmarks.scripts.data import sha256
from benchmarks.scripts.instruments.sitecustomize import install_aiperf
from benchmarks.scripts.report import accuracy_summary, report, serving_summary
from benchmarks.scripts.serving import (
    request_lengths,
    select_requests,
    summarize_requests,
    validate_metrics,
)
from benchmarks.scripts.support import percentile, read_json, write_json, write_jsonl


def test_numeric_answer_uses_final_answer_and_decimal_normalization():
    assert answer("#### 99\n</think>\n#### -1,250.00", strict=True) == "-1.25E+3"
    assert answer("#### .5.", strict=True) == "0.5"
    assert answer("Worked it out. Final answer: 42.", strict=False) == "42"
    assert answer("Worked it out. Final answer: 42.", strict=True) is None


def test_aiperf_adapter_recognizes_both_completion_object_markers():
    from types import SimpleNamespace

    class Endpoint:
        def extract_completions_response_data(self, payload):
            if payload["object"] == "text_completion":
                return payload["choices"][0]["text"]
            return None

    install_aiperf(SimpleNamespace(CompletionsEndpoint=Endpoint))
    payload = {"object": "text_completion.chunk", "choices": [{"text": " hello"}]}
    endpoint = Endpoint()
    assert endpoint.extract_completions_response_data(payload) == " hello"
    assert payload["object"] == "text_completion.chunk"
    assert (
        endpoint.extract_completions_response_data(
            {**payload, "object": "text_completion"}
        )
        == " hello"
    )


def test_validated_serving_result_requires_exact_server_token_lengths():
    row = {
        "is_complete": True,
        "request_count": {"avg": 2},
        "input_sequence_length": {"min": 128, "max": 128},
        "output_sequence_length": {"min": 256, "max": 256},
    }
    assert validate_metrics(row, 2, 128, 256) == []
    row["output_sequence_length"]["min"] = 254
    assert validate_metrics(row, 2, 128, 256) == [
        "output_sequence_length outside fixed length 256"
    ]
    assert validate_metrics(row, 3, 128, 256)[0] == (
        "request count or completion status differs"
    )


def test_serving_report_excludes_invalid_repeats(tmp_path):
    for repeat in (1, 2):
        folder = tmp_path / "serving" / "xtokens" / "i128-o256-c1" / f"r{repeat}"
        write_json(
            folder / "case.json",
            {
                "input_tokens": 128,
                "output_tokens": 256,
                "concurrency": 1,
                "repeat": repeat,
                "requests": 1,
                "sample_ids": ["same"],
            },
        )
        write_json(
            folder / "aiperf/profile_export_aiperf.json",
            {"output_token_throughput": {"avg": 10 * repeat}},
        )
        issues = ["bad token count"] if repeat == 2 else []
        write_json(folder / "validation.json", {"problems": issues})
    (result,) = serving_summary(tmp_path)
    assert result["valid_repeats"] == 1
    assert result["metrics"]["output_tokens_per_sec"] == 10


def test_accuracy_report_aggregates_gsm8k_shards(tmp_path):
    for shard in range(2):
        folder = tmp_path / f"gsm-shard-{shard}" / "accuracy" / "xtokens"
        write_json(
            folder / "case.json", {"problem_indices": [shard], "full_problem_count": 2}
        )
        write_json(folder / "summary.json", {"requests": 1})
        write_jsonl(
            folder / "responses.jsonl",
            [{"id": shard, "strict_correct": shard == 0}],
        )
    result = accuracy_summary(tmp_path)
    assert result["complete"]
    assert result["completed"] == 2
    assert result["strict_correct"] == 1
    assert percentile([1.0, 9.0], 0.5) == 5.0


def test_mixed_serving_keeps_length_order_and_counts_missing_responses(tmp_path):
    lengths = request_lengths([1024, 8192, 16384], requests=3, warmup=1)
    rows = {
        length: [{"id": f"{length}-{i}", "input_tokens": length} for i in range(4)]
        for length in set(lengths)
    }
    selected = select_requests(rows, lengths, 42)
    assert [row["input_tokens"] for row in selected] == [1024, 1024, 8192, 16384]
    assert len({row["id"] for row in selected}) == len(selected)
    assert select_requests(rows, lengths, 42) == selected

    def record(session, length, phase):
        return {
            "metadata": {"conversation_id": session, "phase_kind": phase},
            "metrics": {
                "usage_prompt_tokens": {"value": length},
                "usage_completion_tokens": {"value": 256},
                "time_to_first_token": {"value": 12.0},
            },
        }

    expected = {"short": 1024, "medium": 8192, "long": 16384}
    validation = summarize_requests(
        [record("warmup", 1024, "warmup"), record("short", 1024, "profiling")],
        expected,
        256,
    )
    assert validation["problems"] == ["missing or invalid request records"]
    assert validation["by_input_tokens"][1024]["ttft_p95_ms"] == 12.0
    assert validation["by_input_tokens"][16384]["requests_ok"] == 0
    directory = tmp_path / "mixed" / "mix-o256-c8" / "r1"
    write_json(directory / "validation.json", validation)
    write_json(
        directory / "case.json",
        {
            "input_tokens": list(expected.values()),
            "output_tokens": 256,
            "concurrency": 8,
            "repeat": 1,
            "requests": 3,
            "sample_ids": list(expected),
        },
    )
    result = read_json(report(tmp_path).with_suffix(".json"))["mixed"][0]
    assert result["valid_repeats"] == 0
    assert result["by_input_tokens"]["1024"]["ttft_p95_ms"] == 12.0
    assert result["by_input_tokens"]["16384"]["failure_rate"] == 1.0


def test_accuracy_report_pairs_concurrency_runs(tmp_path):
    correct = {"gold": "1", "strict_answer": "1", "strict_correct": True}
    for concurrency, responses in (
        (1, [{"id": 0, **correct}, {"id": 1, **correct}]),
        (
            16,
            [
                {"id": 0, "gold": "1", "strict_answer": "2", "finish": "length"},
                {"id": 1, "gold": "1", "error": "timeout"},
            ],
        ),
    ):
        folder = tmp_path / "accuracy" / f"c{concurrency}"
        write_json(
            folder / "case.json",
            {
                "configuration": {"concurrency": concurrency, "seed": 42},
                "fewshot_indices": [0],
                "problem_indices": [0, 1],
                "full_problem_count": 2,
            },
        )
        write_jsonl(folder / "responses.jsonl", responses)
        write_json(folder / "summary.json", {"requests": 2})
    summary = accuracy_summary(tmp_path)
    assert summary["runs"][1]["strict_correct"] == 2
    assert summary["runs"][16]["errors"] == 1
    assert summary["comparisons"][16] == {
        "answer_changed": [0],
        "correct_to_incorrect": [0],
        "new_truncations": [0],
        "new_errors": [1],
    }


def test_all_routes_four_cases_without_gpu(monkeypatch, tmp_path):
    model_dir = tmp_path / "checkpoint"
    tokenizer = model_dir / "tokenizer.json"
    write_json(tokenizer, {})
    manifest = {"tokenizer_sha256": sha256(tokenizer), "cases": {}}
    prepared = tmp_path / "data" / "prepared"
    for length in (1024, 8192, 16384):
        write_jsonl(prepared / f"input-{length}.jsonl", [{}] * 16)
        manifest["cases"][str(length)] = {"count": 16}
    for split in ("train", "test"):
        path = tmp_path / "data" / "raw" / "gsm8k" / f"{split}.jsonl"
        write_jsonl(path, [{}])
        manifest[f"gsm8k_{split}_sha256"] = sha256(path)
    write_json(prepared / "manifest.json", manifest)
    destination = tmp_path / "results"
    monkeypatch.setattr(run, "ROOT", tmp_path)
    monkeypatch.setattr(
        run.subprocess,
        "check_output",
        lambda args, **kwargs: "0.14.0" if "--version" in args else "HEAD",
    )
    monkeypatch.setattr(
        run.sys,
        "argv",
        [
            "run",
            "all",
            "--model",
            str(model_dir),
            "--output",
            str(destination),
            "--requests",
            "16",
            "--repeats",
            "1",
            "--warmup",
            "0",
        ],
    )

    @contextmanager
    def no_server(*args, **kwargs):
        yield "unused"

    calls = []

    def record(_url, _runtime, case, path, **kwargs):
        calls.append((str(path.relative_to(destination)), case))

    monkeypatch.setattr(run, "serving_engine", no_server)
    for runner in (run.model, run.serving, run.accuracy):
        monkeypatch.setattr(runner, "evaluate", record)
    run.main()
    assert [path for path, _ in calls] == [
        "model",
        "serving",
        "mixed",
        "accuracy/c1",
        "accuracy/c16",
    ]
    assert calls[1][1]["matrix"]["concurrency"] == [4, 8, 16]
    assert calls[2][1]["input_cycle"] == [1024, 8192, 16384]
    assert calls[2][1]["matrix"]["concurrency"] == [8]
    assert read_json(destination / "mixed" / "run.json")["configuration"] == calls[2][1]
