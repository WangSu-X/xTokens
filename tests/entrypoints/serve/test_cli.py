from __future__ import annotations

from typing import Any

import pytest

from x_tokens.entrypoints.serve import cli


def test_cli_passes_unified_logging_config_to_uvicorn(monkeypatch) -> None:
    invocation: dict[str, Any] = {}

    def fake_run(app: object, **kwargs: Any) -> None:
        invocation["app"] = app
        invocation.update(kwargs)

    monkeypatch.setattr(cli.uvicorn, "run", fake_run)

    cli.main(["--model", "test-model", "--no-access-log"])

    assert invocation["access_log"] is False
    assert invocation["log_config"]["loggers"]["uvicorn"]["handlers"] == ["xtokens"]
    assert invocation["log_config"]["loggers"]["uvicorn.access"]["handlers"] == [
        "xtokens_access"
    ]


def test_cli_builds_explicit_parallel_config(monkeypatch) -> None:
    invocation: dict[str, Any] = {}

    def fake_create_app(config: object) -> object:
        invocation["config"] = config
        return object()

    monkeypatch.setattr(cli, "create_app", fake_create_app)
    monkeypatch.setattr(cli.uvicorn, "run", lambda *args, **kwargs: None)

    cli.main(
        [
            "--model",
            "/models/qwen3",
            "--served-model-name",
            "test-model",
            "--dtype",
            "bfloat16",
            "--local-files-only",
            "--max-num-seqs",
            "8",
            "--max-model-len",
            "8192",
            "--distributed-executor-backend",
            "mp",
            "--tensor-parallel-size",
            "2",
            "--parallel-device",
            "cuda:0",
            "--parallel-device",
            "cuda:1",
        ]
    )

    config = invocation["config"]
    assert config.model_config.model == "/models/qwen3"
    assert config.model_config.served_model_name == "test-model"
    assert config.model_config.max_model_len == 8192
    assert config.executor_config.dtype == "bfloat16"
    assert config.executor_config.local_files_only
    assert config.scheduler_config.max_num_seqs == 8
    assert config.parallel_config.distributed_executor_backend == "mp"
    assert config.parallel_config.tensor_parallel_size == 2
    assert config.parallel_config.devices == ("cuda:0", "cuda:1")


def test_cli_defaults_served_model_name_to_model(monkeypatch) -> None:
    invocation: dict[str, Any] = {}
    monkeypatch.setattr(
        cli, "create_app", lambda config: invocation.update(config=config)
    )
    monkeypatch.setattr(cli.uvicorn, "run", lambda *args, **kwargs: None)

    cli.main(["--model", "org/model"])

    assert invocation["config"].model_config.served_model_name == "org/model"


@pytest.mark.parametrize(
    ("engine_args", "expected_mode"),
    [
        ([], "multi_proc"),
        (["--engine", "multiproc"], "multi_proc"),
        (["--engine", "inproc"], "in_proc"),
    ],
)
def test_cli_selects_engine_core_transport(
    monkeypatch, engine_args, expected_mode
) -> None:
    invocation: dict[str, Any] = {}
    monkeypatch.setattr(
        cli, "create_app", lambda config: invocation.update(config=config)
    )
    monkeypatch.setattr(cli.uvicorn, "run", lambda *args, **kwargs: None)

    cli.main(["--model", "test-model", *engine_args])

    assert invocation["config"].server_config.core_client_mode == expected_mode


@pytest.mark.parametrize(
    "option",
    [
        "--hf-model",
        "--hf-device",
        "--hf-dtype",
        "--hf-local-files-only",
        "--hf-max-num-seqs",
        "--executor-backend",
    ],
)
def test_cli_rejects_removed_executor_options(option: str) -> None:
    with pytest.raises(SystemExit) as error:
        cli.main(["--model", "test-model", option])

    assert error.value.code == 2


def test_cli_configures_native_chunked_prefill(monkeypatch) -> None:
    invocation: dict[str, Any] = {}
    monkeypatch.setattr(
        cli, "create_app", lambda config: invocation.update(config=config)
    )
    monkeypatch.setattr(cli.uvicorn, "run", lambda *args, **kwargs: None)

    cli.main(
        [
            "--model",
            "test-model",
            "--kv-cache-memory-bytes",
            "1024",
            "--enable-chunked-prefill",
            "--max-prefill-batch-tokens",
            "4096",
            "--prefill-chunk-alignment",
            "768",
        ]
    )

    config = invocation["config"]
    assert config.scheduler_config.max_prefill_batch_tokens == 4096
    assert config.scheduler_config.enable_chunked_prefill
    assert config.scheduler_config.prefill_chunk_alignment == 768
    assert config.kv_cache_config.block_size == 16
    assert config.kv_cache_config.memory_bytes == 1024


def test_cli_rejects_two_kv_cache_budgets() -> None:
    with pytest.raises(SystemExit) as error:
        cli.main(
            [
                "--model",
                "test-model",
                "--num-gpu-blocks",
                "3",
                "--kv-cache-memory-bytes",
                "1024",
            ]
        )
    assert error.value.code == 2
