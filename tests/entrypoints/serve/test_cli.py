from __future__ import annotations

from typing import Any

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
            "--executor-backend",
            "custom",
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
    assert config.executor_config.backend == "custom"
    assert config.parallel_config.distributed_executor_backend == "mp"
    assert config.parallel_config.tensor_parallel_size == 2
    assert config.parallel_config.devices == ("cuda:0", "cuda:1")
