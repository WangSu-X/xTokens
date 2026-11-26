from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from x_tokens.config import ModelConfig, ParallelConfig, ServerConfig, XTokensConfig
from x_tokens.core.protocol import SchedulerOutput
from x_tokens.engine.core_client import MultiProcessClient
from x_tokens.engine.input_processor import InputProcessor
from x_tokens.engine.llm_engine import LLMEngine
from x_tokens.engine.output_processor import OutputProcessor
from x_tokens.engine.protocol import (
    EngineEvent,
    EngineHealth,
    ErrorEvent,
    FinishedEvent,
    FinishReason,
    GenerateRequest,
    TokenEvent,
)
from x_tokens.entrypoints.serve import app as serve_app
from x_tokens.entrypoints.serve import create_app
from x_tokens.executor import ModelExecutor
from x_tokens.executor.protocol import Executor
from x_tokens.worker.model_runner import ModelRunnerOutput


class FakeEngine:
    def __init__(self, *, ready: bool = True, error_after_token: bool = False) -> None:
        self.ready = ready
        self.error_after_token = error_after_token
        self.aborted: list[str] = []
        self.closed = False
        self.calls: list[GenerateRequest] = []

    def generate(self, request: GenerateRequest) -> AsyncIterator[EngineEvent]:
        self.calls.append(request)

        async def stream() -> AsyncIterator[EngineEvent]:
            yield TokenEvent(request.request_id, 1, "Hello")
            if self.error_after_token:
                yield ErrorEvent(request.request_id, "mock failure")
                return
            yield TokenEvent(request.request_id, 2, " world")
            yield FinishedEvent(request.request_id, FinishReason.STOP, 3, 2)

        return stream()

    async def abort(self, request_id: str) -> None:
        self.aborted.append(request_id)

    async def health(self) -> EngineHealth:
        return EngineHealth(self.ready, "fake")

    async def close(self) -> None:
        self.closed = True


class FakeExecutor(Executor):
    eos_token_ids = frozenset((9,))
    num_gpu_blocks = 256

    def execute_step(self, batch: SchedulerOutput) -> ModelRunnerOutput:
        return ModelRunnerOutput(
            {
                request.request_id: 9
                for request in batch.running_requests
                if request.sampling_params is not None
            }
        )


def _make_fake_executor(*args: object, **kwargs: object) -> FakeExecutor:
    return FakeExecutor()


class FakeInputProcessor(InputProcessor):
    pad_token_id = 0
    eos_token_ids = frozenset((9,))

    def process(self, request: GenerateRequest) -> GenerateRequest:
        return GenerateRequest(
            request.request_id, request.model, (1,), request.sampling
        )

    @property
    def tokenizer(self) -> object:
        return object()


class FakeOutputProcessor(OutputProcessor):
    def process_token(self, request_id: str, token_id: int) -> str:
        del request_id, token_id
        return ""

    def finish(self, request_id: str) -> None:
        del request_id


def client_for(
    engine: FakeEngine, monkeypatch: pytest.MonkeyPatch, **config: object
) -> TestClient:
    monkeypatch.setattr(serve_app, "LLMEngine", lambda _: engine)
    app = create_app(
        XTokensConfig(
            model_config=ModelConfig(served_model_name="test-model"),
            server_config=ServerConfig(**config),
        )
    )
    return TestClient(app)


def test_completion_uses_client_request_id_and_openai_error_format(monkeypatch) -> None:
    engine = FakeEngine()
    with client_for(engine, monkeypatch) as client:
        response = client.post(
            "/v1/completions",
            headers={"X-Request-ID": "benchmark-1"},
            json={
                "model": "test-model",
                "prompt": "hello",
                "ignore_eos": True,
            },
        )
        assert response.status_code == 200
        assert response.json()["id"] == "benchmark-1"
        assert response.headers["x-request-id"] == "benchmark-1"
        assert response.json()["usage"] == {
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "total_tokens": 5,
        }
        assert engine.calls[0].prompt == "hello"
        assert engine.calls[0].sampling.ignore_eos is True

        invalid = client.post(
            "/v1/completions",
            json={"model": "test-model", "prompt": "hello", "temperature": -1},
        )
        assert invalid.status_code == 422
        assert invalid.json()["error"]["type"] == "invalid_request_error"


def test_ready_gate_and_api_key(monkeypatch) -> None:
    engine = FakeEngine(ready=False)
    with client_for(engine, monkeypatch, api_key="secret") as client:
        assert client.get("/ready").status_code == 503
        unauthorized = client.get("/v1/models")
        assert unauthorized.status_code == 401
        wrong_key = client.get(
            "/v1/models", headers={"Authorization": "Bearer wrong-key"}
        )
        assert wrong_key.status_code == 401
        response = client.post(
            "/v1/completions",
            headers={"Authorization": "Bearer secret"},
            json={"model": "test-model", "prompt": "hello"},
        )
        assert response.status_code == 503
        assert engine.calls == []
    assert engine.closed


def test_chat_stream_includes_usage_and_late_error(monkeypatch) -> None:
    engine = FakeEngine(error_after_token=True)
    with client_for(engine, monkeypatch) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"X-Request-ID": "chat-stream-1"},
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hello"}],
                "ignore_eos": True,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert response.headers["x-request-id"] == "chat-stream-1"
        assert '"id":"chat-stream-1"' in response.text
        assert '"content":"Hello"' in response.text
        assert '"message":"mock failure"' in response.text
        assert response.text.endswith("data: [DONE]\n\n")
        assert engine.calls[0].request_id == "chat-stream-1"
        assert engine.calls[0].sampling.ignore_eos is True


def test_default_client_uses_multiproc_llm_engine(
    monkeypatch, install_test_executor
) -> None:
    install_test_executor(_make_fake_executor)
    monkeypatch.setattr(
        "x_tokens.engine.llm_engine.TokenizerInputProcessor.from_config",
        lambda *args, **kwargs: FakeInputProcessor(),
    )
    monkeypatch.setattr(
        "x_tokens.engine.llm_engine.TokenizerOutputProcessor",
        lambda _: FakeOutputProcessor(),
    )
    app = create_app(
        XTokensConfig(model_config=ModelConfig(served_model_name="test-model"))
    )

    with TestClient(app) as client:
        assert isinstance(app.state.engine.core_client, MultiProcessClient)
        response = client.post(
            "/v1/completions",
            json={"model": "test-model", "prompt": "hello"},
        )

    assert response.status_code == 200
    assert response.json()["choices"][0]["finish_reason"] == "stop"


def test_multiproc_server_ignores_eos_until_max_tokens(
    monkeypatch, install_test_executor
) -> None:
    install_test_executor(_make_fake_executor)
    monkeypatch.setattr(
        "x_tokens.engine.llm_engine.TokenizerInputProcessor.from_config",
        lambda *args, **kwargs: FakeInputProcessor(),
    )
    monkeypatch.setattr(
        "x_tokens.engine.llm_engine.TokenizerOutputProcessor",
        lambda _: FakeOutputProcessor(),
    )
    app = create_app(
        XTokensConfig(model_config=ModelConfig(served_model_name="test-model"))
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/completions",
            json={
                "model": "test-model",
                "prompt": "hello",
                "max_tokens": 2,
                "ignore_eos": True,
            },
        )

    assert response.status_code == 200
    assert response.json()["choices"][0]["finish_reason"] == "length"
    assert response.json()["usage"]["completion_tokens"] == 2


def test_engine_core_constructs_the_configured_executor_class(monkeypatch) -> None:
    monkeypatch.setattr(
        "x_tokens.engine.llm_engine.TokenizerInputProcessor.from_config",
        lambda *args, **kwargs: FakeInputProcessor(),
    )
    executor_class = Mock(return_value=FakeExecutor())
    get_class = Mock(return_value=executor_class)
    monkeypatch.setattr(ModelExecutor, "get_class", get_class)
    config = XTokensConfig(
        model_config=ModelConfig(served_model_name="test-model"),
        parallel_config=ParallelConfig(devices=("cpu",), collective_backend="gloo"),
        server_config=ServerConfig(core_client_mode="in_proc"),
    )

    engine = LLMEngine(config)

    get_class.assert_called_once_with(config.parallel_config)
    executor_class.assert_called_once_with(
        config, pad_token_id=0, eos_token_ids=frozenset({9})
    )
    asyncio.run(engine.close())


@pytest.mark.parametrize("num_gpu_blocks", [0, 1])
def test_engine_core_cleans_up_invalid_kv_capacity(monkeypatch, num_gpu_blocks) -> None:
    monkeypatch.setattr(
        "x_tokens.engine.llm_engine.TokenizerInputProcessor.from_config",
        lambda *args, **kwargs: FakeInputProcessor(),
    )
    executor = FakeExecutor()
    executor.num_gpu_blocks = num_gpu_blocks
    executor.shutdown = Mock()
    monkeypatch.setattr(
        ModelExecutor, "get_class", Mock(return_value=Mock(return_value=executor))
    )

    config = XTokensConfig(server_config=ServerConfig(core_client_mode="in_proc"))
    with pytest.raises((RuntimeError, ValueError), match="KV cache"):
        LLMEngine(config)

    executor.shutdown.assert_called_once_with()
