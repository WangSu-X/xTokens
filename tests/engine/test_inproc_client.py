from __future__ import annotations

import asyncio
import time
from contextlib import aclosing
from functools import partial
from pathlib import Path

import pytest

from x_tokens.config import (
    KVCacheConfig,
    ModelConfig,
    SchedulerConfig,
    ServerConfig,
    XTokensConfig,
)
from x_tokens.core import EngineCore
from x_tokens.core.protocol import SchedulerOutput
from x_tokens.core.scheduler import Scheduler
from x_tokens.engine.core_client import InProcessClient, MultiProcessClient
from x_tokens.engine.input_processor import InputProcessor
from x_tokens.engine.llm_engine import LLMEngine
from x_tokens.engine.output_processor import OutputProcessor
from x_tokens.engine.protocol import (
    CoreErrorEvent,
    CoreFinishedEvent,
    CoreTokenEvent,
    FinishedEvent,
    FinishReason,
    GenerateRequest,
    SamplingParams,
    TokenEvent,
)
from x_tokens.executor.protocol import Executor, ModelForwardOutput

pytestmark = pytest.mark.usefixtures("fake_core_executor")


class FakeExecutor(Executor):
    eos_token_ids = frozenset((9,))
    num_gpu_blocks = 256

    def execute_model(self, batch: SchedulerOutput) -> ModelForwardOutput:
        return ModelForwardOutput()

    def sample_tokens(
        self, output: ModelForwardOutput, batch: SchedulerOutput
    ) -> tuple[int, ...]:
        del output
        return tuple(9 for _ in range(batch.num_samples))


class FakeInputProcessor(InputProcessor):
    pad_token_id = 0
    eos_token_ids = frozenset((9,))

    def process(self, request: GenerateRequest) -> GenerateRequest:
        return GenerateRequest(
            request.request_id, request.model, (1,), request.sampling
        )


class FakeOutputProcessor(OutputProcessor):
    def process_token(self, request_id: str, token_id: int) -> str:
        del request_id, token_id
        return ""

    def finish(self, request_id: str) -> None:
        del request_id


def test_inproc_client_owns_core_and_steps_in_the_calling_thread() -> None:
    client = InProcessClient(
        XTokensConfig(model_config=ModelConfig(served_model_name="test-model")),
        0,
        frozenset({9}),
    )
    assert isinstance(client, InProcessClient)
    assert isinstance(client.engine_core, EngineCore)
    client.add_request(
        GenerateRequest("inproc-request", "test-model", (1,), SamplingParams(2))
    )

    outputs = asyncio.run(client.get_output())

    assert isinstance(outputs[-1], CoreFinishedEvent)
    assert outputs[-1].request_id == "inproc-request"
    client.close()
    assert not client.health().ready


def test_inproc_client_uses_kv_cache_scheduler_from_core_config() -> None:
    config = XTokensConfig(
        model_config=ModelConfig(served_model_name="test-model", max_model_len=8),
        scheduler_config=SchedulerConfig(max_num_seqs=1),
        kv_cache_config=KVCacheConfig(block_size=2, num_gpu_blocks=8),
    )
    client = InProcessClient(
        config,
        0,
        frozenset({9}),
    )
    try:
        assert isinstance(client.engine_core.scheduler, Scheduler)
        client.add_request(
            GenerateRequest("first", "test-model", (1,), SamplingParams(2))
        )
        client.add_request(
            GenerateRequest("second", "test-model", (2,), SamplingParams(2))
        )
        assert asyncio.run(client.get_output())[-1].request_id == "first"
        assert asyncio.run(client.get_output())[-1].request_id == "second"
        assert client.engine_core.scheduler.kv_cache_manager.stats().active_blocks == 0
    finally:
        client.close()


@pytest.mark.parametrize("client_cls", [InProcessClient, MultiProcessClient])
def test_core_clients_validate_model_and_tokenized_input(client_cls) -> None:
    config = XTokensConfig(model_config=ModelConfig(served_model_name="test-model"))
    client = client_cls(config, 0, frozenset({9}))

    async def receive_outputs() -> None:
        sampling = SamplingParams(2)
        client.add_request(GenerateRequest("bad-model", "other-model", (1,), sampling))
        assert await asyncio.wait_for(client.get_output(), timeout=5) == (
            CoreErrorEvent(
                "bad-model", "requested model is not loaded by this EngineCore"
            ),
        )
        client.add_request(
            GenerateRequest("bad-prompt", "test-model", "hello", sampling)
        )
        assert await asyncio.wait_for(client.get_output(), timeout=5) == (
            CoreErrorEvent(
                "bad-prompt", "EngineCore requires preprocessed prompt token IDs"
            ),
        )
        client.add_request(GenerateRequest("valid", "test-model", (1,), sampling))
        assert await asyncio.wait_for(client.get_output(), timeout=5) == (
            CoreFinishedEvent("valid", FinishReason.STOP, 1, 1),
        )

    try:
        asyncio.run(receive_outputs())
    finally:
        client.close()


def test_llm_engine_normalizes_direct_inproc_outputs() -> None:
    async def scenario() -> None:
        engine = LLMEngine(
            XTokensConfig(
                model_config=ModelConfig(served_model_name="test-model"),
                server_config=ServerConfig(core_client_mode="in_proc"),
            ),
            input_processor=FakeInputProcessor(),
            output_processor=FakeOutputProcessor(),
        )
        assert isinstance(engine.core_client, InProcessClient)
        request = GenerateRequest("request", "test-model", "hello", SamplingParams(2))
        events = [event async for event in engine.generate(request)]
        assert not any(isinstance(event, TokenEvent) for event in events)
        assert isinstance(events[-1], FinishedEvent)
        await engine.close()

    asyncio.run(scenario())


def test_llm_engine_rejects_duplicate_id_without_disrupting_original() -> None:
    async def scenario() -> None:
        engine = LLMEngine(
            XTokensConfig(model_config=ModelConfig(served_model_name="test-model")),
            input_processor=FakeInputProcessor(),
            output_processor=FakeOutputProcessor(),
        )
        request = GenerateRequest(
            "same-id", "test-model", "hello", SamplingParams(2, ignore_eos=True)
        )
        try:
            async with aclosing(engine.generate(request)) as original:
                assert await anext(original) == TokenEvent("same-id", 9, "")
                async with aclosing(engine.generate(request)) as duplicate:
                    with pytest.raises(
                        ValueError, match="request_id is already active"
                    ):
                        await anext(duplicate)
                assert [event async for event in original] == [
                    TokenEvent("same-id", 9, ""),
                    FinishedEvent("same-id", FinishReason.LENGTH, 1, 2),
                ]
        finally:
            await engine.close()

    asyncio.run(asyncio.wait_for(scenario(), timeout=5))


def test_llm_engine_routes_concurrent_zmq_outputs() -> None:
    async def scenario() -> None:
        config = XTokensConfig(
            model_config=ModelConfig(served_model_name="test-model"),
        )
        engine = LLMEngine(
            config,
            input_processor=FakeInputProcessor(),
            output_processor=FakeOutputProcessor(),
        )
        try:
            assert isinstance(engine.core_client, MultiProcessClient)
            assert engine.core_client.health().ready
            requests = [
                GenerateRequest(
                    f"zmq-request-{index}",
                    "test-model",
                    "hello",
                    SamplingParams(1),
                )
                for index in range(3)
            ]

            async def collect(request: GenerateRequest):
                return [event async for event in engine.generate(request)]

            results = await asyncio.gather(*(collect(request) for request in requests))
            for request, events in zip(requests, results, strict=True):
                assert isinstance(events[-1], FinishedEvent)
                assert events[-1].request_id == request.request_id
        finally:
            await engine.close()

    asyncio.run(scenario())


def _make_fake_executor(config, pad_token_id, eos_token_ids):
    del config, pad_token_id, eos_token_ids
    return FakeExecutor()


@pytest.fixture
def fake_core_executor(install_test_executor) -> None:
    install_test_executor(_make_fake_executor)


def test_multiproc_core_finishes_before_any_get_output() -> None:
    config = XTokensConfig(
        model_config=ModelConfig(served_model_name="test-model"),
        server_config=ServerConfig(core_client_mode="multi_proc"),
    )
    client = MultiProcessClient(config, 0, frozenset({9}))
    finished = CoreFinishedEvent("autonomous", FinishReason.LENGTH, 1, 3)
    try:
        client.add_request(
            GenerateRequest(
                "autonomous", "test-model", (1,), SamplingParams(3, ignore_eos=True)
            )
        )
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not client.output_socket.getsockopt(
            client.zmq.EVENTS
        ):
            time.sleep(0.01)
        assert client.output_socket.getsockopt(client.zmq.EVENTS) & client.zmq.POLLIN

        async def collect():
            events = []
            while not events or events[-1] != finished:
                events.extend(await client.get_output())
            return events

        assert asyncio.run(asyncio.wait_for(collect(), timeout=5)) == [
            CoreTokenEvent("autonomous", 9)
        ] * 3 + [finished]
    finally:
        client.close()

    assert client.process is not None
    assert client.process.exitcode == 0
    assert client.output_socket.closed
    assert not Path(client.temp_dir.name).exists()


def test_multiproc_reports_unknown_request_type_and_keeps_serving() -> None:
    config = XTokensConfig(model_config=ModelConfig(served_model_name="test-model"))
    client = MultiProcessClient(config, 0, frozenset({9}))

    async def receive_outputs() -> None:
        client.input_socket.send_multipart(
            [client.client_identity, b'{"request_type":"unknown"}']
        )
        with pytest.raises(RuntimeError, match="unknown request type: unknown"):
            await asyncio.wait_for(client.get_output(), timeout=5)

        client.add_request(
            GenerateRequest("valid", "test-model", (1,), SamplingParams(2))
        )
        assert await asyncio.wait_for(client.get_output(), timeout=5) == (
            CoreFinishedEvent("valid", FinishReason.STOP, 1, 1),
        )

    try:
        asyncio.run(receive_outputs())
    finally:
        client.close()


def test_multiproc_reports_process_exit_to_async_receiver() -> None:
    client = MultiProcessClient(XTokensConfig(), 0, frozenset({9}))
    try:
        assert client.process is not None
        client.process.terminate()
        client.process.join(timeout=5)
        with pytest.raises(
            RuntimeError, match="EngineCore process exited unexpectedly"
        ):
            asyncio.run(asyncio.wait_for(client.get_output(), timeout=5))
    finally:
        client.close()


def test_cancelled_multiproc_receive_keeps_later_token_events() -> None:
    config = XTokensConfig(model_config=ModelConfig(served_model_name="test-model"))
    client = MultiProcessClient(config, 0, frozenset({9}))

    async def collect():
        pending = asyncio.create_task(client.get_output())
        await asyncio.sleep(0)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        client.add_request(
            GenerateRequest(
                "cancel-receive", "test-model", (1,), SamplingParams(1, ignore_eos=True)
            )
        )
        return await client.get_output()

    try:
        result = asyncio.run(asyncio.wait_for(collect(), timeout=5))
        assert result == (
            CoreTokenEvent("cancel-receive", 9),
            CoreFinishedEvent("cancel-receive", FinishReason.LENGTH, 1, 1),
        )
    finally:
        client.close()


class InvalidKVCacheExecutor(FakeExecutor):
    num_gpu_blocks = 0

    def __init__(self, shutdown_marker: Path) -> None:
        self.shutdown_marker = shutdown_marker

    def shutdown(self) -> None:
        self.shutdown_marker.touch()


def _make_invalid_executor(config, pad_token_id, eos_token_ids, *, shutdown_marker):
    del config, pad_token_id, eos_token_ids
    return InvalidKVCacheExecutor(shutdown_marker)


def test_multiproc_core_reports_startup_failure_and_shuts_down_executor(
    tmp_path: Path, install_test_executor
) -> None:
    shutdown_marker = tmp_path / "executor-closed"
    install_test_executor(
        partial(_make_invalid_executor, shutdown_marker=shutdown_marker)
    )
    config = XTokensConfig(server_config=ServerConfig(core_client_mode="multi_proc"))
    with pytest.raises(RuntimeError, match="executor did not initialize KV cache"):
        MultiProcessClient(config, 0, frozenset({9}))
    assert shutdown_marker.exists()


def test_llm_engine_ignores_eos_until_max_tokens() -> None:
    async def scenario() -> None:
        engine = LLMEngine(
            XTokensConfig(model_config=ModelConfig(served_model_name="test-model")),
            input_processor=FakeInputProcessor(),
            output_processor=FakeOutputProcessor(),
        )
        request = GenerateRequest(
            "eval",
            "test-model",
            "hello",
            SamplingParams(max_tokens=2, ignore_eos=True),
        )

        events = [event async for event in engine.generate(request)]

        assert events[:2] == [TokenEvent("eval", 9, ""), TokenEvent("eval", 9, "")]
        assert events[-1] == FinishedEvent("eval", FinishReason.LENGTH, 1, 2)
        await engine.close()

    asyncio.run(scenario())
