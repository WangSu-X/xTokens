from __future__ import annotations

import logging
from queue import Queue
from unittest.mock import Mock

import pytest

from x_tokens.config import KVCacheConfig, ModelConfig, SchedulerConfig, XTokensConfig
from x_tokens.core import EngineCore, EngineCoreConfig
from x_tokens.core import engine_core as engine_core_module
from x_tokens.core.protocol import SchedulerOutput
from x_tokens.engine.protocol import (
    CoreErrorEvent,
    CoreFinishedEvent,
    CoreTokenEvent,
    EngineCoreRequest,
    EngineCoreRequestType,
    FinishReason,
    SamplingParams,
)
from x_tokens.executor import ModelExecutor
from x_tokens.executor.protocol import Executor, ModelForwardOutput


class FakeExecutor(Executor):
    eos_token_ids = frozenset((9,))
    num_gpu_blocks = 256

    def __init__(self, token_ids: list[int]) -> None:
        self.token_ids = iter(token_ids)
        self.batches: list[list[str]] = []

    def execute_model(self, batch: SchedulerOutput) -> ModelForwardOutput:
        self.batches.append([request.request_id for request in batch.running_requests])
        return ModelForwardOutput()

    def sample_tokens(
        self, output: ModelForwardOutput, batch: SchedulerOutput
    ) -> tuple[int, ...]:
        del output
        return tuple(next(self.token_ids) for _ in range(batch.num_samples))


def create_core(
    executor: FakeExecutor,
    *,
    max_num_seqs: int = 4,
    max_model_len: int = 4096,
    cache_config: KVCacheConfig | None = None,
) -> EngineCore:
    config = EngineCoreConfig(
        XTokensConfig(
            model_config=ModelConfig(
                served_model_name="test-model", max_model_len=max_model_len
            ),
            scheduler_config=SchedulerConfig(max_num_seqs=max_num_seqs),
            kv_cache_config=cache_config
            if cache_config is not None
            else KVCacheConfig(),
        ),
        pad_token_id=0,
        eos_token_ids=executor.eos_token_ids,
    )
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            ModelExecutor, "get_class", Mock(return_value=Mock(return_value=executor))
        )
        return EngineCore(config)


def request(
    request_id: str, *, max_tokens: int = 2, ignore_eos: bool = False
) -> EngineCoreRequest:
    return EngineCoreRequest(
        request_id,
        SamplingParams(max_tokens=max_tokens, ignore_eos=ignore_eos),
        (1,),
    )


def test_core_steps_kv_batches_and_returns_events_directly() -> None:
    executor = FakeExecutor([2, 3, 9, 4])
    core = create_core(executor, max_num_seqs=2)
    for request_id in ("one", "two"):
        core.add_request(request(request_id))

    first, executed = core.step_fn()
    core.post_step(model_executed=executed)
    second, executed = core.step_fn()
    core.post_step(model_executed=executed)

    assert executor.batches == [["one", "two"], ["one", "two"]]
    assert first.get(0) == (
        CoreTokenEvent("one", 2),
        CoreTokenEvent("two", 3),
    )
    assert second.get(0) == (
        CoreFinishedEvent("one", FinishReason.STOP, 1, 2),
        CoreTokenEvent("two", 4),
        CoreFinishedEvent("two", FinishReason.LENGTH, 1, 2),
    )


def test_core_process_input_drains_queue_before_stepping() -> None:
    executor = FakeExecutor([2, 3])
    core = create_core(executor)
    core._input_queue = Queue()
    for request_id in ("one", "two"):
        core._input_queue.put((EngineCoreRequestType.ADD, request(request_id)))

    core._process_input()
    _, executed = core.step_fn()

    assert core._input_queue.empty()
    assert executed
    assert executor.batches == [["one", "two"]]


@pytest.mark.parametrize("queue_request", [False, True], ids=["idle", "queued-request"])
def test_core_busy_loop_skips_step_after_close(
    monkeypatch, queue_request: bool
) -> None:
    executor = FakeExecutor([2])
    shutdown = Mock()
    monkeypatch.setattr(executor, "shutdown", shutdown, raising=False)
    core = create_core(executor)
    core._input_queue = Queue()
    if queue_request:
        core._input_queue.put((EngineCoreRequestType.ADD, request("queued")))
    core._input_queue.put((EngineCoreRequestType.CLOSE, None))
    core._input_queue.put((EngineCoreRequestType.ADD, request("after-close")))
    step = Mock(wraps=core.step_fn)
    monkeypatch.setattr(core, "step_fn", step)

    core.run_busy_loop()

    assert core.closed
    assert not core.has_work
    assert core._input_queue.get_nowait()[1].request_id == "after-close"
    step.assert_not_called()
    core.close()
    shutdown.assert_called_once_with()


def test_core_logs_model_step_timing(monkeypatch) -> None:
    debug = Mock()
    monkeypatch.setattr(
        engine_core_module.logger,
        "isEnabledFor",
        lambda level: level == logging.DEBUG,
    )
    monkeypatch.setattr(engine_core_module.logger, "debug", debug)
    core = create_core(FakeExecutor([2]))
    core.add_request(request("timed"))

    core.step_fn()

    debug.assert_called_once()
    assert debug.call_args.args[0].startswith("model step finished:")


def test_core_emits_ignored_eos_until_request_length() -> None:
    core = create_core(FakeExecutor([9, 9]))
    core.add_request(request("eval", ignore_eos=True))

    first, _ = core.step_fn()
    second, _ = core.step_fn()

    assert first.get(0) == (CoreTokenEvent("eval", 9),)
    assert second.get(0) == (
        CoreTokenEvent("eval", 9),
        CoreFinishedEvent("eval", FinishReason.LENGTH, 1, 2),
    )


def test_core_isolates_request_validation_errors_without_a_queue() -> None:
    core = create_core(FakeExecutor([9]))
    for item in (request("bad", max_tokens=0), request("good")):
        core.add_request(item)

    outputs, _ = core.step_fn()

    assert isinstance(outputs.get(0)[0], CoreErrorEvent)
    assert outputs.get(0)[0].request_id == "bad"
    assert outputs.get(0)[1] == CoreFinishedEvent("good", FinishReason.STOP, 1, 1)


def test_core_creates_scheduler_with_configured_limits() -> None:
    executor = FakeExecutor([2, 3])
    core = create_core(executor, max_num_seqs=1, max_model_len=8)
    core.add_request(request("one"))
    core.add_request(request("two"))

    outputs, executed = core.step_fn()

    assert executed
    assert executor.batches == [["one"]]
    assert outputs.get(0) == (CoreTokenEvent("one", 2),)


def test_core_passes_resolved_executor_cache_capacity_to_scheduler() -> None:
    executor = FakeExecutor([])
    core = create_core(
        executor,
        max_model_len=8,
        cache_config=KVCacheConfig(block_size=4, memory_bytes=4096),
    )

    cache_manager = core.scheduler.kv_cache_manager
    assert cache_manager.block_pool.free_queue.size == executor.num_gpu_blocks
    assert cache_manager.block_size == 4


def test_core_model_failure_releases_kv_blocks(monkeypatch) -> None:
    executor = FakeExecutor([])
    monkeypatch.setattr(
        executor, "execute_model", Mock(side_effect=RuntimeError("failed"))
    )
    core = create_core(executor)
    core.add_request(request("failed"))

    outputs, executed = core.step_fn()

    assert executed
    assert outputs.get(0) == (CoreErrorEvent("failed", "failed"),)
    assert core.scheduler.kv_cache_manager.stats().active_blocks == 0
    assert not core.scheduler.has_work
