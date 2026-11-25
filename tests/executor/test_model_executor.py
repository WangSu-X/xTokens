from __future__ import annotations

import pytest
import torch
from torch import nn

from x_tokens.config import KVCacheConfig, ParallelConfig
from x_tokens.core.protocol import SchedulerOutput, SchedulerRequest
from x_tokens.engine.protocol import EngineCoreRequest, SamplingParams
from x_tokens.executor import (
    ModelExecutor,
    MultiprocModelExecutor,
    UniProcModelExecutor,
)
from x_tokens.kv_cache import KVCacheSpec
from x_tokens.modeling import ModelInput, ModelOutput
from x_tokens.worker.model_runner import GPUModelRunnerFactory


class TinyNextTokenModel(nn.Module):
    def __init__(self, vocab_size: int = 16) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.anchor = nn.Parameter(torch.zeros(()))
        self.kv_cache = None

    def forward(self, inputs: ModelInput) -> ModelOutput:
        assert inputs.input_ids is not None
        next_tokens = (inputs.input_ids[:, -1] + 1) % self.vocab_size
        logits = torch.full(
            (inputs.input_ids.shape[0], self.vocab_size),
            -1000.0,
            device=inputs.input_ids.device,
        )
        logits.scatter_(1, next_tokens.unsqueeze(1), 1000.0)
        return ModelOutput(logits)

    def get_kv_cache_spec(self, block_size: int) -> KVCacheSpec:
        return KVCacheSpec(1, block_size, 1, 1, self.anchor.element_size())

    def set_kv_cache(self, kv_cache: object | None) -> None:
        self.kv_cache = kv_cache


def tiny_model_factory(context: object) -> TinyNextTokenModel:
    del context
    return TinyNextTokenModel()


def batch(*contexts: tuple[int, ...]) -> SchedulerOutput:
    requests = []
    num_scheduled_tokens = {}
    for index, context in enumerate(contexts):
        request = EngineCoreRequest(
            f"request-{index}",
            SamplingParams(max_tokens=2, temperature=0),
            context,
        )
        requests.append(SchedulerRequest(request, (index,)))
        num_scheduled_tokens[request.request_id] = len(context)
    return SchedulerOutput(tuple(requests), num_scheduled_tokens)


def runner_factory() -> GPUModelRunnerFactory:
    return GPUModelRunnerFactory(tiny_model_factory, pad_token_id=0)


def cache_config() -> KVCacheConfig:
    return KVCacheConfig(block_size=4, num_gpu_blocks=8)


def test_uniproc_executor_runs_pending_step_and_samples_on_device() -> None:
    executor = UniProcModelExecutor(
        ParallelConfig(devices=("cpu",), collective_backend="gloo"),
        runner_factory(),
        eos_token_ids=frozenset((15,)),
        kv_cache_config=cache_config(),
    )
    scheduled = batch((1, 2), (4,))
    try:
        output = executor.execute_model(scheduled)

        assert output.runner_step_id is not None
        assert executor.sample_tokens(output, scheduled) == (3, 5)
        assert executor.eos_token_ids == frozenset((15,))
        executor.check_health()
    finally:
        executor.shutdown()


def test_executor_class_selection_is_independent_of_model_factory() -> None:
    assert (
        ModelExecutor.get_class(
            ParallelConfig(devices=("cpu",), collective_backend="gloo")
        )
        is UniProcModelExecutor
    )
    assert (
        ModelExecutor.get_class(
            ParallelConfig(
                distributed_executor_backend="mp",
                devices=("cpu:0", "cpu:1"),
                tensor_parallel_size=2,
                collective_backend="gloo",
            )
        )
        is MultiprocModelExecutor
    )


def test_executor_rejects_execute_execute_without_sampling() -> None:
    executor = UniProcModelExecutor(
        ParallelConfig(devices=("cpu",), collective_backend="gloo"),
        runner_factory(),
        kv_cache_config=cache_config(),
    )
    scheduled = batch((1,))
    try:
        executor.execute_model(scheduled)
        with pytest.raises(RuntimeError, match="sampled first"):
            executor.execute_model(scheduled)
    finally:
        executor.shutdown()


def test_executor_resolves_fixed_kv_bytes_to_logical_blocks() -> None:
    config = KVCacheConfig(block_size=4, memory_bytes=8 * 32 + 31)
    executor = UniProcModelExecutor(
        ParallelConfig(devices=("cpu",), collective_backend="gloo"),
        runner_factory(),
        kv_cache_config=config,
    )
    try:
        assert executor.num_gpu_blocks == 8
    finally:
        executor.shutdown()


def test_executor_shuts_down_workers_when_cache_initialization_fails() -> None:
    class TrackedExecutor(UniProcModelExecutor):
        def shutdown(self) -> None:
            super().shutdown()
            assert not self.runner_groups
            shutdowns.append(self)

    shutdowns = []
    with pytest.raises(ValueError, match="CPU KV cache requires num_gpu_blocks"):
        TrackedExecutor(
            ParallelConfig(devices=("cpu",), collective_backend="gloo"),
            runner_factory(),
            kv_cache_config=KVCacheConfig(),
        )
    assert len(shutdowns) == 1


def test_executor_rejects_reserved_parallel_execution_modes() -> None:
    with pytest.raises(NotImplementedError, match="only supports TP"):
        MultiprocModelExecutor(
            ParallelConfig(
                distributed_executor_backend="mp",
                devices=("cpu:0", "cpu:1"),
                pipeline_parallel_size=2,
                collective_backend="gloo",
            ),
            runner_factory(),
            kv_cache_config=cache_config(),
        )


def test_multiproc_executor_runs_tp_spmd_on_cpu_gloo(monkeypatch) -> None:
    executor = MultiprocModelExecutor(
        ParallelConfig(
            distributed_executor_backend="mp",
            devices=("cpu:0", "cpu:1"),
            tensor_parallel_size=2,
            collective_backend="gloo",
            timeout_s=20,
        ),
        runner_factory(),
        kv_cache_config=cache_config(),
    )
    scheduled = batch((7,), (2, 8))
    try:
        collective_rpc = executor.collective_rpc
        methods = []

        def record_method(method, *args, **kwargs):
            methods.append(method)
            return collective_rpc(method, *args, **kwargs)

        monkeypatch.setattr(executor, "collective_rpc", record_method)
        assert executor.execute_step(scheduled) == (8, 9)
        assert methods == ["execute_and_sample"]

        output = executor.execute_model(scheduled)

        assert executor.sample_tokens(output, scheduled) == (8, 9)
        executor.check_health()
    finally:
        executor.shutdown()
