from __future__ import annotations

import pytest
import torch
from torch import nn

from x_tokens.config import KVCacheConfig, ParallelConfig, XTokensConfig
from x_tokens.core.protocol import SchedulerOutput, SchedulerRequest
from x_tokens.distributed import ParallelContext
from x_tokens.engine.protocol import SamplingParams
from x_tokens.executor import (
    ModelExecutor,
    MultiprocModelExecutor,
    UniProcModelExecutor,
)
from x_tokens.kv_cache import KVCacheSpec
from x_tokens.modeling import ModelInput, ModelOutput
from x_tokens.worker import GPUWorker
from x_tokens.worker.model_runner import (
    GPUModelRunner,
    ModelRunnerOutput,
    ModelRunnerRequest,
)


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


@pytest.fixture
def tiny_runner(monkeypatch) -> None:
    def create_runner(worker: GPUWorker, context: ParallelContext) -> GPUModelRunner:
        return GPUModelRunner(
            context,
            TinyNextTokenModel(),
            pad_token_id=worker.config.pad_token_id,
            is_output_rank=worker.config.rank == worker.config.output_rank,
        )

    monkeypatch.setattr(GPUWorker, "_create_model_runner", create_runner)


def batch(*contexts: tuple[int, ...]) -> SchedulerOutput:
    requests = []
    num_scheduled_tokens = {}
    for index, context in enumerate(contexts):
        request_id = f"request-{index}"
        requests.append(
            SchedulerRequest(
                request_id=request_id,
                scheduled_token_ids=context,
                num_computed_tokens=0,
                block_ids=(index,),
                sampling_params=SamplingParams(max_tokens=2, temperature=0),
            )
        )
        num_scheduled_tokens[request_id] = len(context)
    return SchedulerOutput(tuple(requests), num_scheduled_tokens)


def cpu_config(kv_cache_config: KVCacheConfig) -> XTokensConfig:
    return XTokensConfig(
        parallel_config=ParallelConfig(devices=("cpu",), collective_backend="gloo"),
        kv_cache_config=kv_cache_config,
    )


def cache_config() -> KVCacheConfig:
    return KVCacheConfig(block_size=4, num_gpu_blocks=8)


@pytest.mark.usefixtures("tiny_runner")
def test_uniproc_executor_samples_by_request_id() -> None:
    executor = UniProcModelExecutor(
        cpu_config(cache_config()), pad_token_id=0, eos_token_ids=frozenset((15,))
    )
    scheduled = batch((1, 2), (4,))
    try:
        output = executor.execute_step(scheduled)

        assert output == ModelRunnerOutput({"request-0": 3, "request-1": 5})
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


@pytest.mark.usefixtures("tiny_runner")
def test_executor_runs_steps_serially_without_step_ids() -> None:
    executor = UniProcModelExecutor(cpu_config(cache_config()), pad_token_id=0)
    scheduled = batch((1,))
    try:
        first = executor.execute_step(scheduled)
        second = executor.execute_step(scheduled)
        assert first.sampled_token_ids == second.sampled_token_ids
    finally:
        executor.shutdown()


@pytest.mark.usefixtures("tiny_runner")
def test_executor_passes_only_the_scheduled_chunk_to_worker() -> None:
    executor = UniProcModelExecutor(cpu_config(cache_config()), pad_token_id=0)
    scheduled_request = SchedulerRequest(
        request_id="prefill",
        scheduled_token_ids=(7, 8, 9),
        num_computed_tokens=4,
        block_ids=(1, 2),
        sampling_params=None,
    )
    batch = SchedulerOutput((scheduled_request,), {"prefill": 3})

    try:
        assert executor.prepare_worker_input(batch) == (
            ModelRunnerRequest(
                request_id="prefill",
                scheduled_token_ids=(7, 8, 9),
                num_computed_tokens=4,
                block_ids=(1, 2),
                sampling_metadata=None,
            ),
        )
    finally:
        executor.shutdown()


@pytest.mark.usefixtures("tiny_runner")
def test_executor_resolves_fixed_kv_bytes_to_logical_blocks() -> None:
    config = KVCacheConfig(block_size=4, memory_bytes=8 * 32 + 31)
    executor = UniProcModelExecutor(cpu_config(config), pad_token_id=0)
    try:
        assert executor.num_gpu_blocks == 8
    finally:
        executor.shutdown()


@pytest.mark.usefixtures("tiny_runner")
def test_executor_shuts_down_workers_when_cache_initialization_fails() -> None:
    class TrackedExecutor(UniProcModelExecutor):
        def shutdown(self) -> None:
            super().shutdown()
            assert not self.worker_handles
            shutdowns.append(self)

    shutdowns = []
    with pytest.raises(ValueError, match="CPU KV cache requires num_gpu_blocks"):
        TrackedExecutor(cpu_config(KVCacheConfig()), pad_token_id=0)
    assert len(shutdowns) == 1


def test_executor_rejects_reserved_parallel_execution_modes() -> None:
    config = XTokensConfig(
        parallel_config=ParallelConfig(
            distributed_executor_backend="mp",
            devices=("cpu:0", "cpu:1"),
            pipeline_parallel_size=2,
            collective_backend="gloo",
        ),
        kv_cache_config=cache_config(),
    )
    with pytest.raises(NotImplementedError, match="only supports TP"):
        MultiprocModelExecutor(config, pad_token_id=0)
