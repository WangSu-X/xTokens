from __future__ import annotations

import pytest
import torch
from torch import nn

from x_tokens.config import ParallelConfig
from x_tokens.core.scheduler import ScheduledRequest, SchedulerOutput
from x_tokens.engine.types import GenerateRequest, SamplingParams
from x_tokens.executor import (
    ModelExecutor,
    MultiprocModelExecutor,
    UniProcModelExecutor,
)
from x_tokens.modeling import ModelInput, ModelOutput
from x_tokens.worker.model_runner import GPUModelRunnerFactory


class TinyNextTokenModel(nn.Module):
    def __init__(self, vocab_size: int = 16) -> None:
        super().__init__()
        self.vocab_size = vocab_size

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


def tiny_model_factory(context: object) -> TinyNextTokenModel:
    del context
    return TinyNextTokenModel()


def batch(*contexts: tuple[int, ...]) -> SchedulerOutput:
    requests = []
    for index, context in enumerate(contexts):
        request = GenerateRequest(
            f"request-{index}",
            "tiny",
            context,
            SamplingParams(max_tokens=2, temperature=0),
        )
        requests.append(ScheduledRequest(request, context))
    return SchedulerOutput(tuple(requests))


def runner_factory() -> GPUModelRunnerFactory:
    return GPUModelRunnerFactory(tiny_model_factory, pad_token_id=0)


def test_uniproc_executor_runs_pending_step_and_samples_on_device() -> None:
    executor = UniProcModelExecutor(
        ParallelConfig(devices=("cpu",), collective_backend="gloo"),
        runner_factory(),
        eos_token_ids=frozenset((15,)),
    )
    scheduled = batch((1, 2), (4,))
    try:
        output = executor.execute_model(scheduled)

        assert output.logits is None
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
    )
    scheduled = batch((1,))
    try:
        executor.execute_model(scheduled)
        with pytest.raises(RuntimeError, match="sampled first"):
            executor.execute_model(scheduled)
    finally:
        executor.shutdown()


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
        )


def test_multiproc_executor_runs_tp_spmd_on_cpu_gloo() -> None:
    executor = MultiprocModelExecutor(
        ParallelConfig(
            distributed_executor_backend="mp",
            devices=("cpu:0", "cpu:1"),
            tensor_parallel_size=2,
            collective_backend="gloo",
            timeout_s=20,
        ),
        runner_factory(),
    )
    scheduled = batch((7,), (2, 8))
    try:
        output = executor.execute_model(scheduled)

        assert executor.sample_tokens(output, scheduled) == (8, 9)
        executor.check_health()
    finally:
        executor.shutdown()
