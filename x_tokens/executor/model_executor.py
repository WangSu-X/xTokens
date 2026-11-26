"""Common orchestration for local and multiprocess model executors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from x_tokens.config import ParallelConfig, XTokensConfig
from x_tokens.core.protocol import SchedulerOutput
from x_tokens.distributed import ParallelTopology, RunnerGroupSpec
from x_tokens.worker.model_runner import (
    ModelRunnerInput,
    ModelRunnerOutput,
    ModelRunnerRequest,
    SamplingMetadata,
)

from .protocol import Executor, ModelForwardOutput, RunnerStepId, WorkerHandle


@dataclass(slots=True)
class RunnerGroup:
    spec: RunnerGroupSpec
    worker_handles: tuple[WorkerHandle, ...]


class ModelExecutor(Executor, ABC):
    """Engine-facing control plane for one model-parallel replica."""

    @staticmethod
    def get_class(parallel_config: ParallelConfig) -> type[ModelExecutor]:
        if parallel_config.distributed_executor_backend == "uni":
            from .uniproc_model_executor import UniProcModelExecutor

            return UniProcModelExecutor
        if parallel_config.distributed_executor_backend == "mp":
            from .multiproc_model_executor import MultiprocModelExecutor

            return MultiprocModelExecutor
        raise ValueError(
            "unsupported distributed executor backend: "
            f"{parallel_config.distributed_executor_backend}"
        )

    def __init__(
        self,
        config: XTokensConfig,
        *,
        pad_token_id: int,
        eos_token_ids: frozenset[int] = frozenset(),
    ) -> None:
        parallel_config = config.parallel_config
        unsupported_sizes = {
            "pipeline_parallel_size": parallel_config.pipeline_parallel_size,
            "context_parallel_size": parallel_config.context_parallel_size,
            "data_parallel_size": parallel_config.data_parallel_size,
        }
        unsupported = [name for name, size in unsupported_sizes.items() if size != 1]
        if parallel_config.enable_expert_parallel:
            unsupported.append("expert_parallel")
        if unsupported:
            raise NotImplementedError(
                "the first ModelExecutor implementation only supports TP; "
                f"unsupported execution modes: {', '.join(unsupported)}"
            )
        self.config = config
        self.parallel_config = parallel_config
        self.pad_token_id = pad_token_id
        self.topology = ParallelTopology.build(parallel_config)
        self.configured_eos_token_ids = eos_token_ids
        self.num_gpu_blocks = 0
        self.next_step_id = 0
        self.pending_step: RunnerStepId | None = None
        self.failed_error: BaseException | None = None
        self.runner_groups: tuple[RunnerGroup, ...] = ()
        self._init_executor()
        try:
            self._initialize_kv_cache()
        except BaseException:
            with suppress(Exception):
                self.shutdown()
            raise

    @property
    def eos_token_ids(self) -> frozenset[int]:
        return self.configured_eos_token_ids

    @abstractmethod
    def _init_executor(self) -> None: ...

    def collective_rpc(
        self,
        method: str,
        *args: Any,
        runner_group: int = 0,
    ) -> tuple[Any, ...]:
        self._raise_if_failed()
        group = self._group(runner_group)
        try:
            for handle in group.worker_handles:
                handle.send(method, args)
            responses: list[Any] = []
            first_error: Exception | None = None
            for handle in group.worker_handles:
                try:
                    responses.append(handle.recv(self.parallel_config.timeout_s))
                except Exception as error:  # noqa: BLE001 - drain every rank response.
                    responses.append(None)
                    if first_error is None:
                        first_error = error
            if first_error is not None:
                raise first_error
            return tuple(responses)
        except BaseException as error:
            self.failed_error = error
            raise

    def execute_model(self, batch: SchedulerOutput) -> ModelForwardOutput:
        if not batch.running_requests:
            return ModelForwardOutput()
        runner_input = self._prepare_step(batch)
        step = RunnerStepId(0, runner_input.step_id)
        response = self._output_response(
            self.collective_rpc("execute_model", runner_input), runner_group=0
        )
        if (
            not isinstance(response, ModelRunnerOutput)
            or response.step_id != step.step_id
        ):
            raise RuntimeError(
                f"output rank returned an invalid result for step {step.step_id}"
            )
        self.pending_step = step
        return ModelForwardOutput(runner_step_id=step)

    def execute_step(self, batch: SchedulerOutput) -> tuple[int, ...]:
        if not batch.running_requests:
            return ()
        inputs = self._prepare_step(batch)
        responses = self.collective_rpc("execute_and_sample", inputs)
        return self._read_token_batch(responses, batch.num_samples, runner_group=0)

    def _prepare_step(self, batch: SchedulerOutput) -> ModelRunnerInput:
        if self.pending_step is not None:
            raise RuntimeError(
                f"executor step {self.pending_step.step_id} must be sampled first"
            )
        step_id = self.next_step_id
        self.next_step_id += 1
        requests = []
        for item in batch.running_requests:
            request = item.request
            start = request.num_computed_tokens
            count = batch.num_scheduled_tokens[item.request_id]
            sampling = request.sampling_params
            requests.append(
                ModelRunnerRequest(
                    request.all_token_ids[start : start + count],
                    start,
                    item.block_ids,
                    SamplingMetadata(
                        sampling.temperature, sampling.top_p, sampling.top_k
                    )
                    if count == request.num_uncomputed_tokens
                    else None,
                )
            )
        return ModelRunnerInput(step_id, tuple(requests))

    def sample_tokens(
        self, output: ModelForwardOutput, batch: SchedulerOutput
    ) -> tuple[int, ...]:
        if not batch.running_requests:
            return ()
        step = output.runner_step_id
        if not isinstance(step, RunnerStepId) or step != self.pending_step:
            raise RuntimeError("model output does not identify the pending runner step")
        try:
            responses = self.collective_rpc(
                "sample_tokens", step.step_id, runner_group=step.runner_group
            )
            return self._read_token_batch(
                responses, batch.num_samples, runner_group=step.runner_group
            )
        finally:
            self.pending_step = None

    def check_health(self) -> None:
        self._raise_if_failed()
        for group in self.runner_groups:
            for handle in group.worker_handles:
                handle.check_health()

    @abstractmethod
    def shutdown(self) -> None: ...

    def _group(self, index: int) -> RunnerGroup:
        if index < 0 or index >= len(self.runner_groups):
            raise ValueError(f"unknown runner group: {index}")
        return self.runner_groups[index]

    def _output_response(self, responses: tuple[Any, ...], *, runner_group: int) -> Any:
        output_index = self._output_handle_index(self._group(runner_group))
        if any(
            item is not None
            for index, item in enumerate(responses)
            if index != output_index
        ):
            raise RuntimeError("a non-output rank returned a model result")
        return responses[output_index]

    def _read_token_batch(
        self, responses: tuple[Any, ...], batch_size: int, *, runner_group: int
    ) -> tuple[int, ...]:
        token_ids = self._output_response(responses, runner_group=runner_group)
        if not isinstance(token_ids, tuple) or len(token_ids) != batch_size:
            raise RuntimeError("output rank returned an invalid token batch")
        return token_ids

    def _raise_if_failed(self) -> None:
        if self.failed_error is not None:
            raise RuntimeError(
                "model executor is in a failed state"
            ) from self.failed_error

    def _initialize_kv_cache(self) -> None:
        kv_cache_config = self.config.kv_cache_config
        capacities = self.collective_rpc("get_kv_cache_capacity", kv_cache_config)
        if not all(isinstance(capacity, int) for capacity in capacities):
            raise RuntimeError("worker returned an invalid KV cache capacity")
        self.num_gpu_blocks = min(capacities)
        if self.num_gpu_blocks < 1:
            raise RuntimeError("no GPU KV cache blocks are available")
        self.collective_rpc(
            "initialize_kv_cache",
            self.num_gpu_blocks,
            kv_cache_config.block_size,
        )

    @staticmethod
    def _output_handle_index(group: RunnerGroup) -> int:
        return group.spec.ranks.index(group.spec.output_rank)
