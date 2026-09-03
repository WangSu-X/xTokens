"""Common orchestration for local and multiprocess model executors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from x_tokens.config import KVCacheConfig, ParallelConfig
from x_tokens.core.scheduler import SchedulerOutput
from x_tokens.distributed import ParallelTopology, RunnerGroupSpec
from x_tokens.worker.model_runner import (
    ModelRunnerFactory,
    ModelRunnerInput,
    ModelRunnerOutput,
    SamplingMetadata,
)

from .base import Executor, ModelForwardOutput
from .worker_handle import WorkerHandle


@dataclass(frozen=True, slots=True)
class RunnerStepId:
    runner_group: int
    step_id: int


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
        parallel_config: ParallelConfig,
        runner_factory: ModelRunnerFactory,
        *,
        eos_token_ids: frozenset[int] = frozenset(),
        kv_cache_config: KVCacheConfig,
    ) -> None:
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
        self.parallel_config = parallel_config
        self.runner_factory = runner_factory
        self.topology = ParallelTopology.build(parallel_config)
        self._eos_token_ids = eos_token_ids
        self.kv_cache_config = kv_cache_config
        self.num_gpu_blocks = 0
        self._next_step_id = 0
        self._pending_step: RunnerStepId | None = None
        self._failed_error: BaseException | None = None
        self.runner_groups: tuple[RunnerGroup, ...] = ()
        self._init_executor()
        self._initialize_kv_cache()

    @property
    def eos_token_ids(self) -> frozenset[int]:
        return self._eos_token_ids

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
            self._failed_error = error
            raise

    def execute_model(self, batch: SchedulerOutput) -> ModelForwardOutput:
        if not batch.requests:
            return ModelForwardOutput()
        if self._pending_step is not None:
            raise RuntimeError(
                f"executor step {self._pending_step.step_id} must be sampled first"
            )
        if len(batch.kv_cache_metadata) != len(batch.requests):
            raise ValueError("custom executor requires KV cache metadata")
        runner_group = 0
        step = RunnerStepId(runner_group, self._next_step_id)
        self._next_step_id += 1
        token_ids = tuple(
            request.context_token_ids[-metadata.query_len :]
            for request, metadata in zip(
                batch.requests, batch.kv_cache_metadata, strict=True
            )
        )
        runner_input = ModelRunnerInput(
            step.step_id,
            token_ids,
            tuple(
                SamplingMetadata(
                    request.request.sampling.temperature,
                    request.request.sampling.top_p,
                    request.request.sampling.top_k,
                )
                for request in batch.requests
            ),
            batch.kv_cache_metadata,
        )
        responses = self.collective_rpc(
            "execute_model", runner_input, runner_group=runner_group
        )
        output_index = self._output_handle_index(self._group(runner_group))
        response = responses[output_index]
        if (
            not isinstance(response, ModelRunnerOutput)
            or response.step_id != step.step_id
        ):
            raise RuntimeError(
                f"output rank returned an invalid result for step {step.step_id}"
            )
        if any(
            item is not None
            for index, item in enumerate(responses)
            if index != output_index
        ):
            raise RuntimeError("a non-output rank returned a model result")
        self._pending_step = step
        return ModelForwardOutput(runner_step_id=step)

    def sample_tokens(
        self, output: ModelForwardOutput, batch: SchedulerOutput
    ) -> tuple[int, ...]:
        if not batch.requests:
            return ()
        step = output.runner_step_id
        if not isinstance(step, RunnerStepId) or step != self._pending_step:
            raise RuntimeError("model output does not identify the pending runner step")
        try:
            responses = self.collective_rpc(
                "sample_tokens", step.step_id, runner_group=step.runner_group
            )
            group = self._group(step.runner_group)
            token_ids = responses[self._output_handle_index(group)]
            if not isinstance(token_ids, tuple) or len(token_ids) != len(
                batch.requests
            ):
                raise RuntimeError("output rank returned an invalid token batch")
            return token_ids
        finally:
            self._pending_step = None

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

    def _raise_if_failed(self) -> None:
        if self._failed_error is not None:
            raise RuntimeError(
                "model executor is in a failed state"
            ) from self._failed_error

    def _initialize_kv_cache(self) -> None:
        capacities = self.collective_rpc("get_kv_cache_capacity", self.kv_cache_config)
        if not all(isinstance(capacity, int) for capacity in capacities):
            raise RuntimeError("worker returned an invalid KV cache capacity")
        self.num_gpu_blocks = min(capacities)
        if self.num_gpu_blocks < 1:
            raise RuntimeError("no GPU KV cache blocks are available")
        self.collective_rpc(
            "initialize_kv_cache",
            self.num_gpu_blocks,
            self.kv_cache_config.block_size,
        )

    @staticmethod
    def _output_handle_index(group: RunnerGroup) -> int:
        return group.spec.ranks.index(group.spec.output_rank)


def create_model_executor(
    parallel_config: ParallelConfig,
    runner_factory: ModelRunnerFactory,
    *,
    eos_token_ids: frozenset[int] = frozenset(),
    kv_cache_config: KVCacheConfig,
) -> ModelExecutor:
    executor_class = ModelExecutor.get_class(parallel_config)
    return executor_class(
        parallel_config,
        runner_factory,
        eos_token_ids=eos_token_ids,
        kv_cache_config=kv_cache_config,
    )
