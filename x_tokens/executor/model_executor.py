"""Common orchestration for local and multiprocess model executors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import suppress
from typing import Any

from x_tokens.config import ParallelConfig, XTokensConfig
from x_tokens.core.protocol import SchedulerOutput
from x_tokens.distributed import ParallelTopology
from x_tokens.worker.model_runner import (
    ModelRunnerOutput,
    ModelRunnerRequest,
    SamplingMetadata,
)

from .protocol import Executor, WorkerHandle


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
        self.failed_error: BaseException | None = None
        self.output_rank = 0
        self.worker_handles: tuple[WorkerHandle, ...] = ()
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
    ) -> tuple[Any, ...]:
        self._raise_if_failed()
        if not self.worker_handles:
            raise RuntimeError("model executor has no active workers")
        try:
            for handle in self.worker_handles:
                handle.send(method, args)
            responses: list[Any] = []
            first_error: Exception | None = None
            for handle in self.worker_handles:
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

    def execute_step(self, batch: SchedulerOutput) -> ModelRunnerOutput:
        if not batch.running_requests:
            return ModelRunnerOutput()
        inputs = self.prepare_worker_input(batch)
        responses = self.collective_rpc("execute_and_sample", inputs)
        output = responses[self.output_rank]
        if not isinstance(output, ModelRunnerOutput):
            raise TypeError("output rank returned an invalid model result")
        return output

    def prepare_worker_input(
        self, batch: SchedulerOutput
    ) -> tuple[ModelRunnerRequest, ...]:
        requests = []
        for item in batch.running_requests:
            sampling = item.sampling_params
            sampling_metadata = (
                SamplingMetadata(sampling.temperature, sampling.top_p, sampling.top_k)
                if sampling is not None
                else None
            )
            requests.append(
                ModelRunnerRequest(
                    request_id=item.request_id,
                    scheduled_token_ids=item.scheduled_token_ids,
                    num_computed_tokens=item.num_computed_tokens,
                    block_ids=item.block_ids,
                    sampling_metadata=sampling_metadata,
                )
            )
        return tuple(requests)

    def check_health(self) -> None:
        self._raise_if_failed()
        for handle in self.worker_handles:
            handle.check_health()

    @abstractmethod
    def shutdown(self) -> None: ...

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
