"""Device, distributed, and rank-level runner lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from x_tokens.config import KVCacheConfig, ParallelConfig
from x_tokens.distributed import ParallelTopology, materialize_parallel_context
from x_tokens.worker.protocol import ModelWorker

from .model_runner import (
    ModelRunner,
    ModelRunnerFactory,
    ModelRunnerInput,
    ModelRunnerOutput,
)


@dataclass(frozen=True, slots=True)
class GPUWorkerConfig:
    parallel_config: ParallelConfig
    topology: ParallelTopology
    rank: int
    distributed_init_method: str
    output_rank: int


class GPUWorker(ModelWorker):
    def __init__(
        self,
        config: GPUWorkerConfig,
        runner_factory: ModelRunnerFactory,
        *,
        torch_module: Any | None = None,
    ) -> None:
        if torch_module is None:
            import torch

            torch_module = torch
        self.config = config
        self.runner_factory = runner_factory
        self.torch_module = torch_module
        self.model_runner: ModelRunner | None = None
        self.initialized = False

    def init_device(self) -> None:
        if self.initialized:
            raise RuntimeError("worker device is already initialized")
        assignment = self.config.topology.assignment(self.config.rank)
        device = self.torch_module.device(assignment.device)
        if device.type == "cuda":
            self.torch_module.cuda.set_device(device)
        if self.config.topology.world_size > 1:
            self.torch_module.distributed.init_process_group(
                backend=self.config.parallel_config.collective_backend,
                init_method=self.config.distributed_init_method,
                world_size=self.config.topology.world_size,
                rank=self.config.rank,
                timeout=timedelta(seconds=self.config.parallel_config.timeout_s),
            )
        context = materialize_parallel_context(
            self.config.topology,
            self.config.rank,
            torch_module=self.torch_module,
        )
        self.model_runner = self.runner_factory(
            context, is_output_rank=self.config.rank == self.config.output_rank
        )
        self.initialized = True

    def load_model(self) -> None:
        self._runner().load_model()
        if (
            self.config.topology.world_size > 1
            and self.torch_module.distributed.is_initialized()
        ):
            assignment = self.config.topology.assignment(self.config.rank)
            device = self.torch_module.device(assignment.device)
            barrier_kwargs = (
                {"device_ids": [device.index]}
                if device.type == "cuda" and device.index is not None
                else {}
            )
            self.torch_module.distributed.barrier(**barrier_kwargs)

    def execute_model(self, inputs: ModelRunnerInput) -> ModelRunnerOutput | None:
        return self._runner().execute_model(inputs)

    def execute_and_sample(self, inputs: ModelRunnerInput) -> tuple[int, ...] | None:
        runner = self._runner()
        combined = getattr(runner, "execute_and_sample", None)
        if combined is not None:
            return combined(inputs)
        output = runner.execute_model(inputs)
        if self.config.rank != self.config.output_rank:
            if output is not None:
                raise RuntimeError("a non-output rank returned a model result")
            return None
        if (
            not isinstance(output, ModelRunnerOutput)
            or output.step_id != inputs.step_id
        ):
            raise RuntimeError("output rank returned an invalid model result")
        return runner.sample_tokens(inputs.step_id)

    def get_kv_cache_capacity(self, config: KVCacheConfig) -> int:
        return self._runner().get_kv_cache_capacity(config)

    def initialize_kv_cache(self, num_blocks: int, block_size: int) -> None:
        self._runner().initialize_kv_cache(num_blocks, block_size)

    def sample_tokens(self, step_id: int) -> tuple[int, ...] | None:
        return self._runner().sample_tokens(step_id)

    def check_health(self) -> None:
        self._runner()

    def shutdown(self) -> None:
        if self.model_runner is not None:
            self.model_runner.shutdown()
            self.model_runner = None
        if (
            self.config.topology.world_size > 1
            and self.torch_module.distributed.is_initialized()
        ):
            self.torch_module.distributed.destroy_process_group()
        self.initialized = False

    def _runner(self) -> ModelRunner:
        if self.model_runner is None:
            raise RuntimeError("worker model runner is not initialized")
        return self.model_runner
