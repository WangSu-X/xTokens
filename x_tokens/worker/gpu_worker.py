"""Device, distributed, and rank-level runner lifecycle."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from x_tokens.config import KVCacheConfig, XTokensConfig
from x_tokens.distributed import (
    ParallelContext,
    ParallelTopology,
    materialize_parallel_context,
)
from x_tokens.logger import init_logger
from x_tokens.modeling.config import ModelLoadConfig
from x_tokens.modeling.loader import resolve_model_path
from x_tokens.modeling.registry import DEFAULT_MODEL_REGISTRY
from x_tokens.worker.protocol import ModelWorker

from .model_runner import (
    ModelRunner,
    ModelRunnerInput,
    ModelRunnerOutput,
)

logger = init_logger(__name__)


@dataclass(frozen=True, slots=True)
class GPUWorkerConfig:
    xtokens_config: XTokensConfig
    topology: ParallelTopology
    rank: int
    distributed_init_method: str
    output_rank: int
    pad_token_id: int


class GPUWorker(ModelWorker):
    def __init__(
        self,
        config: GPUWorkerConfig,
        *,
        torch_module: Any | None = None,
    ) -> None:
        if torch_module is None:
            import torch

            torch_module = torch
        self.config = config
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
            parallel_config = self.config.xtokens_config.parallel_config
            self.torch_module.distributed.init_process_group(
                backend=parallel_config.collective_backend,
                init_method=self.config.distributed_init_method,
                world_size=self.config.topology.world_size,
                rank=self.config.rank,
                timeout=timedelta(seconds=parallel_config.timeout_s),
            )
        context = materialize_parallel_context(
            self.config.topology,
            self.config.rank,
            torch_module=self.torch_module,
        )
        self.model_runner = self._create_model_runner(context)
        self.initialized = True

    def _create_model_runner(self, context: ParallelContext) -> ModelRunner:
        config = self.config.xtokens_config
        model_name = config.model_config.model_name
        model_path = resolve_model_path(
            model_name,
            local_files_only=config.executor_config.local_files_only,
        )
        with (model_path / "config.json").open(encoding="utf-8") as config_file:
            hf_config = json.load(config_file)
        if not isinstance(hf_config, dict):
            raise TypeError("config.json must contain a JSON object")

        load_config = ModelLoadConfig(
            model_path=model_path,
            hf_config=hf_config,
            dtype=config.executor_config.dtype,
        )
        registration = DEFAULT_MODEL_REGISTRY.resolve(load_config.hf_config)
        is_output_rank = self.config.rank == self.config.output_rank
        if is_output_rank:
            logger.info(
                "resolved model: model=%s path=%s architecture=%s model_type=%s tp=%d",
                model_name,
                load_config.model_path,
                registration.architecture,
                registration.model_type,
                config.parallel_config.tensor_parallel_size,
            )
        return registration.runner_factory(
            load_config,
            context,
            pad_token_id=self.config.pad_token_id,
            is_output_rank=is_output_rank,
        )

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
