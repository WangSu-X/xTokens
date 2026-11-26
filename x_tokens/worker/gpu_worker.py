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
    ModelRunnerOutput,
    ModelRunnerRequest,
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
        self.model_runner: ModelRunner
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
        self.model_runner.load_model()
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

    def execute_and_sample(
        self, requests: tuple[ModelRunnerRequest, ...]
    ) -> ModelRunnerOutput | None:
        self.model_runner.execute_model(requests)
        if self.config.rank == self.config.output_rank:
            return self.model_runner.sample_tokens()
        return None

    def get_kv_cache_capacity(self, config: KVCacheConfig) -> int:
        return self.model_runner.get_kv_cache_capacity(config)

    def initialize_kv_cache(self, num_blocks: int, block_size: int) -> None:
        self.model_runner.initialize_kv_cache(num_blocks, block_size)

    def check_health(self) -> None:
        if not self.initialized:
            raise RuntimeError("worker model runner is not initialized")

    def shutdown(self) -> None:
        if self.initialized:
            self.model_runner.shutdown()
            del self.model_runner
            self.initialized = False
        if (
            self.config.topology.world_size > 1
            and self.torch_module.distributed.is_initialized()
        ):
            self.torch_module.distributed.destroy_process_group()
