"""Top-level xTokens configuration and structured sub-configurations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True, slots=True)
class ModelConfig:
    served_model_name: str = "x-tokens"
    model: str | None = None
    max_model_len: int = 4096

    def __post_init__(self) -> None:
        if not self.served_model_name:
            raise ValueError("served_model_name must not be empty")
        if self.model is not None and not self.model:
            raise ValueError("model must not be empty")
        if self.max_model_len < 2:
            raise ValueError("max_model_len must be at least 2")

    @property
    def model_name(self) -> str:
        return self.model or self.served_model_name


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    max_num_seqs: int = 4

    def __post_init__(self) -> None:
        if self.max_num_seqs < 1:
            raise ValueError("max_num_seqs must be positive")


@dataclass(frozen=True, slots=True)
class ExecutorConfig:
    dtype: str = "auto"
    local_files_only: bool = False


@dataclass(frozen=True, slots=True)
class KVCacheConfig:
    block_size: int = 16
    gpu_memory_utilization: float = 0.9
    num_gpu_blocks: int | None = None
    enable_prefix_caching: bool = True

    def __post_init__(self) -> None:
        if self.block_size < 1:
            raise ValueError("KV cache block_size must be positive")
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        if self.num_gpu_blocks is not None and self.num_gpu_blocks < 1:
            raise ValueError("num_gpu_blocks must be positive")


@dataclass(frozen=True, slots=True)
class ParallelConfig:
    """Immutable process and model-parallel settings.

    The physical rank layout is ``DP x PP x CP x TP``. Expert parallelism is
    an overlay on those ranks and therefore does not increase ``world_size``.
    """

    distributed_executor_backend: Literal["uni", "mp"] = "uni"
    devices: tuple[str, ...] = ("cuda:0",)
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    context_parallel_size: int = 1
    data_parallel_size: int = 1
    enable_expert_parallel: bool = False
    expert_parallel_size: int | None = None
    collective_backend: str = "nccl"
    timeout_s: float = 120.0

    def __post_init__(self) -> None:
        sizes = {
            "tensor_parallel_size": self.tensor_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "context_parallel_size": self.context_parallel_size,
            "data_parallel_size": self.data_parallel_size,
        }
        for name, size in sizes.items():
            if size < 1:
                raise ValueError(f"{name} must be positive")
        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if not self.collective_backend:
            raise ValueError("collective_backend must not be empty")
        if len(self.devices) != self.world_size:
            raise ValueError(
                "devices must contain exactly one entry per physical rank: "
                f"expected {self.world_size}, got {len(self.devices)}"
            )
        if len(set(self.devices)) != len(self.devices):
            raise ValueError("devices must not contain duplicates")
        if self.distributed_executor_backend == "uni" and self.world_size != 1:
            raise ValueError("uni executor requires world_size=1")
        if not self.enable_expert_parallel and self.expert_parallel_size is not None:
            raise ValueError(
                "expert_parallel_size requires enable_expert_parallel=True"
            )
        if self.enable_expert_parallel:
            ep_size = self.expert_parallel_size or self.model_parallel_size
            if ep_size < 1 or self.model_parallel_size % ep_size != 0:
                raise ValueError(
                    "expert_parallel_size must divide TP x CP within a runner group"
                )

    @property
    def model_parallel_size(self) -> int:
        return self.tensor_parallel_size * self.context_parallel_size

    @property
    def runner_group_size(self) -> int:
        return (
            self.pipeline_parallel_size
            * self.context_parallel_size
            * self.tensor_parallel_size
        )

    @property
    def world_size(self) -> int:
        return self.data_parallel_size * self.runner_group_size


@dataclass(frozen=True, slots=True)
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    api_key: str | None = None
    request_timeout_s: float | None = 600.0
    shutdown_timeout_s: float = 30.0
    shutdown_policy: Literal["abort", "drain"] = "abort"
    max_request_body_size: int = 4 * 1024 * 1024
    cors_origins: tuple[str, ...] = ()
    access_log: bool = True
    core_client_mode: Literal["in_proc", "multi_proc"] = "in_proc"

    def __post_init__(self) -> None:
        if self.port < 1 or self.port > 65535:
            raise ValueError("port must be between 1 and 65535")
        if self.request_timeout_s is not None and self.request_timeout_s <= 0:
            raise ValueError("request_timeout_s must be positive or None")
        if self.shutdown_timeout_s <= 0:
            raise ValueError("shutdown_timeout_s must be positive")
        if self.max_request_body_size <= 0:
            raise ValueError("max_request_body_size must be positive")


@dataclass(frozen=True, slots=True)
class XTokensConfig:
    model_config: ModelConfig = field(default_factory=ModelConfig)
    scheduler_config: SchedulerConfig = field(default_factory=SchedulerConfig)
    executor_config: ExecutorConfig = field(default_factory=ExecutorConfig)
    kv_cache_config: KVCacheConfig = field(default_factory=KVCacheConfig)
    parallel_config: ParallelConfig = field(default_factory=ParallelConfig)
    server_config: ServerConfig = field(default_factory=ServerConfig)
