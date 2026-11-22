"""Backend-independent configuration for an EngineCore."""

from __future__ import annotations

from dataclasses import dataclass, replace

from x_tokens.config import KVCacheConfig, XTokensConfig


@dataclass(frozen=True, slots=True)
class EngineCoreConfig:
    """Scheduling and model-routing settings owned by EngineCore."""

    model_aliases: tuple[str, ...]
    kv_cache_config: KVCacheConfig
    max_model_len: int = 4096
    max_num_seqs: int = 4

    def __post_init__(self) -> None:
        if not self.model_aliases:
            raise ValueError("model_aliases must not be empty")
        if self.kv_cache_config.num_gpu_blocks is None:
            raise ValueError("EngineCore requires resolved num_gpu_blocks")
        if self.max_model_len < 2:
            raise ValueError("max_model_len must be at least 2")
        if self.max_num_seqs < 1:
            raise ValueError("max_num_seqs must be positive")

    def accepts_model(self, model: str) -> bool:
        return model in self.model_aliases

    @classmethod
    def from_config(
        cls, config: XTokensConfig, num_gpu_blocks: int
    ) -> EngineCoreConfig:
        if num_gpu_blocks < 1:
            raise RuntimeError("executor did not initialize KV cache")
        return cls(
            model_aliases=(config.model_config.served_model_name,),
            kv_cache_config=replace(
                config.kv_cache_config, num_gpu_blocks=num_gpu_blocks
            ),
            max_model_len=config.model_config.max_model_len,
            max_num_seqs=config.scheduler_config.max_num_seqs,
        )
