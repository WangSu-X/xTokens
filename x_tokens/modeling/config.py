"""Immutable model loading and architecture configurations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from x_tokens.config import ParallelConfig


@dataclass(frozen=True, slots=True)
class ModelLoadConfig:
    model: str
    revision: str | None = None
    cache_dir: str | None = None
    local_files_only: bool = False
    dtype: str = "auto"
    strict_weights: bool = True

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("model must not be empty")


@dataclass(frozen=True, slots=True)
class Qwen3MoeModelConfig:
    model_type: str
    architectures: tuple[str, ...]
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    rope_scaling: Mapping[str, Any] | None
    max_position_embeddings: int
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    decoder_sparse_step: int
    mlp_only_layers: tuple[int, ...]
    intermediate_size: int
    norm_topk_prob: bool
    tie_word_embeddings: bool
    attention_bias: bool
    hidden_act: str
    torch_dtype: str

    @classmethod
    def from_hf_config(cls, config: Mapping[str, Any]) -> Qwen3MoeModelConfig:
        required = (
            "vocab_size",
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "rms_norm_eps",
            "max_position_embeddings",
            "num_experts_per_tok",
            "moe_intermediate_size",
            "intermediate_size",
        )
        missing = [name for name in required if name not in config]
        if "num_experts" not in config and "num_local_experts" not in config:
            missing.append("num_experts")
        if missing:
            raise ValueError(
                "Qwen3-MoE config is missing required fields: " + ", ".join(missing)
            )
        model_config = cls(
            model_type=str(config.get("model_type", "")),
            architectures=tuple(str(item) for item in config.get("architectures", ())),
            vocab_size=int(config["vocab_size"]),
            hidden_size=int(config["hidden_size"]),
            num_hidden_layers=int(config["num_hidden_layers"]),
            num_attention_heads=int(config["num_attention_heads"]),
            num_key_value_heads=int(config["num_key_value_heads"]),
            head_dim=int(config["head_dim"]),
            rms_norm_eps=float(config["rms_norm_eps"]),
            rope_theta=float(
                config.get(
                    "rope_theta",
                    config.get("rope_parameters", {}).get("rope_theta", 10000.0),
                )
            ),
            rope_scaling=config.get("rope_scaling"),
            max_position_embeddings=int(config["max_position_embeddings"]),
            num_experts=int(
                config["num_experts"]
                if "num_experts" in config
                else config["num_local_experts"]
            ),
            num_experts_per_tok=int(config["num_experts_per_tok"]),
            moe_intermediate_size=int(config["moe_intermediate_size"]),
            decoder_sparse_step=int(config.get("decoder_sparse_step", 1)),
            mlp_only_layers=tuple(
                int(item) for item in config.get("mlp_only_layers", ())
            ),
            intermediate_size=int(config["intermediate_size"]),
            norm_topk_prob=bool(config.get("norm_topk_prob", False)),
            tie_word_embeddings=bool(config.get("tie_word_embeddings", False)),
            attention_bias=bool(config.get("attention_bias", False)),
            hidden_act=str(config.get("hidden_act", "silu")),
            torch_dtype=str(config.get("torch_dtype", config.get("dtype", "float32"))),
        )
        model_config.validate()
        return model_config

    def validate(self) -> None:
        if self.model_type != "qwen3_moe":
            raise ValueError(f"unsupported model_type for Qwen3-MoE: {self.model_type}")
        positive = {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "max_position_embeddings": self.max_position_embeddings,
            "num_experts": self.num_experts,
            "num_experts_per_tok": self.num_experts_per_tok,
            "moe_intermediate_size": self.moe_intermediate_size,
            "decoder_sparse_step": self.decoder_sparse_step,
            "intermediate_size": self.intermediate_size,
        }
        for name, value in positive.items():
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )
        if self.num_experts_per_tok > self.num_experts:
            raise ValueError("num_experts_per_tok must not exceed num_experts")
        if self.hidden_act != "silu":
            raise ValueError("only hidden_act=silu is supported")
        if self.attention_bias:
            raise ValueError("attention_bias=True is not supported")
        if self.rope_scaling not in (
            None,
            {"rope_type": "default"},
            {"type": "default"},
        ):
            raise ValueError("only default RoPE scaling is supported")
        if any(
            layer < 0 or layer >= self.num_hidden_layers
            for layer in self.mlp_only_layers
        ):
            raise ValueError("mlp_only_layers contains an invalid layer index")

    def validate_parallel(self, parallel_config: ParallelConfig) -> None:
        tp_size = parallel_config.tensor_parallel_size
        if self.num_attention_heads % tp_size:
            raise ValueError("num_attention_heads must be divisible by TP size")
        if self.num_key_value_heads >= tp_size:
            if self.num_key_value_heads % tp_size:
                raise ValueError("num_key_value_heads must be divisible by TP size")
        elif tp_size % self.num_key_value_heads:
            raise ValueError(
                "TP size must be divisible by num_key_value_heads when KV heads replicate"
            )
        if self.moe_intermediate_size % tp_size:
            raise ValueError("moe_intermediate_size must be divisible by TP size")
        if self.intermediate_size % tp_size:
            raise ValueError("intermediate_size must be divisible by TP size")

    def is_sparse_layer(self, layer_index: int) -> bool:
        return layer_index not in self.mlp_only_layers and (
            (layer_index + 1) % self.decoder_sparse_step == 0
        )
