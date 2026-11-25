"""Configuration for the Qwen3.5 hybrid decoder used by Qwen3.8-27B."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from x_tokens.config import ParallelConfig


@dataclass(frozen=True, slots=True)
class Qwen3_5TextConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    layer_types: tuple[str, ...]
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int
    rms_norm_eps: float
    rope_theta: float
    rotary_dim: int
    torch_dtype: str
    tie_word_embeddings: bool
    weight_block_size: tuple[int, int] | None

    @classmethod
    def from_hf_config(cls, config: Mapping[str, Any]) -> Qwen3_5TextConfig:
        text = config.get("text_config", config)
        rope = text.get("rope_parameters") or {}
        if (
            text.get("attention_bias", False)
            or not text.get("attn_output_gate", True)
            or text.get("output_gate_type", "swish") != "swish"
            or text.get("hidden_act", "silu") != "silu"
            or rope.get("rope_type", "default") != "default"
        ):
            raise ValueError(
                "unsupported Qwen3.5 attention, activation or RoPE configuration"
            )
        quantization = config.get("quantization_config") or {}
        weight_block_size = None
        if quantization:
            block_size = quantization.get("weight_block_size") or ()
            if (
                quantization.get("quant_method") != "fp8"
                or quantization.get("fmt", "e4m3") != "e4m3"
                or len(block_size) != 2
                or min(block_size) < 1
            ):
                raise ValueError(
                    "Qwen3.5 supports BF16 and block-scaled E4M3 checkpoints"
                )
            weight_block_size = (int(block_size[0]), int(block_size[1]))

        layer_types = tuple(text["layer_types"])
        if (
            len(layer_types) != int(text["num_hidden_layers"])
            or not layer_types
            or set(layer_types) - {"linear_attention", "full_attention"}
        ):
            raise ValueError("invalid Qwen3.5 layer_types")
        head_dim = int(text["head_dim"])
        rotary_dim = int(
            head_dim
            * rope.get("partial_rotary_factor", text.get("partial_rotary_factor", 0.25))
        )
        if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2:
            raise ValueError(
                "Qwen3.5 rotary dimension must be positive, even and within head_dim"
            )
        return cls(
            vocab_size=int(text["vocab_size"]),
            hidden_size=int(text["hidden_size"]),
            intermediate_size=int(text["intermediate_size"]),
            layer_types=layer_types,
            num_attention_heads=int(text["num_attention_heads"]),
            num_key_value_heads=int(text["num_key_value_heads"]),
            head_dim=head_dim,
            linear_num_key_heads=int(text["linear_num_key_heads"]),
            linear_num_value_heads=int(text["linear_num_value_heads"]),
            linear_key_head_dim=int(text["linear_key_head_dim"]),
            linear_value_head_dim=int(text["linear_value_head_dim"]),
            linear_conv_kernel_dim=int(text["linear_conv_kernel_dim"]),
            rms_norm_eps=float(text["rms_norm_eps"]),
            rope_theta=float(rope.get("rope_theta", text.get("rope_theta", 10000.0))),
            rotary_dim=rotary_dim,
            torch_dtype=str(text.get("dtype", text.get("torch_dtype", "bfloat16"))),
            tie_word_embeddings=bool(text.get("tie_word_embeddings", False)),
            weight_block_size=weight_block_size,
        )

    def validate_parallel(self, parallel_config: ParallelConfig) -> None:
        tp = parallel_config.tensor_parallel_size
        if tp not in (1, 2):
            raise ValueError("Qwen3.5 supports TP sizes 1 and 2")
        for name in (
            "num_attention_heads",
            "num_key_value_heads",
            "linear_num_key_heads",
            "linear_num_value_heads",
            "intermediate_size",
        ):
            value = getattr(self, name)
            if value < 1 or value % tp:
                raise ValueError(f"{name} must be positive and divisible by TP size")
        if (
            self.num_attention_heads % self.num_key_value_heads
            or self.linear_num_value_heads % self.linear_num_key_heads
        ):
            raise ValueError(
                "Qwen3.5 query/value heads must group evenly over key heads"
            )
        if (
            min(
                self.linear_key_head_dim,
                self.linear_value_head_dim,
                self.linear_conv_kernel_dim,
            )
            < 1
        ):
            raise ValueError("Qwen3.5 linear attention dimensions must be positive")
