"""Inference-only Qwen3-MoE with tensor parallelism and paged KV storage."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from x_tokens.distributed import ParallelContext, ParallelGroupKind
from x_tokens.kv_cache import KVCacheMetadata, KVCacheSpec
from x_tokens.modeling.config import ModelLoadConfig, Qwen3MoeModelConfig
from x_tokens.modeling.interfaces import ModelInput, ModelOutput, XTokensModel
from x_tokens.modeling.layers import (
    MergedColumnParallelLinear,
    ParallelLMHead,
    QKVParallelLinear,
    ReplicatedLinear,
    RMSNorm,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from x_tokens.modeling.loader.safetensors_loader import (
    SafeTensorsWeightLoader,
    WeightLoadReport,
)
from x_tokens.modeling.loader.source import ResolvedModelSource
from x_tokens.modeling.registry import RunnerFactoryBuilder
from x_tokens.worker.kv_cache import GPUKVCache
from x_tokens.worker.model_runner import GPUModelRunner, ModelRunnerFactory


def resolve_torch_dtype(
    load_config: ModelLoadConfig, model_config: Qwen3MoeModelConfig
) -> torch.dtype:
    dtype_name = (
        model_config.torch_dtype if load_config.dtype == "auto" else load_config.dtype
    )
    aliases = {
        "float32": torch.float32,
        "float": torch.float32,
        "float16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return aliases[dtype_name]
    except KeyError as error:
        raise ValueError(f"unsupported model dtype: {dtype_name}") from error


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_dim: int,
        theta: float,
        *,
        device: torch.device,
    ) -> None:
        super().__init__()
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self, position_ids: torch.Tensor, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        frequencies = torch.einsum(
            "bi,j->bij", position_ids.float(), self.inv_freq.float()
        )
        embeddings = torch.cat((frequencies, frequencies), dim=-1)
        return embeddings.cos().to(dtype), embeddings.sin().to(dtype)


def _rotate_half(inputs: torch.Tensor) -> torch.Tensor:
    first, second = inputs.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class Qwen3MoeAttention(nn.Module):
    def __init__(
        self,
        config: Qwen3MoeModelConfig,
        layer_index: int,
        context: ParallelContext,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        group = context.group(ParallelGroupKind.TENSOR)
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads // group.world_size
        self.num_kv_heads = max(1, config.num_key_value_heads // group.world_size)
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim**-0.5
        self.layer_index = layer_index
        self.kv_cache: GPUKVCache | None = None
        kwargs = {
            "group": group,
            "device": context.device,
            "dtype": dtype,
            "initialize": False,
        }
        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            config.head_dim,
            config.num_attention_heads,
            config.num_key_value_heads,
            bias=config.attention_bias,
            **kwargs,
        )
        self.o_proj = RowParallelLinear(
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
            **kwargs,
        )
        self.q_norm = RMSNorm(
            config.head_dim,
            eps=config.rms_norm_eps,
            device=context.device,
            dtype=dtype,
        )
        self.k_norm = RMSNorm(
            config.head_dim,
            eps=config.rms_norm_eps,
            device=context.device,
            dtype=dtype,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache_metadata: tuple[KVCacheMetadata, ...],
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        qkv = self.qkv_proj(hidden_states)
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        query, key, value = qkv.split((q_size, kv_size, kv_size), dim=-1)
        query = self.q_norm(
            query.view(batch_size, sequence_length, self.num_heads, self.head_dim)
        )
        key = self.k_norm(
            key.view(batch_size, sequence_length, self.num_kv_heads, self.head_dim)
        )
        value = value.view(
            batch_size, sequence_length, self.num_kv_heads, self.head_dim
        )
        rope_cos = cos.unsqueeze(2)
        rope_sin = sin.unsqueeze(2)
        query = query * rope_cos + _rotate_half(query) * rope_sin
        key = key * rope_cos + _rotate_half(key) * rope_sin
        query = query.transpose(1, 2)
        if self.kv_cache is None:
            raise RuntimeError("Qwen3-MoE requires an initialized KV cache")
        if len(kv_cache_metadata) != batch_size:
            raise ValueError("Qwen3-MoE requires KV metadata per request")
        self.kv_cache.write(self.layer_index, kv_cache_metadata, key, value)
        key, value = self.kv_cache.gather(self.layer_index, kv_cache_metadata)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        key_length = key.shape[2]
        allowed = torch.zeros(
            batch_size,
            1,
            sequence_length,
            key_length,
            dtype=torch.bool,
            device=hidden_states.device,
        )
        key_positions = torch.arange(key_length, device=hidden_states.device)
        for batch_index, metadata in enumerate(kv_cache_metadata):
            query_start = sequence_length - metadata.query_len
            query_positions = torch.arange(
                metadata.context_len - metadata.query_len,
                metadata.context_len,
                device=hidden_states.device,
            )
            allowed[batch_index, 0, query_start:, : metadata.context_len] = (
                key_positions[: metadata.context_len].unsqueeze(0)
                <= query_positions.unsqueeze(1)
            )
        if self.num_key_value_groups != 1:
            key = key.repeat_interleave(self.num_key_value_groups, dim=1)
            value = value.repeat_interleave(self.num_key_value_groups, dim=1)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=allowed,
            dropout_p=0.0,
            scale=self.scaling,
        )
        attended = attended.transpose(1, 2).reshape(batch_size, sequence_length, q_size)
        return self.o_proj(attended)


class DenseSwiGLU(nn.Module):
    def __init__(
        self,
        config: Qwen3MoeModelConfig,
        context: ParallelContext,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        kwargs = {
            "group": context.group(ParallelGroupKind.TENSOR),
            "device": context.device,
            "dtype": dtype,
            "initialize": False,
        }
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            (config.intermediate_size, config.intermediate_size),
            bias=False,
            **kwargs,
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            **kwargs,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(hidden_states).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class TensorParallelExperts(nn.Module):
    def __init__(
        self,
        config: Qwen3MoeModelConfig,
        context: ParallelContext,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_size = config.hidden_size
        self.group = context.group(ParallelGroupKind.TENSOR)
        self.local_intermediate_size = (
            config.moe_intermediate_size // self.group.world_size
        )
        self.gate_up_proj = nn.Parameter(
            torch.empty(
                config.num_experts,
                2 * self.local_intermediate_size,
                config.hidden_size,
                device=context.device,
                dtype=dtype,
            )
        )
        self.down_proj = nn.Parameter(
            torch.empty(
                config.num_experts,
                config.hidden_size,
                self.local_intermediate_size,
                device=context.device,
                dtype=dtype,
            )
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        output = torch.zeros_like(hidden_states)
        for expert_index_tensor in torch.unique(selected_experts):
            expert_index = int(expert_index_tensor.item())
            token_indices, route_slots = torch.where(selected_experts == expert_index)
            expert_inputs = hidden_states[token_indices]
            gate_up = F.linear(expert_inputs, self.gate_up_proj[expert_index])
            gate, up = gate_up.chunk(2, dim=-1)
            partial = F.linear(F.silu(gate) * up, self.down_proj[expert_index])
            partial = partial * routing_weights[token_indices, route_slots].unsqueeze(
                -1
            )
            output.index_add_(0, token_indices, partial.to(output.dtype))
        return self.group.all_reduce(output)


class Qwen3MoeSparseMoeBlock(nn.Module):
    def __init__(
        self,
        config: Qwen3MoeModelConfig,
        context: ParallelContext,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            device=context.device,
            dtype=dtype,
            initialize=False,
        )
        self.experts = TensorParallelExperts(config, context, dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_shape = hidden_states.shape
        flat = hidden_states.reshape(-1, self.hidden_size)
        router_logits = self.gate(flat)
        probabilities = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        routing_weights, selected_experts = torch.topk(
            probabilities, self.top_k, dim=-1
        )
        if self.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(
                dim=-1, keepdim=True
            )
        routed = self.experts(
            flat, selected_experts, routing_weights.to(router_logits.dtype)
        )
        return routed.reshape(original_shape)


class Qwen3MoeDecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen3MoeModelConfig,
        layer_index: int,
        context: ParallelContext,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3MoeAttention(config, layer_index, context, dtype)
        self.mlp: DenseSwiGLU | Qwen3MoeSparseMoeBlock
        if config.is_sparse_layer(layer_index):
            self.mlp = Qwen3MoeSparseMoeBlock(config, context, dtype)
        else:
            self.mlp = DenseSwiGLU(config, context, dtype)
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            device=context.device,
            dtype=dtype,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            device=context.device,
            dtype=dtype,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache_metadata: tuple[KVCacheMetadata, ...],
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(
            self.input_layernorm(hidden_states),
            cos,
            sin,
            kv_cache_metadata,
        )
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


class Qwen3MoeForCausalLM(nn.Module, XTokensModel):
    def __init__(
        self,
        config: Qwen3MoeModelConfig,
        context: ParallelContext,
        dtype: torch.dtype,
        source: ResolvedModelSource,
        load_config: ModelLoadConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.context = context
        self.source = source
        self.load_config = load_config
        tp_group = context.group(ParallelGroupKind.TENSOR)
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            group=tp_group,
            device=context.device,
            dtype=dtype,
        )
        self.layers = nn.ModuleList(
            Qwen3MoeDecoderLayer(config, index, context, dtype)
            for index in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            device=context.device,
            dtype=dtype,
        )
        self.rotary_emb = RotaryEmbedding(
            config.head_dim, config.rope_theta, device=context.device
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            group=tp_group,
            device=context.device,
            dtype=dtype,
        )
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        self.load_report: WeightLoadReport | None = None
        self.kv_cache: GPUKVCache | None = None

    def load_model(self) -> None:
        loader = SafeTensorsWeightLoader(
            self.source,
            strict=self.load_config.strict_weights,
        )
        self.load_report = loader.load(self)

    def forward(self, inputs: ModelInput) -> ModelOutput:
        if inputs.input_ids is None:
            raise ValueError("Qwen3-MoE TP baseline requires input_ids")
        if not inputs.kv_cache_metadata:
            raise ValueError("Qwen3-MoE requires KV cache metadata")
        hidden_states = self.embed_tokens(inputs.input_ids)
        cos, sin = self.rotary_emb(inputs.position_ids, hidden_states.dtype)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                cos,
                sin,
                inputs.kv_cache_metadata,
            )
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states[:, -1, :])
        return ModelOutput(logits)

    def get_kv_cache_spec(self, block_size: int) -> KVCacheSpec:
        return KVCacheSpec(
            num_layers=self.config.num_hidden_layers,
            block_size=block_size,
            num_kv_heads=self.layers[0].self_attn.num_kv_heads,
            head_dim=self.config.head_dim,
            dtype_bytes=self.embed_tokens.weight.element_size(),
        )

    def set_kv_cache(self, kv_cache: GPUKVCache | None) -> None:
        self.kv_cache = kv_cache
        for layer in self.layers:
            layer.self_attn.kv_cache = kv_cache


@dataclass(frozen=True, slots=True)
class Qwen3MoeRunnerFactory(RunnerFactoryBuilder, ModelRunnerFactory):
    source: ResolvedModelSource
    model_config: Qwen3MoeModelConfig
    load_config: ModelLoadConfig
    pad_token_id: int

    def __call__(
        self, context: ParallelContext, *, is_output_rank: bool
    ) -> GPUModelRunner:
        dtype = resolve_torch_dtype(self.load_config, self.model_config)
        model = Qwen3MoeForCausalLM(
            self.model_config,
            context,
            dtype,
            self.source,
            self.load_config,
        )
        return GPUModelRunner(
            context,
            model,
            pad_token_id=self.pad_token_id,
            is_output_rank=is_output_rank,
        )
