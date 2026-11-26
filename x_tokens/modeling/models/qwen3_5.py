"""Text inference for the hybrid Qwen3.5 decoder and Qwen3.8-27B."""

from __future__ import annotations

from itertools import accumulate

import torch
import torch.nn.functional as F
from torch import nn

from x_tokens.distributed import ParallelContext, ParallelGroupKind
from x_tokens.kv_cache import KVCacheSlots, KVCacheSpec, LinearAttentionCacheSpec
from x_tokens.modeling.config import ModelLoadConfig
from x_tokens.modeling.layers import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ParallelLMHead,
    RMSNorm,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from x_tokens.modeling.layers.rotary import RotaryEmbedding, rotate_half
from x_tokens.modeling.loader.qwen3_5_loader import load_qwen3_5_weights
from x_tokens.modeling.loader.safetensors_loader import WeightLoadReport
from x_tokens.modeling.protocol import ModelInput, ModelOutput
from x_tokens.modeling.qwen3_5_config import Qwen3_5TextConfig
from x_tokens.worker.kv_cache import GPUKVCache
from x_tokens.worker.model_runner import GPUModelRunner


class Qwen3_5Model(nn.Module):
    """Qwen3.5 text decoder that produces normalized last-token hidden states."""

    def __init__(
        self,
        config: Qwen3_5TextConfig,
        context: ParallelContext,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        group = context.group(ParallelGroupKind.TENSOR)
        tensor_options = {"device": context.device, "dtype": dtype}
        linear_options = dict(tensor_options, group=group, bias=False, initialize=False)
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size, group=group, **tensor_options
        )
        self.norm = OffsetRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, **tensor_options
        )
        self.layers = nn.ModuleList()
        self.num_full_layers = 0
        self.num_linear_layers = 0
        for layer_type in config.layer_types:
            if layer_type == "full_attention":
                cache_layer = self.num_full_layers
                self.num_full_layers += 1
            else:
                cache_layer = self.num_linear_layers
                self.num_linear_layers += 1
            self.layers.append(
                Qwen3_5DecoderLayer(config, layer_type, cache_layer, linear_options)
            )
        self.rotary_emb = RotaryEmbedding(
            config.rotary_dim, config.rope_theta, device=context.device
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cache: GPUKVCache,
        metadata: tuple[KVCacheSlots, ...],
    ) -> torch.Tensor:
        hidden = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(position_ids, hidden.dtype)
        cos, sin = cos.unsqueeze(2), sin.unsqueeze(2)
        for layer in self.layers:
            hidden = layer(hidden, cache, metadata, position_ids, cos, sin)
        return self.norm(hidden[:, -1])


class Qwen3_5DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen3_5TextConfig,
        layer_type: str,
        cache_layer: int,
        linear_options: dict,
    ) -> None:
        super().__init__()
        self.layer_type = layer_type
        if layer_type == "full_attention":
            self.self_attn = Qwen3_5Attention(config, cache_layer, linear_options)
        else:
            self.linear_attn = Qwen3_5GatedDeltaNet(config, cache_layer, linear_options)
        norm_options = {
            "eps": config.rms_norm_eps,
            "device": linear_options["device"],
            "dtype": linear_options["dtype"],
        }
        self.input_layernorm = OffsetRMSNorm(config.hidden_size, **norm_options)
        self.post_attention_layernorm = OffsetRMSNorm(
            config.hidden_size, **norm_options
        )
        self.mlp = Qwen3_5MLP(config, linear_options)

    def forward(
        self,
        hidden: torch.Tensor,
        cache: GPUKVCache,
        metadata: tuple[KVCacheSlots, ...],
        positions: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        normalized = self.input_layernorm(hidden)
        if self.layer_type == "full_attention":
            attended = self.self_attn(normalized, cache, metadata, positions, cos, sin)
        else:
            attended = self.linear_attn(normalized, cache, metadata)
        hidden = hidden + attended
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class Qwen3_5GatedDeltaNet(nn.Module):
    def __init__(
        self, config: Qwen3_5TextConfig, cache_layer: int, linear_options: dict
    ) -> None:
        super().__init__()
        tp = linear_options["group"].world_size
        self.cache_layer = cache_layer
        self.num_key_heads = config.linear_num_key_heads // tp
        self.num_value_heads = config.linear_num_value_heads // tp
        self.key_head_dim = config.linear_key_head_dim
        self.value_head_dim = config.linear_value_head_dim
        self.key_width = self.num_key_heads * self.key_head_dim
        self.value_width = self.num_value_heads * self.value_head_dim
        self.conv_channels = 2 * self.key_width + self.value_width
        self.conv_history = config.linear_conv_kernel_dim - 1
        self.eps = config.rms_norm_eps
        in_features = config.hidden_size
        self.in_proj_qkv = MergedColumnParallelLinear(
            in_features,
            (self.key_width * tp, self.key_width * tp, self.value_width * tp),
            **linear_options,
        )
        self.in_proj_z = ColumnParallelLinear(
            in_features, self.value_width * tp, **linear_options
        )
        self.in_proj_b = ColumnParallelLinear(
            in_features, config.linear_num_value_heads, **linear_options
        )
        self.in_proj_a = ColumnParallelLinear(
            in_features, config.linear_num_value_heads, **linear_options
        )
        self.out_proj = RowParallelLinear(
            self.value_width * tp, in_features, **linear_options
        )
        tensor_options = {
            "device": linear_options["device"],
            "dtype": linear_options["dtype"],
        }
        self.conv1d = nn.Conv1d(
            self.conv_channels,
            self.conv_channels,
            config.linear_conv_kernel_dim,
            groups=self.conv_channels,
            bias=False,
            **tensor_options,
        )
        self.A_log = nn.Parameter(torch.empty(self.num_value_heads, **tensor_options))
        self.dt_bias = nn.Parameter(torch.empty(self.num_value_heads, **tensor_options))
        self.norm = RMSNorm(self.value_head_dim, eps=self.eps, **tensor_options)
        self.flash_chunk = None
        self.flash_recurrent = None
        if linear_options["device"].type == "cuda":
            from fla.ops.gated_delta_rule import (
                chunk_gated_delta_rule,
                fused_recurrent_gated_delta_rule,
            )

            self.flash_chunk = chunk_gated_delta_rule
            self.flash_recurrent = fused_recurrent_gated_delta_rule

    def forward(
        self,
        hidden: torch.Tensor,
        cache: GPUKVCache,
        metadata: tuple[KVCacheSlots, ...],
    ) -> torch.Tensor:
        mixed = self.in_proj_qkv(hidden)
        z = self.in_proj_z(hidden).unflatten(
            -1, (self.num_value_heads, self.value_head_dim)
        )
        a, b = self.in_proj_a(hidden), self.in_proj_b(hidden)
        beta = b.sigmoid().float()
        log_decay = -self.A_log.float().exp() * F.softplus(
            a.float() + self.dt_bias.float()
        )
        output = torch.zeros_like(z)
        for row, item in enumerate(metadata):
            tokens = slice(hidden.shape[1] - item.query_len, None)
            output[row, tokens] = self._run_request(
                mixed[row, tokens],
                beta[row, tokens],
                log_decay[row, tokens],
                cache,
                item,
            )
        gated = self.norm(output) * F.silu(z.float())
        return self.out_proj(gated.to(hidden.dtype).flatten(-2))

    def _run_request(
        self,
        mixed: torch.Tensor,
        beta: torch.Tensor,
        log_decay: torch.Tensor,
        cache: GPUKVCache,
        metadata: KVCacheSlots,
    ) -> torch.Tensor:
        block_size = cache.spec.block_size
        start = metadata.context_len - metadata.query_len
        previous_block = (
            metadata.block_ids[(start - 1) // block_size] if start else None
        )
        conv_state, recurrent = cache.linear_state(self.cache_layer, previous_block)
        conv_input = torch.cat((conv_state, mixed.transpose(0, 1)), dim=-1)
        if mixed.shape[0] == 1:
            convolved = (
                (conv_input.float() * self.conv1d.weight[:, 0].float())
                .sum(dim=-1, keepdim=True)
                .to(mixed.dtype)
            )
        else:
            convolved = self.conv1d(conv_input.unsqueeze(0)).squeeze(0)
        query, key, value = (
            F.silu(convolved)
            .transpose(0, 1)
            .split((self.key_width, self.key_width, self.value_width), dim=-1)
        )
        repetitions = self.num_value_heads // self.num_key_heads
        query = query.unflatten(-1, (self.num_key_heads, self.key_head_dim))
        key = key.unflatten(-1, (self.num_key_heads, self.key_head_dim))
        value = value.unflatten(-1, (self.num_value_heads, self.value_head_dim))
        query = query.repeat_interleave(repetitions, dim=1)
        key = key.repeat_interleave(repetitions, dim=1)

        outputs = []
        offset = 0
        while offset < metadata.query_len:
            position = start + offset
            block_index, block_offset = divmod(position, block_size)
            span_len = block_size - block_offset
            if self.flash_chunk is None:
                span_len = min(64, span_len)
            end = min(metadata.query_len, offset + span_len)
            span = slice(offset, end)
            output, recurrent = self._gated_delta_rule(
                query[span],
                key[span],
                value[span],
                log_decay[span],
                beta[span],
                recurrent,
            )
            outputs.append(output)
            if (start + end) % block_size == 0 or end == metadata.query_len:
                cache.write_linear_state(
                    self.cache_layer,
                    metadata.block_ids[block_index],
                    conv_input[:, end : end + self.conv_history],
                    recurrent,
                )
            offset = end
        return torch.cat(outputs).to(mixed.dtype)

    def _gated_delta_rule(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        log_decay: torch.Tensor,
        beta: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kernel = self.flash_recurrent if query.shape[0] == 1 else self.flash_chunk
        if kernel is not None:
            output, final_state = kernel(
                query.unsqueeze(0),
                key.unsqueeze(0),
                value.unsqueeze(0),
                g=log_decay.unsqueeze(0),
                beta=beta.unsqueeze(0),
                initial_state=state.unsqueeze(0),
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
            return output.squeeze(0), final_state.squeeze(0)

        output, final_state = self._torch_gated_delta_rule(
            query.transpose(0, 1).float(),
            key.transpose(0, 1).float(),
            value.transpose(0, 1).float(),
            log_decay.transpose(0, 1),
            beta.transpose(0, 1),
            state,
        )
        return output.transpose(0, 1), final_state

    @staticmethod
    def _torch_gated_delta_rule(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        log_decay: torch.Tensor,
        beta: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate one span with tensors in [heads, tokens, channels] layout."""
        query = query * torch.rsqrt(query.square().sum(dim=-1, keepdim=True) + 1e-6)
        query = query * query.shape[-1] ** -0.5
        key = key * torch.rsqrt(key.square().sum(dim=-1, keepdim=True) + 1e-6)

        if query.shape[-2] == 1:
            state = state * log_decay.exp().unsqueeze(-1)
            delta = (value - key @ state) * beta.unsqueeze(-1)
            state = state + key.transpose(-1, -2) @ delta
            return query @ state, state

        # Each token's update depends on earlier updates through decayed key
        # similarities. A unit lower-triangular solve evaluates all updates together.
        cumulative = log_decay.cumsum(dim=-1)
        decay = (
            (cumulative.unsqueeze(-1) - cumulative.unsqueeze(-2)).tril().exp().tril()
        )
        beta_key = beta.unsqueeze(-1) * key
        initial_decay = cumulative.exp().unsqueeze(-1)
        dependencies = ((beta_key @ key.transpose(-1, -2)) * decay).tril(-1)
        residual = value * beta.unsqueeze(-1) - (beta_key * initial_decay) @ state
        updates = torch.linalg.solve_triangular(
            dependencies, residual, upper=False, unitriangular=True
        )
        output = (query * initial_decay) @ state
        output = output + ((query @ key.transpose(-1, -2)) * decay) @ updates
        end_decay = (cumulative[..., -1:] - cumulative).exp().unsqueeze(-1)
        state = (
            state * initial_decay[..., -1:, :]
            + (key * end_decay).transpose(-1, -2) @ updates
        )
        return output, state


class Qwen3_5Attention(nn.Module):
    def __init__(
        self, config: Qwen3_5TextConfig, cache_layer: int, linear_options: dict
    ) -> None:
        super().__init__()
        tp = linear_options["group"].world_size
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads // tp
        self.num_kv_heads = config.num_key_value_heads // tp
        self.rotary_dim = config.rotary_dim
        self.cache_layer = cache_layer
        q_width = config.num_attention_heads * self.head_dim
        kv_width = config.num_key_value_heads * self.head_dim
        self.q_proj = ColumnParallelLinear(
            config.hidden_size, 2 * q_width, **linear_options
        )
        self.k_proj = ColumnParallelLinear(
            config.hidden_size, kv_width, **linear_options
        )
        self.v_proj = ColumnParallelLinear(
            config.hidden_size, kv_width, **linear_options
        )
        self.o_proj = RowParallelLinear(q_width, config.hidden_size, **linear_options)
        norm_options = {
            "eps": config.rms_norm_eps,
            "device": linear_options["device"],
            "dtype": linear_options["dtype"],
        }
        self.q_norm = OffsetRMSNorm(self.head_dim, **norm_options)
        self.k_norm = OffsetRMSNorm(self.head_dim, **norm_options)
        self.flash_attention = None
        if linear_options["device"].type == "cuda" and linear_options["dtype"] in (
            torch.float16,
            torch.bfloat16,
        ):
            from flash_attn import flash_attn_varlen_func

            self.flash_attention = flash_attn_varlen_func

    def _apply_rope(
        self, tensor: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        rotated, remainder = tensor.split(
            (self.rotary_dim, self.head_dim - self.rotary_dim), dim=-1
        )
        rotated = rotated * cos + rotate_half(rotated) * sin
        return torch.cat((rotated, remainder), dim=-1)

    def forward(
        self,
        hidden: torch.Tensor,
        cache: GPUKVCache,
        metadata: tuple[KVCacheSlots, ...],
        positions: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        query, key, value, gate = self._project(hidden, cos, sin)
        cache.write(self.cache_layer, metadata, key, value)
        if self.flash_attention is not None:
            key, value = cache.gather_packed(self.cache_layer, metadata)
            attended = self._flash_attention(query, key, value, metadata)
        else:
            key, value = cache.gather(self.cache_layer, metadata)
            attended = self._torch_attention(query, key, value, positions)
        return self.o_proj((attended * gate.sigmoid()).flatten(-2))

    def _flash_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        metadata: tuple[KVCacheSlots, ...],
    ) -> torch.Tensor:
        query_lengths = [item.query_len for item in metadata]
        context_lengths = [item.context_len for item in metadata]
        packed_query = torch.cat(
            [query[row, -length:] for row, length in enumerate(query_lengths)]
        )
        cu_query = torch.tensor(
            list(accumulate(query_lengths, initial=0)),
            dtype=torch.int32,
            device=query.device,
        )
        cu_context = torch.tensor(
            list(accumulate(context_lengths, initial=0)),
            dtype=torch.int32,
            device=query.device,
        )
        # The query chunk is the suffix of its KV context, matching FlashAttention's
        # bottom-right causal alignment.
        packed_output = self.flash_attention(
            packed_query,
            key,
            value,
            cu_query,
            cu_context,
            max(query_lengths),
            max(context_lengths),
            causal=True,
        )
        output = torch.zeros_like(query)
        for row, chunk in enumerate(packed_output.split(query_lengths)):
            output[row, -chunk.shape[0] :] = chunk
        return output

    def _torch_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        repetitions = self.num_heads // self.num_kv_heads
        key = key.transpose(1, 2).repeat_interleave(repetitions, dim=1)
        value = value.transpose(1, 2).repeat_interleave(repetitions, dim=1)
        # Real query positions are always less than their request's context length.
        # Padded query rows are discarded by the cache writer and output head.
        key_positions = torch.arange(key.shape[-2], device=query.device)
        allowed = key_positions <= positions[:, None, :, None]
        output = F.scaled_dot_product_attention(
            query.transpose(1, 2), key, value, attn_mask=allowed, dropout_p=0.0
        )
        return output.transpose(1, 2)

    def _project(
        self, hidden: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        shape = hidden.shape[:-1]
        query, key, value = (
            self.q_proj(hidden),
            self.k_proj(hidden),
            self.v_proj(hidden),
        )
        query, gate = query.view(*shape, self.num_heads, 2 * self.head_dim).chunk(
            2, dim=-1
        )
        kv_shape = (*shape, self.num_kv_heads, self.head_dim)
        key = key.view(kv_shape)
        query = self._apply_rope(self.q_norm(query), cos, sin)
        key = self._apply_rope(self.k_norm(key), cos, sin)
        return query, key, value.view(kv_shape), gate


class Qwen3_5MLP(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig, linear_options: dict) -> None:
        super().__init__()
        hidden, intermediate = config.hidden_size, config.intermediate_size
        self.gate_proj = ColumnParallelLinear(hidden, intermediate, **linear_options)
        self.up_proj = ColumnParallelLinear(hidden, intermediate, **linear_options)
        self.down_proj = RowParallelLinear(intermediate, hidden, **linear_options)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        activated = F.silu(self.gate_proj(hidden)) * self.up_proj(hidden)
        return self.down_proj(activated)


class OffsetRMSNorm(RMSNorm):
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        values = hidden.float()
        scale = torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + self.eps)
        return (values * scale * (1 + self.weight.float())).to(hidden.dtype)


class Qwen3_5ForCausalLM(nn.Module):
    def __init__(
        self,
        config: Qwen3_5TextConfig,
        context: ParallelContext,
        dtype: torch.dtype,
        load_config: ModelLoadConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.context = context
        self.load_config = load_config
        self.model = Qwen3_5Model(config, context, dtype)
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            group=context.group(ParallelGroupKind.TENSOR),
            device=context.device,
            dtype=dtype,
        )
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.load_report: WeightLoadReport | None = None
        self.kv_cache: GPUKVCache | None = None

    def load_model(self) -> None:
        self.load_report = load_qwen3_5_weights(self)

    def get_kv_cache_spec(self, block_size: int) -> KVCacheSpec:
        config = self.config
        tp = self.context.group(ParallelGroupKind.TENSOR).world_size
        linear = None
        if self.model.num_linear_layers:
            heads = config.linear_num_value_heads // tp
            linear = LinearAttentionCacheSpec(
                self.model.num_linear_layers,
                2 * (config.linear_num_key_heads // tp) * config.linear_key_head_dim
                + heads * config.linear_value_head_dim,
                config.linear_conv_kernel_dim - 1,
                heads,
                config.linear_key_head_dim,
                config.linear_value_head_dim,
            )
        return KVCacheSpec(
            self.model.num_full_layers,
            block_size,
            config.num_key_value_heads // tp,
            config.head_dim,
            self.model.embed_tokens.weight.element_size(),
            linear,
        )

    def set_kv_cache(self, cache: GPUKVCache | None) -> None:
        self.kv_cache = cache

    def forward(self, inputs: ModelInput) -> ModelOutput:
        if (
            inputs.input_ids is None
            or not inputs.kv_cache_slots
            or self.kv_cache is None
        ):
            raise ValueError(
                "Qwen3.5 requires token IDs, KV metadata and an initialized cache"
            )
        hidden = self.model(
            inputs.input_ids,
            inputs.position_ids,
            self.kv_cache,
            inputs.kv_cache_slots,
        )
        return ModelOutput(self.lm_head(hidden))


def create_qwen3_5_runner(
    load_config: ModelLoadConfig,
    context: ParallelContext,
    *,
    pad_token_id: int,
    is_output_rank: bool,
) -> GPUModelRunner:
    model_config = Qwen3_5TextConfig.from_hf_config(load_config.hf_config)
    model_config.validate_tensor_parallel(
        context.group(ParallelGroupKind.TENSOR).world_size
    )
    dtype_name = (
        model_config.torch_dtype if load_config.dtype == "auto" else load_config.dtype
    )
    dtype = {
        "float32": torch.float32,
        "float": torch.float32,
        "float16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(dtype_name)
    if dtype is None:
        raise ValueError(f"unsupported model dtype: {dtype_name}")
    model = Qwen3_5ForCausalLM(model_config, context, dtype, load_config)
    return GPUModelRunner(
        context, model, pad_token_id=pad_token_id, is_output_rank=is_output_rank
    )
