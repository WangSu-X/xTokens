"""Text inference for the hybrid Qwen3.5 decoder and Qwen3.8-27B."""

from __future__ import annotations

from itertools import zip_longest

import torch
import torch.nn.functional as F
from torch import nn

from x_tokens.distributed import ParallelContext, ParallelGroupKind
from x_tokens.modeling.config import ModelLoadConfig
from x_tokens.modeling.kv_cache import (
    FlashAttentionMetadata,
    FullAttentionCacheSpec,
    KVCacheSpec,
    LinearAttentionCacheSpec,
    get_forward_context,
)
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
from x_tokens.modeling.protocol import ModelOutput
from x_tokens.modeling.qwen3_5_config import Qwen3_5TextConfig
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
        self.layers = nn.ModuleList(
            Qwen3_5DecoderLayer(
                config, layer_type, f"model.layers.{index}", linear_options
            )
            for index, layer_type in enumerate(config.layer_types)
        )
        self.rotary_emb = RotaryEmbedding(
            config.rotary_dim, config.rope_theta, device=context.device
        )

    def forward(
        self, input_ids: torch.Tensor, position_ids: torch.Tensor
    ) -> torch.Tensor:
        hidden = self.embed_tokens(input_ids)
        ctx = get_forward_context()
        cos, sin = self.rotary_emb(position_ids, hidden.dtype)
        ctx.rope = cos.unsqueeze(2), sin.unsqueeze(2)
        for layer in self.layers:
            hidden = layer(hidden)
        return self.norm(hidden[:, -1])


class Qwen3_5DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen3_5TextConfig,
        layer_type: str,
        layer_name: str,
        linear_options: dict,
    ) -> None:
        super().__init__()
        self.layer_type = layer_type
        if layer_type == "full_attention":
            self.self_attn = Qwen3_5Attention(
                config, f"{layer_name}.self_attn", linear_options
            )
        else:
            self.linear_attn = Qwen3_5GatedDeltaNet(
                config, f"{layer_name}.linear_attn", linear_options
            )
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

    def get_kv_cache_spec(self, block_size: int) -> tuple[str, KVCacheSpec]:
        if self.layer_type == "full_attention":
            return self.self_attn.cache_name, self.self_attn.get_kv_cache_spec(
                block_size
            )
        return self.linear_attn.cache_name, self.linear_attn.cache_spec

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        normalized = self.input_layernorm(hidden)
        if self.layer_type == "full_attention":
            attended = self.self_attn(normalized)
        else:
            attended = self.linear_attn(normalized)
        hidden = hidden + attended
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class Qwen3_5GatedDeltaNet(nn.Module):
    def __init__(
        self, config: Qwen3_5TextConfig, cache_name: str, linear_options: dict
    ) -> None:
        super().__init__()
        tp = linear_options["group"].world_size
        self.cache_name = cache_name
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

    @property
    def cache_spec(self) -> LinearAttentionCacheSpec:
        return LinearAttentionCacheSpec(
            self.conv_channels,
            self.conv_history,
            self.num_value_heads,
            self.key_head_dim,
            self.value_head_dim,
            self.conv1d.weight.dtype,
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        ctx = get_forward_context()
        metadata = ctx.linear_attention
        assert metadata is not None
        conv_cache, recurrent_cache = ctx.kv_cache_dict[self.cache_name]
        mixed = self.in_proj_qkv(hidden)
        z = self.in_proj_z(hidden).unflatten(
            -1, (self.num_value_heads, self.value_head_dim)
        )
        beta = self.in_proj_b(hidden).sigmoid().float()
        log_decay = -self.A_log.float().exp() * F.softplus(
            self.in_proj_a(hidden).float() + self.dt_bias.float()
        )
        output = torch.zeros_like(z)

        prefill_token_indices = metadata.prefill_token_indices
        prefill_inputs, prefill_convolved = [], []
        for row, length, read_index, spans in zip(
            metadata.prefill_rows,
            metadata.prefill_query_lens,
            metadata.prefill_read_indices,
            metadata.prefill_spans,
            strict=True,
        ):
            if read_index is None:
                history = conv_cache.new_zeros(conv_cache.shape[1:])
            else:
                history = conv_cache[read_index]
            conv_input = torch.cat(
                (history, mixed[row, -length:].transpose(0, 1)), dim=-1
            )
            convolved = self.conv1d(conv_input.unsqueeze(0)).transpose(1, 2)
            if prefill_token_indices is not None:
                prefill_inputs.append(conv_input)
                prefill_convolved.append(convolved)
                continue

            state = (
                recurrent_cache.new_zeros(recurrent_cache.shape[1:])
                if read_index is None
                else recurrent_cache[read_index]
            )
            query, key, value = self.split_qkv(convolved)
            row_beta = beta[row : row + 1, -length:]
            row_decay = log_decay[row : row + 1, -length:]
            row_output = output[row, -length:]

            start = 0
            for end, write_index in spans:
                span = slice(start, end)
                if self.flash_chunk is None:
                    attended, state = self.torch_gated_delta_rule(
                        query[0, span].transpose(0, 1).float(),
                        key[0, span].transpose(0, 1).float(),
                        value[0, span].transpose(0, 1).float(),
                        row_decay[0, span].transpose(0, 1),
                        row_beta[0, span].transpose(0, 1),
                        state,
                    )
                    row_output[span] = attended.transpose(0, 1)
                else:
                    attended, final_state = self.flash_chunk(
                        query[:, span],
                        key[:, span],
                        value[:, span],
                        g=row_decay[:, span],
                        beta=row_beta[:, span],
                        initial_state=state.unsqueeze(0),
                        output_final_state=True,
                        use_qk_l2norm_in_kernel=True,
                    )
                    state = final_state[0]
                    row_output[span] = attended[0]
                if write_index is not None:
                    conv_cache[write_index].copy_(
                        conv_input[:, end : end + self.conv_history]
                    )
                    recurrent_cache[write_index].copy_(state)
                start = end

        if prefill_token_indices is not None:
            query, key, value = self.split_qkv(torch.cat(prefill_convolved, dim=1))
            prefill_beta = beta.flatten(0, 1).index_select(0, prefill_token_indices)
            prefill_decay = log_decay.flatten(0, 1).index_select(
                0, prefill_token_indices
            )
            prefill_output = output.new_empty(
                (
                    prefill_token_indices.numel(),
                    self.num_value_heads,
                    self.value_head_dim,
                )
            )
            assert self.flash_chunk is not None
            for round_number, (indices, cu_seqlens, reads, writes, spans) in enumerate(
                zip(
                    metadata.prefill_round_token_indices,
                    metadata.prefill_round_cu_seqlens,
                    metadata.prefill_round_read_indices,
                    metadata.prefill_round_write_indices,
                    zip_longest(*metadata.prefill_spans),
                    strict=True,
                )
            ):
                if round_number == 0 and None in metadata.prefill_read_indices:
                    state = recurrent_cache[reads]
                    state.masked_fill_(reads[:, None, None, None] < 0, 0)
                else:
                    state = recurrent_cache.index_select(0, reads)
                batch_size = reads.numel() if cu_seqlens is None else 1
                qk_shape = (batch_size, -1, self.num_value_heads, self.key_head_dim)
                value_shape = (
                    batch_size,
                    -1,
                    self.num_value_heads,
                    self.value_head_dim,
                )
                gate_shape = (batch_size, -1, self.num_value_heads)
                attended, state = self.flash_chunk(
                    query.index_select(1, indices).view(qk_shape),
                    key.index_select(1, indices).view(qk_shape),
                    value.index_select(1, indices).view(value_shape),
                    g=prefill_decay.index_select(0, indices).view(gate_shape),
                    beta=prefill_beta.index_select(0, indices).view(gate_shape),
                    initial_state=state,
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=True,
                    cu_seqlens=cu_seqlens,
                )
                prefill_output.index_copy_(0, indices, attended.flatten(0, 1))
                recurrent_cache.index_copy_(0, writes, state)
                for conv_input, span in zip(prefill_inputs, spans, strict=True):
                    if span is not None:
                        end, write_index = span
                        conv_cache[write_index].copy_(
                            conv_input[:, end : end + self.conv_history]
                        )
            output.flatten(0, 1).index_copy_(0, prefill_token_indices, prefill_output)

        rows = metadata.decode_rows
        if rows.numel():
            history = conv_cache.index_select(0, metadata.decode_read_indices)
            state = recurrent_cache.index_select(0, metadata.decode_read_indices)
            conv_input = torch.cat((history, mixed[rows, -1].unsqueeze(-1)), dim=-1)
            convolved = (
                (conv_input.float() * self.conv1d.weight[:, 0].float())
                .sum(dim=-1)
                .to(mixed.dtype)
                .unsqueeze(1)
            )
            query, key, value = self.split_qkv(convolved)
            decode_beta = beta[rows, -1].unsqueeze(1)
            decode_decay = log_decay[rows, -1].unsqueeze(1)
            if self.flash_recurrent is None:
                attended, state = self.torch_gated_delta_rule(
                    query.transpose(1, 2).float(),
                    key.transpose(1, 2).float(),
                    value.transpose(1, 2).float(),
                    decode_decay.transpose(1, 2),
                    decode_beta.transpose(1, 2),
                    state,
                )
                output[rows, -1] = attended.squeeze(2).to(output.dtype)
            else:
                attended, state = self.flash_recurrent(
                    query,
                    key,
                    value,
                    g=decode_decay,
                    beta=decode_beta,
                    initial_state=state,
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=True,
                )
                output[rows, -1] = attended[:, 0]
            conv_cache.index_copy_(
                0, metadata.decode_write_indices, conv_input[..., 1:]
            )
            recurrent_cache.index_copy_(0, metadata.decode_write_indices, state)

        gated = self.norm(output) * F.silu(z.float())
        return self.out_proj(gated.to(hidden.dtype).flatten(-2))

    def split_qkv(
        self, convolved: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query, key, value = F.silu(convolved).split(
            (self.key_width, self.key_width, self.value_width), dim=-1
        )
        repetitions = self.num_value_heads // self.num_key_heads
        query = query.unflatten(-1, (self.num_key_heads, self.key_head_dim))
        key = key.unflatten(-1, (self.num_key_heads, self.key_head_dim))
        value = value.unflatten(-1, (self.num_value_heads, self.value_head_dim))
        return (
            query.repeat_interleave(repetitions, dim=-2),
            key.repeat_interleave(repetitions, dim=-2),
            value,
        )

    @staticmethod
    def torch_gated_delta_rule(
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
        self, config: Qwen3_5TextConfig, cache_name: str, linear_options: dict
    ) -> None:
        super().__init__()
        tp = linear_options["group"].world_size
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads // tp
        self.num_kv_heads = config.num_key_value_heads // tp
        self.rotary_dim = config.rotary_dim
        self.cache_name = cache_name
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

    def get_kv_cache_spec(self, block_size: int) -> FullAttentionCacheSpec:
        return FullAttentionCacheSpec(
            block_size, self.num_kv_heads, self.head_dim, self.k_proj.weight.dtype
        )

    def apply_rope(
        self, tensor: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        rotated, remainder = tensor.split(
            (self.rotary_dim, self.head_dim - self.rotary_dim), dim=-1
        )
        rotated = rotated * cos + rotate_half(rotated) * sin
        return torch.cat((rotated, remainder), dim=-1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        ctx = get_forward_context()
        metadata = ctx.full_attention
        assert metadata is not None
        cache = ctx.kv_cache_dict[self.cache_name]
        assert isinstance(cache, torch.Tensor)
        cos, sin = ctx.rope
        shape = hidden.shape[:-1]
        query, gate = (
            self.q_proj(hidden)
            .view(*shape, self.num_heads, 2 * self.head_dim)
            .chunk(2, dim=-1)
        )
        kv_shape = (*shape, self.num_kv_heads, self.head_dim)
        key = self.k_proj(hidden).view(kv_shape)
        value = self.v_proj(hidden).view(kv_shape)
        query = self.apply_rope(self.q_norm(query), cos, sin)
        key = self.apply_rope(self.k_norm(key), cos, sin)

        cache = cache.flatten(1, 2)
        key_cache, value_cache = cache.unbind(0)
        query_indices = metadata.query_indices
        key_cache.index_copy_(
            0, metadata.slot_mapping, key.flatten(0, 1).index_select(0, query_indices)
        )
        value_cache.index_copy_(
            0, metadata.slot_mapping, value.flatten(0, 1).index_select(0, query_indices)
        )
        key, value = cache.index_select(1, metadata.context_slots).unbind(0)

        if self.flash_attention is not None:
            packed_query = query.flatten(0, 1).index_select(0, query_indices)
            packed_output = self.flash_attention(
                packed_query,
                key,
                value,
                metadata.query_start_loc,
                metadata.seq_start_loc,
                metadata.max_query_len,
                metadata.max_seq_len,
                causal=True,
            )
            attended = query.new_zeros(query.shape)
            attended.flatten(0, 1).index_copy_(0, query_indices, packed_output)
        else:
            attended = self.torch_attention(query, key, value, metadata)
        return self.o_proj((attended * gate.sigmoid()).flatten(-2))

    def torch_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        metadata: FlashAttentionMetadata,
    ) -> torch.Tensor:
        shape = (query.shape[0], metadata.max_seq_len, self.num_kv_heads, self.head_dim)
        padded_key = key.new_zeros(shape)
        padded_value = value.new_zeros(shape)
        offset = 0
        for row, length in enumerate(metadata.seq_lens):
            end = offset + length
            padded_key[row, :length] = key[offset:end]
            padded_value[row, :length] = value[offset:end]
            offset = end

        repetitions = self.num_heads // self.num_kv_heads
        padded_key = padded_key.transpose(1, 2).repeat_interleave(repetitions, dim=1)
        padded_value = padded_value.transpose(1, 2).repeat_interleave(
            repetitions, dim=1
        )
        return F.scaled_dot_product_attention(
            query.transpose(1, 2),
            padded_key,
            padded_value,
            attn_mask=metadata.causal_mask,
            dropout_p=0.0,
        ).transpose(1, 2)


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

    def load_model(self) -> None:
        self.load_report = load_qwen3_5_weights(self)

    def get_kv_cache_specs(self, block_size: int) -> dict[str, KVCacheSpec]:
        return dict(layer.get_kv_cache_spec(block_size) for layer in self.model.layers)

    def forward(
        self, input_ids: torch.Tensor, position_ids: torch.Tensor
    ) -> ModelOutput:
        hidden = self.model(input_ids, position_ids)
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
