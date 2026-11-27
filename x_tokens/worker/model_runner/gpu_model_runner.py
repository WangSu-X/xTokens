"""Rank-local model execution, input preparation, and sampling."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import accumulate, zip_longest
from typing import Any

from torch import Tensor

from x_tokens.config import KVCacheConfig
from x_tokens.distributed import ParallelContext
from x_tokens.logger import init_logger
from x_tokens.modeling import ModelOutput, XTokensModel
from x_tokens.modeling.kv_cache import (
    FlashAttentionMetadata,
    FlashLinearAttentionMetadata,
    ForwardContext,
    FullAttentionCacheSpec,
    LayerCache,
    LinearAttentionCacheSpec,
    set_forward_context,
)
from x_tokens.worker.kv_cache import allocate_kv_cache

from .protocol import (
    ModelRunner,
    ModelRunnerOutput,
    ModelRunnerRequest,
    SamplingMetadata,
)

logger = init_logger(__name__)
MAX_CPU_LINEAR_SPAN = 64


@dataclass(slots=True)
class ExecuteModelState:
    logits: Any
    sampling_metadata: dict[str, SamplingMetadata]


class GPUModelRunner(ModelRunner):
    def __init__(
        self,
        context: ParallelContext,
        model: XTokensModel,
        *,
        pad_token_id: int,
        is_output_rank: bool,
        torch_module: Any | None = None,
    ) -> None:
        if torch_module is None:
            import torch

            torch_module = torch
        self.context = context
        self.model = model
        self.pad_token_id = pad_token_id
        self.is_output_rank = is_output_rank
        self.torch_module = torch_module
        self.execute_model_state: ExecuteModelState | None = None
        self.kv_cache_dict: dict[str, LayerCache] = {}
        self.block_size: int | None = None
        self.has_full_attention = False
        self.has_linear_attention = False
        self.has_torch_attention = False

    def load_model(self) -> None:
        load_model = getattr(self.model, "load_model", None)
        if callable(load_model):
            load_model()
        report = getattr(self.model, "load_report", None)
        if report is not None:
            logger.info(
                "model weights loaded: rank=%d/%d source_tensors=%d "
                "parameters=%d local_bytes=%d",
                report.rank,
                report.world_size,
                report.loaded_source_tensors,
                report.loaded_parameters,
                report.local_parameter_bytes,
            )
        to = getattr(self.model, "to", None)
        if callable(to):
            to(self.context.device)
        eval_model = getattr(self.model, "eval", None)
        if callable(eval_model):
            eval_model()

    def get_kv_cache_capacity(self, config: KVCacheConfig) -> int:
        if config.num_gpu_blocks is not None:
            return config.num_gpu_blocks
        spec_per_layer = self.model.get_kv_cache_specs(config.block_size)
        bytes_per_block = sum(spec.bytes_per_block for spec in spec_per_layer.values())
        if config.memory_bytes is not None:
            return config.memory_bytes // bytes_per_block
        if self.context.device.type != "cuda":
            raise ValueError("CPU KV cache requires num_gpu_blocks or memory_bytes")
        free_bytes, total_bytes = self.torch_module.cuda.mem_get_info(
            self.context.device
        )
        reserved_headroom = int(total_bytes * (1 - config.gpu_memory_utilization))
        return max(0, free_bytes - reserved_headroom) // bytes_per_block

    def initialize_kv_cache(self, num_blocks: int, block_size: int) -> None:
        if self.block_size is not None:
            raise RuntimeError("KV cache is already initialized")
        spec_per_layer = self.model.get_kv_cache_specs(block_size)
        self.kv_cache_dict = allocate_kv_cache(
            spec_per_layer, num_blocks, self.context.device
        )
        self.block_size = block_size
        self.has_full_attention = any(
            isinstance(spec, FullAttentionCacheSpec) for spec in spec_per_layer.values()
        )
        self.has_linear_attention = any(
            isinstance(spec, LinearAttentionCacheSpec)
            for spec in spec_per_layer.values()
        )
        self.has_torch_attention = any(
            isinstance(spec, FullAttentionCacheSpec)
            and (
                self.context.device.type != "cuda"
                or spec.dtype
                not in (self.torch_module.float16, self.torch_module.bfloat16)
            )
            for spec in spec_per_layer.values()
        )
        logger.info(
            "KV cache initialized: rank=%d blocks=%d block_size=%d bytes=%d",
            self.context.coordinates.global_rank,
            num_blocks,
            block_size,
            num_blocks * sum(spec.bytes_per_block for spec in spec_per_layer.values()),
        )

    def execute_model(self, requests: tuple[ModelRunnerRequest, ...]) -> None:
        self.execute_model_state = None
        if not requests:
            return
        with self.torch_module.inference_mode():
            input_ids, position_ids, context = self.prepare_model_input(requests)
            with set_forward_context(context):
                output = self.model(input_ids, position_ids)
        if not self.is_output_rank:
            return
        sample_rows = []
        sampling_metadata = {}
        for row, request in enumerate(requests):
            if request.sampling_metadata is not None:
                sample_rows.append(row)
                sampling_metadata[request.request_id] = request.sampling_metadata
        if not sample_rows:
            return

        logits = output.logits if isinstance(output, ModelOutput) else output
        if logits is None:
            raise RuntimeError("output rank model did not produce logits")
        if logits.ndim == 3:
            logits = logits[:, -1, :]
        if logits.ndim != 2 or logits.shape[0] not in (
            len(requests),
            len(sample_rows),
        ):
            raise ValueError(
                "model logits must have shape [batch, vocab] or "
                "[batch, sequence, vocab]"
            )
        if logits.shape[0] != len(sample_rows):
            logits = logits[sample_rows]
        self.execute_model_state = ExecuteModelState(logits, sampling_metadata)

    def sample_tokens(self) -> ModelRunnerOutput:
        """Sample tokens from logits using cached sampling metadata.

        Returns:
            ModelRunnerOutput mapping request_id to sampled token_id
        """
        state = self.execute_model_state
        self.execute_model_state = None
        if state is None:
            return ModelRunnerOutput()

        # Validate sampling parameters upfront
        for request_id, metadata in state.sampling_metadata.items():
            if metadata.temperature < 0:
                raise ValueError(
                    f"request {request_id}: temperature must be non-negative, "
                    f"got {metadata.temperature}"
                )
            if not (0 < metadata.top_p <= 1):
                raise ValueError(
                    f"request {request_id}: top_p must be in (0, 1], got {metadata.top_p}"
                )
            if metadata.top_k is not None and metadata.top_k < 1:
                raise ValueError(
                    f"request {request_id}: top_k must be positive, got {metadata.top_k}"
                )

        # Fast path: all requests use greedy (temperature=0)
        if all(
            metadata.temperature == 0 for metadata in state.sampling_metadata.values()
        ):
            token_ids = state.logits.argmax(dim=-1).tolist()
        else:
            # Sample each request individually
            token_ids = [
                self._sample(state.logits[index], metadata)
                for index, metadata in enumerate(state.sampling_metadata.values())
            ]

        return ModelRunnerOutput(
            dict(zip(state.sampling_metadata, token_ids, strict=True))
        )

    def shutdown(self) -> None:
        self.execute_model_state = None
        self.kv_cache_dict.clear()
        self.block_size = None

    def prepare_model_input(
        self, requests: tuple[ModelRunnerRequest, ...]
    ) -> tuple[Tensor, Tensor, ForwardContext]:
        block_size = self.block_size
        if block_size is None:
            raise RuntimeError("KV cache is not initialized")

        torch = self.torch_module
        device = self.context.device
        query_lengths = tuple(len(request.scheduled_token_ids) for request in requests)
        max_query_len = max(query_lengths)
        token_ids, positions = [], []
        query_indices, slot_mapping, context_slots, seq_lens = [], [], [], []
        prefill_rows, prefill_query_lens, prefill_reads, prefill_spans = [], [], [], []
        decode_rows, decode_reads, decode_writes = [], [], []
        span_limit = (
            block_size
            if device.type == "cuda"
            else min(block_size, MAX_CPU_LINEAR_SPAN)
        )

        for row, (request, query_len) in enumerate(
            zip(requests, query_lengths, strict=True)
        ):
            start = request.num_computed_tokens
            end = start + query_len
            padding = max_query_len - query_len
            block_ids = request.block_ids
            token_ids.append(
                (self.pad_token_id,) * padding + request.scheduled_token_ids
            )
            positions.append([0] * padding + list(range(start, end)))

            if self.has_full_attention:
                seq_lens.append(end)
                query_indices.extend(
                    range(row * max_query_len + padding, (row + 1) * max_query_len)
                )
                for index, block_id in enumerate(block_ids):
                    slot = block_id * block_size
                    length = min(block_size, end - index * block_size)
                    context_slots.extend(range(slot, slot + length))
                slot_mapping.extend(context_slots[-query_len:])

            if self.has_linear_attention:
                read_index = block_ids[(start - 1) // block_size] if start else None
                if request.is_prefill:
                    spans = []
                    position = start
                    while position < end:
                        block, offset = divmod(position, block_size)
                        position += min(end - position, block_size - offset, span_limit)
                        write_index = (
                            block_ids[block]
                            if position == end or position % block_size == 0
                            else None
                        )
                        spans.append((position - start, write_index))
                    prefill_rows.append(row)
                    prefill_query_lens.append(query_len)
                    prefill_reads.append(read_index)
                    prefill_spans.append(tuple(spans))
                else:
                    decode_rows.append(row)
                    decode_reads.append(read_index)
                    decode_writes.append(block_ids[start // block_size])

        position_ids = torch.tensor(positions, device=device, dtype=torch.long)
        full_attention = None
        if self.has_full_attention:
            max_seq_len = max(seq_lens)
            causal_mask = None
            if self.has_torch_attention:
                key_positions = torch.arange(max_seq_len, device=device)
                causal_mask = key_positions <= position_ids[:, None, :, None]
            full_attention = FlashAttentionMetadata(
                seq_lens=tuple(seq_lens),
                max_query_len=max_query_len,
                max_seq_len=max_seq_len,
                query_start_loc=torch.tensor(
                    list(accumulate(query_lengths, initial=0)),
                    device=device,
                    dtype=torch.int32,
                ),
                seq_start_loc=torch.tensor(
                    list(accumulate(seq_lens, initial=0)),
                    device=device,
                    dtype=torch.int32,
                ),
                query_indices=torch.tensor(
                    query_indices, device=device, dtype=torch.long
                ),
                slot_mapping=torch.tensor(
                    slot_mapping, device=device, dtype=torch.long
                ),
                context_slots=torch.tensor(
                    context_slots, device=device, dtype=torch.long
                ),
                causal_mask=causal_mask,
            )
        linear_attention = None
        if self.has_linear_attention:
            prefill_token_indices = None
            round_token_indices, round_cu_seqlens = [], []
            round_reads, round_writes = [], []
            if device.type == "cuda" and len(prefill_rows) > 1:
                packed_indices = []
                for row, length in zip(prefill_rows, prefill_query_lens, strict=True):
                    end = (row + 1) * max_query_len
                    packed_indices.extend(range(end - length, end))
                prefill_token_indices = torch.tensor(
                    packed_indices, device=device, dtype=torch.long
                )
                offsets = list(accumulate(prefill_query_lens, initial=0))
                span_starts = [0] * len(prefill_rows)
                read_indices = [
                    -1 if index is None else index for index in prefill_reads
                ]
                for spans in zip_longest(*prefill_spans):
                    indices, lengths, reads, writes = [], [], [], []
                    for index, span in enumerate(spans):
                        if span is None:
                            continue
                        end, write_index = span
                        assert write_index is not None
                        start = span_starts[index]
                        offset = offsets[index]
                        indices.extend(range(offset + start, offset + end))
                        lengths.append(end - start)
                        reads.append(read_indices[index])
                        writes.append(write_index)
                        span_starts[index] = end
                        read_indices[index] = write_index
                    round_token_indices.append(
                        torch.tensor(indices, device=device, dtype=torch.long)
                    )
                    round_cu_seqlens.append(
                        None
                        if len(set(lengths)) == 1
                        else torch.tensor(
                            list(accumulate(lengths, initial=0)),
                            device=device,
                            dtype=torch.long,
                        )
                    )
                    round_reads.append(
                        torch.tensor(reads, device=device, dtype=torch.long)
                    )
                    round_writes.append(
                        torch.tensor(writes, device=device, dtype=torch.long)
                    )
            linear_attention = FlashLinearAttentionMetadata(
                prefill_rows=tuple(prefill_rows),
                prefill_query_lens=tuple(prefill_query_lens),
                prefill_read_indices=tuple(prefill_reads),
                prefill_spans=tuple(prefill_spans),
                decode_rows=torch.tensor(decode_rows, device=device, dtype=torch.long),
                decode_read_indices=torch.tensor(
                    decode_reads, device=device, dtype=torch.long
                ),
                decode_write_indices=torch.tensor(
                    decode_writes, device=device, dtype=torch.long
                ),
                prefill_token_indices=prefill_token_indices,
                prefill_round_token_indices=tuple(round_token_indices),
                prefill_round_cu_seqlens=tuple(round_cu_seqlens),
                prefill_round_read_indices=tuple(round_reads),
                prefill_round_write_indices=tuple(round_writes),
            )
        return (
            torch.tensor(token_ids, device=device, dtype=torch.long),
            position_ids,
            ForwardContext(self.kv_cache_dict, full_attention, linear_attention),
        )

    def _sample(self, logits: Any, metadata: SamplingMetadata) -> int:
        """Sample a token using temperature, top-k, and top-p filtering.

        Filtering order:
        1. Temperature scaling
        2. Top-K: keep only the k largest logits
        3. Top-P: keep tokens with cumulative probability <= p
        """
        if metadata.temperature == 0:
            return int(self.torch_module.argmax(logits).item())

        # Validate sampling parameters
        if metadata.temperature < 0:
            raise ValueError(
                f"temperature must be non-negative, got {metadata.temperature}"
            )
        if not (0 < metadata.top_p <= 1):
            raise ValueError(f"top_p must be in (0, 1], got {metadata.top_p}")
        if metadata.top_k is not None and metadata.top_k < 1:
            raise ValueError(f"top_k must be positive, got {metadata.top_k}")

        scaled = logits / metadata.temperature

        # Top-K filtering: keep only the k largest values
        if metadata.top_k is not None and metadata.top_k > 0:
            top_k = min(metadata.top_k, scaled.shape[-1])
            _, top_indices = self.torch_module.topk(scaled, top_k)
            # Create mask of top-k indices
            mask = self.torch_module.zeros_like(scaled, dtype=self.torch_module.bool)
            mask.scatter_(0, top_indices, True)
            scaled = scaled.masked_fill(~mask, float("-inf"))

        # Top-P filtering: keep tokens with cumsum(probs) <= p
        if metadata.top_p < 1.0:
            sorted_logits, sorted_indices = self.torch_module.sort(
                scaled, descending=True
            )
            sorted_probs = self.torch_module.softmax(sorted_logits, dim=-1)
            # Cumulative sum: includes current token
            cumsum_probs = self.torch_module.cumsum(sorted_probs, dim=-1)
            # Keep tokens where cumsum <= top_p OR already masked (infinite)
            keep = (cumsum_probs <= metadata.top_p) | self.torch_module.isinf(
                sorted_logits
            )
            sorted_logits = sorted_logits.masked_fill(~keep, float("-inf"))
            # Scatter back to original position
            scaled = self.torch_module.full_like(scaled, float("-inf"))
            scaled.scatter_(0, sorted_indices, sorted_logits)

        probabilities = self.torch_module.softmax(scaled, dim=-1)
        return int(self.torch_module.multinomial(probabilities, 1).item())
