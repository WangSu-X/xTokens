"""Rank-local model execution, input preparation, and sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from x_tokens.config import KVCacheConfig
from x_tokens.distributed import ParallelContext
from x_tokens.kv_cache import KVCacheSlots
from x_tokens.logger import init_logger
from x_tokens.modeling import ModelInput, ModelOutput
from x_tokens.worker.kv_cache import GPUKVCache

from .protocol import (
    ModelRunner,
    ModelRunnerOutput,
    ModelRunnerRequest,
    SamplingMetadata,
)

logger = init_logger(__name__)


@dataclass(slots=True)
class ExecuteModelState:
    logits: Any
    sampling_metadata: dict[str, SamplingMetadata]


class GPUModelRunner(ModelRunner):
    def __init__(
        self,
        context: ParallelContext,
        model: Any,
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
        self.kv_cache: GPUKVCache | None = None
        # Cache tensor options to avoid repeated creation
        self._tensor_options: dict[str, Any] | None = None

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
        get_spec = getattr(self.model, "get_kv_cache_spec", None)
        if not callable(get_spec):
            raise TypeError("model does not expose a KV cache specification")
        spec = get_spec(config.block_size)
        if config.num_gpu_blocks is not None:
            return config.num_gpu_blocks
        if config.memory_bytes is not None:
            return config.memory_bytes // spec.bytes_per_block
        if self.context.device.type != "cuda":
            raise ValueError("CPU KV cache requires num_gpu_blocks or memory_bytes")
        free_bytes, total_bytes = self.torch_module.cuda.mem_get_info(
            self.context.device
        )
        reserved_headroom = int(total_bytes * (1 - config.gpu_memory_utilization))
        available_bytes = max(0, free_bytes - reserved_headroom)
        return available_bytes // spec.bytes_per_block

    def initialize_kv_cache(self, num_blocks: int, block_size: int) -> None:
        if self.kv_cache is not None:
            raise RuntimeError("KV cache is already initialized")
        get_spec = getattr(self.model, "get_kv_cache_spec", None)
        set_cache = getattr(self.model, "set_kv_cache", None)
        if not callable(get_spec) or not callable(set_cache):
            raise TypeError("model does not support KV cache initialization")
        spec = get_spec(block_size)
        parameter = next(iter(self.model.parameters()))
        self.kv_cache = GPUKVCache(
            spec,
            num_blocks=num_blocks,
            device=self.context.device,
            dtype=parameter.dtype,
            torch_module=self.torch_module,
        )
        set_cache(self.kv_cache)
        logger.info(
            "KV cache initialized: rank=%d blocks=%d block_size=%d bytes=%d",
            self.context.coordinates.global_rank,
            num_blocks,
            block_size,
            self.kv_cache.allocated_bytes,
        )

    def execute_model(self, requests: tuple[ModelRunnerRequest, ...]) -> None:
        self.execute_model_state = None
        if not requests:
            return
        with self.torch_module.inference_mode():
            model_input = self._prepare_model_input(requests)
            output = self.model(model_input)
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
            metadata.temperature == 0
            for metadata in state.sampling_metadata.values()
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
        if self.kv_cache is not None:
            set_cache = getattr(self.model, "set_kv_cache", None)
            if callable(set_cache):
                set_cache(None)
            self.kv_cache = None

    def _get_tensor_options(self) -> dict[str, Any]:
        """Get cached tensor options for long dtype tensors."""
        if self._tensor_options is None:
            self._tensor_options = {
                "dtype": self.torch_module.long,
                "device": self.context.device,
            }
        return self._tensor_options

    def _compute_cache_slots(
        self, request: ModelRunnerRequest, block_size: int
    ) -> tuple[int, ...]:
        """Compute global slot indices for a request's tokens."""
        slots = []
        for position in range(
            request.num_computed_tokens,
            request.num_computed_tokens + len(request.scheduled_token_ids),
        ):
            block_idx = position // block_size
            offset = position % block_size
            slots.append(request.block_ids[block_idx] * block_size + offset)
        return tuple(slots)

    def _prepare_model_input(
        self, requests: tuple[ModelRunnerRequest, ...]
    ) -> ModelInput:
        if self.kv_cache is None:
            raise RuntimeError("KV cache is not initialized")
        block_size = self.kv_cache.spec.block_size
        max_length = max(len(request.scheduled_token_ids) for request in requests)
        padded = []
        masks = []
        positions = []
        cache_slots = []

        for request in requests:
            query_len = len(request.scheduled_token_ids)
            start = request.num_computed_tokens
            end = start + query_len
            query_positions = range(start, end)
            padding = max_length - query_len

            # Pad token IDs
            padded.append(
                [self.pad_token_id] * padding + list(request.scheduled_token_ids)
            )
            # Attention mask
            masks.append([0] * padding + [1] * query_len)
            # Absolute positions
            positions.append([0] * padding + list(query_positions))

            # KV cache slots (using extracted helper)
            slots = self._compute_cache_slots(request, block_size)
            cache_slots.append(KVCacheSlots(end, request.block_ids, slots))

        tensor_opts = self._get_tensor_options()
        return ModelInput(
            self.torch_module.tensor(padded, **tensor_opts),
            self.torch_module.tensor(masks, **tensor_opts),
            self.torch_module.tensor(positions, **tensor_opts),
            tuple(cache_slots),
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
            raise ValueError(f"temperature must be non-negative, got {metadata.temperature}")
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
