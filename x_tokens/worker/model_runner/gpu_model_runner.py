"""Rank-local model execution, input preparation, and sampling."""

from __future__ import annotations

from collections.abc import Callable
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
    ModelRunnerFactory,
    ModelRunnerInput,
    ModelRunnerOutput,
    SamplingMetadata,
)

logger = init_logger(__name__)


@dataclass(frozen=True, slots=True)
class GPUModelRunnerFactory(ModelRunnerFactory):
    """Pickle-friendly factory used by uni and spawn-based mp executors."""

    model_factory: Callable[[ParallelContext], Any]
    pad_token_id: int

    def __call__(
        self, context: ParallelContext, *, is_output_rank: bool
    ) -> GPUModelRunner:
        return GPUModelRunner(
            context,
            self.model_factory(context),
            pad_token_id=self.pad_token_id,
            is_output_rank=is_output_rank,
        )


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
        self.pending_step_id: int | None = None
        self.pending_logits: Any | None = None
        self.pending_sampling: tuple[SamplingMetadata, ...] = ()
        self.kv_cache: GPUKVCache | None = None

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

    def execute_model(self, inputs: ModelRunnerInput) -> ModelRunnerOutput | None:
        if self.pending_step_id is not None:
            raise RuntimeError(
                f"runner step {self.pending_step_id} must be sampled before execute"
            )
        if not inputs.requests:
            return ModelRunnerOutput(inputs.step_id) if self.is_output_rank else None
        with self.torch_module.inference_mode():
            output = self._forward(inputs)
        logits = output.logits if isinstance(output, ModelOutput) else output
        if logits is None:
            if self.is_output_rank:
                raise RuntimeError("output rank model did not produce logits")
            return None
        if logits.ndim == 3:
            logits = logits[:, -1, :]
        sample_rows = tuple(
            row
            for row, request in enumerate(inputs.requests)
            if request.sampling_metadata is not None
        )
        if logits.ndim != 2 or logits.shape[0] not in (
            len(inputs.requests),
            len(sample_rows),
        ):
            raise ValueError(
                "model logits must have shape [batch, vocab] or "
                "[batch, sequence, vocab]"
            )
        if not self.is_output_rank:
            return None
        self.pending_step_id = inputs.step_id
        self.pending_logits = (
            logits if logits.shape[0] == len(sample_rows) else logits[list(sample_rows)]
        )
        self.pending_sampling = tuple(
            request.sampling_metadata
            for request in inputs.requests
            if request.sampling_metadata is not None
        )
        return ModelRunnerOutput(inputs.step_id)

    def _forward(self, inputs: ModelRunnerInput) -> ModelOutput:
        return self.model(self._prepare_model_input(inputs))

    def sample_tokens(self, step_id: int) -> tuple[int, ...] | None:
        if not self.is_output_rank:
            return None
        if self.pending_step_id != step_id or self.pending_logits is None:
            raise RuntimeError(
                f"runner has no pending logits for step {step_id}; "
                f"pending={self.pending_step_id}"
            )
        try:
            if not self.pending_sampling:
                return ()
            if all(item.temperature == 0 for item in self.pending_sampling):
                return tuple(self.pending_logits.argmax(dim=-1).tolist())
            return tuple(
                self._sample(self.pending_logits[index], metadata)
                for index, metadata in enumerate(self.pending_sampling)
            )
        finally:
            self.pending_step_id = None
            self.pending_logits = None
            self.pending_sampling = ()

    def shutdown(self) -> None:
        self.pending_step_id = None
        self.pending_logits = None
        self.pending_sampling = ()
        if self.kv_cache is not None:
            set_cache = getattr(self.model, "set_kv_cache", None)
            if callable(set_cache):
                set_cache(None)
            self.kv_cache = None

    def _prepare_model_input(self, inputs: ModelRunnerInput) -> ModelInput:
        if self.kv_cache is None:
            raise RuntimeError("KV cache is not initialized")
        block_size = self.kv_cache.spec.block_size
        max_length = max(len(request.token_ids) for request in inputs.requests)
        padded = []
        masks = []
        positions = []
        cache_slots = []
        for request in inputs.requests:
            query_len = len(request.token_ids)
            start = request.num_computed_tokens
            end = start + query_len
            query_positions = range(start, end)
            padding = max_length - query_len
            padded.append([self.pad_token_id] * padding + list(request.token_ids))
            masks.append([0] * padding + [1] * query_len)
            positions.append([0] * padding + list(query_positions))
            slots = tuple(
                request.block_ids[position // block_size] * block_size
                + position % block_size
                for position in query_positions
            )
            cache_slots.append(KVCacheSlots(end, request.block_ids, slots))

        tensor_options = {
            "dtype": self.torch_module.long,
            "device": self.context.device,
        }
        return ModelInput(
            self.torch_module.tensor(padded, **tensor_options),
            self.torch_module.tensor(masks, **tensor_options),
            self.torch_module.tensor(positions, **tensor_options),
            tuple(cache_slots),
        )

    def _sample(self, logits: Any, metadata: SamplingMetadata) -> int:
        if metadata.temperature == 0:
            return int(self.torch_module.argmax(logits).item())
        scaled = logits / metadata.temperature
        if metadata.top_k is not None:
            top_k = min(metadata.top_k, scaled.shape[-1])
            cutoff = self.torch_module.topk(scaled, top_k).values[-1]
            scaled = scaled.masked_fill(scaled < cutoff, float("-inf"))
        if metadata.top_p < 1:
            sorted_logits, sorted_indices = self.torch_module.sort(
                scaled, descending=True
            )
            sorted_probs = self.torch_module.softmax(sorted_logits, dim=-1)
            remove = self.torch_module.cumsum(sorted_probs, dim=-1) - sorted_probs
            sorted_logits = sorted_logits.masked_fill(
                remove > metadata.top_p, float("-inf")
            )
            scaled = self.torch_module.full_like(scaled, float("-inf"))
            scaled.scatter_(0, sorted_indices, sorted_logits)
        probabilities = self.torch_module.softmax(scaled, dim=-1)
        return int(self.torch_module.multinomial(probabilities, 1).item())
