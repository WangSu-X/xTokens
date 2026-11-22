"""Rank-local model execution, input preparation, and sampling."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from x_tokens.config import KVCacheConfig
from x_tokens.distributed import ParallelContext
from x_tokens.logger import init_logger
from x_tokens.modeling import ModelInput, ModelOutput
from x_tokens.worker.kv_cache import GPUKVCache

from .base import (
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
        self._torch = torch_module
        self._pending_step_id: int | None = None
        self._pending_logits: Any | None = None
        self._pending_sampling: tuple[SamplingMetadata, ...] = ()
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
        if self.context.device.type != "cuda":
            raise ValueError("CPU KV cache requires num_gpu_blocks")
        free_bytes, total_bytes = self._torch.cuda.mem_get_info(self.context.device)
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
            torch_module=self._torch,
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
        if self._pending_step_id is not None:
            raise RuntimeError(
                f"runner step {self._pending_step_id} must be sampled before execute"
            )
        if len(inputs.token_ids) != len(inputs.sampling_metadata):
            raise ValueError("runner token and sampling batch sizes differ")
        if len(inputs.kv_cache_metadata) != len(inputs.token_ids):
            raise ValueError("custom model runner requires KV metadata per request")
        if not inputs.token_ids:
            return ModelRunnerOutput(inputs.step_id) if self.is_output_rank else None
        model_input = self._prepare_model_input(
            inputs.token_ids, inputs.kv_cache_metadata
        )
        with self._torch.inference_mode():
            output = self.model(model_input)
        logits = output.logits if isinstance(output, ModelOutput) else output
        if logits is None:
            if self.is_output_rank:
                raise RuntimeError("output rank model did not produce logits")
            return None
        if logits.ndim == 3:
            logits = logits[:, -1, :]
        if logits.ndim != 2 or logits.shape[0] != len(inputs.token_ids):
            raise ValueError(
                "model logits must have shape [batch, vocab] or "
                "[batch, sequence, vocab]"
            )
        if not self.is_output_rank:
            return None
        self._pending_step_id = inputs.step_id
        self._pending_logits = logits
        self._pending_sampling = inputs.sampling_metadata
        return ModelRunnerOutput(inputs.step_id)

    def sample_tokens(self, step_id: int) -> tuple[int, ...] | None:
        if not self.is_output_rank:
            return None
        if self._pending_step_id != step_id or self._pending_logits is None:
            raise RuntimeError(
                f"runner has no pending logits for step {step_id}; "
                f"pending={self._pending_step_id}"
            )
        try:
            return tuple(
                self._sample(self._pending_logits[index], metadata)
                for index, metadata in enumerate(self._pending_sampling)
            )
        finally:
            self._pending_step_id = None
            self._pending_logits = None
            self._pending_sampling = ()

    def shutdown(self) -> None:
        self._pending_step_id = None
        self._pending_logits = None
        self._pending_sampling = ()
        if self.kv_cache is not None:
            set_cache = getattr(self.model, "set_kv_cache", None)
            if callable(set_cache):
                set_cache(None)
            self.kv_cache = None

    def _prepare_model_input(
        self,
        token_ids: tuple[tuple[int, ...], ...],
        kv_cache_metadata: tuple[Any, ...],
    ) -> ModelInput:
        max_length = max(len(tokens) for tokens in token_ids)
        padded = [
            [self.pad_token_id] * (max_length - len(tokens)) + list(tokens)
            for tokens in token_ids
        ]
        masks = [
            [0] * (max_length - len(tokens)) + [1] * len(tokens) for tokens in token_ids
        ]
        input_tensor = self._torch.tensor(
            padded, dtype=self._torch.long, device=self.context.device
        )
        attention_mask = self._torch.tensor(
            masks, dtype=self._torch.long, device=self.context.device
        )
        positions = [
            [0] * (max_length - item.query_len)
            + list(range(item.context_len - item.query_len, item.context_len))
            for item in kv_cache_metadata
        ]
        position_ids = self._torch.tensor(
            positions, dtype=self._torch.long, device=self.context.device
        )
        return ModelInput(
            input_tensor,
            attention_mask,
            position_ids,
            tuple(kv_cache_metadata),
        )

    def _sample(self, logits: Any, metadata: SamplingMetadata) -> int:
        if metadata.temperature == 0:
            return int(self._torch.argmax(logits).item())
        scaled = logits / metadata.temperature
        if metadata.top_k is not None:
            top_k = min(metadata.top_k, scaled.shape[-1])
            cutoff = self._torch.topk(scaled, top_k).values[-1]
            scaled = scaled.masked_fill(scaled < cutoff, float("-inf"))
        if metadata.top_p < 1:
            sorted_logits, sorted_indices = self._torch.sort(scaled, descending=True)
            sorted_probs = self._torch.softmax(sorted_logits, dim=-1)
            remove = self._torch.cumsum(sorted_probs, dim=-1) - sorted_probs
            sorted_logits = sorted_logits.masked_fill(
                remove > metadata.top_p, float("-inf")
            )
            scaled = self._torch.full_like(scaled, float("-inf"))
            scaled.scatter_(0, sorted_indices, sorted_logits)
        probabilities = self._torch.softmax(scaled, dim=-1)
        return int(self._torch.multinomial(probabilities, 1).item())
