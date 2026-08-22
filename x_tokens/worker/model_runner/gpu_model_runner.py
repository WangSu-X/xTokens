"""Rank-local model execution, input preparation, and sampling."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from x_tokens.distributed import ParallelContext
from x_tokens.logger import init_logger
from x_tokens.modeling import ModelInput, ModelOutput

from .base import ModelRunnerInput, ModelRunnerOutput, SamplingMetadata

logger = init_logger(__name__)


@dataclass(frozen=True, slots=True)
class GPUModelRunnerFactory:
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


class GPUModelRunner:
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

    def execute_model(self, inputs: ModelRunnerInput) -> ModelRunnerOutput | None:
        if self._pending_step_id is not None:
            raise RuntimeError(
                f"runner step {self._pending_step_id} must be sampled before execute"
            )
        if len(inputs.token_ids) != len(inputs.sampling_metadata):
            raise ValueError("runner token and sampling batch sizes differ")
        if not inputs.token_ids:
            return ModelRunnerOutput(inputs.step_id) if self.is_output_rank else None
        model_input = self._prepare_model_input(inputs.token_ids)
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

    def _prepare_model_input(
        self, token_ids: tuple[tuple[int, ...], ...]
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
        position_ids = attention_mask.cumsum(dim=-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)
        return ModelInput(input_tensor, attention_mask, position_ids)

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
