"""Engine-side prompt preprocessing."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from .protocol import GenerateRequest, InputProcessor


class TokenizerInputProcessor(InputProcessor):
    """Adapt a Hugging Face-compatible tokenizer for the Engine boundary."""

    def __init__(
        self, tokenizer: Any, *, generation_eos: int | list[int] | None = None
    ) -> None:
        self.hf_tokenizer = tokenizer
        self.generation_eos = generation_eos

    @classmethod
    def from_config(
        cls, model: str, *, local_files_only: bool
    ) -> TokenizerInputProcessor:
        try:
            from transformers import AutoTokenizer, GenerationConfig
        except ImportError as error:
            raise RuntimeError(
                "Tokenizer input processing requires transformers"
            ) from error
        tokenizer = AutoTokenizer.from_pretrained(
            model, local_files_only=local_files_only
        )
        generation_eos = None
        if (
            not Path(model).is_dir()
            or (Path(model) / "generation_config.json").is_file()
        ):
            try:
                generation_config = GenerationConfig.from_pretrained(
                    model, local_files_only=local_files_only
                )
            except OSError:
                pass
            else:
                generation_eos = generation_config.eos_token_id
        return cls(tokenizer, generation_eos=generation_eos)

    @property
    def pad_token_id(self) -> int:
        pad_token_id = self.hf_tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.hf_tokenizer.eos_token_id
        if pad_token_id is None:
            raise ValueError("tokenizer must define pad_token_id or eos_token_id")
        return int(pad_token_id)

    @property
    def eos_token_ids(self) -> frozenset[int]:
        value = self.generation_eos
        if value is None:
            value = self.hf_tokenizer.eos_token_id
        if value is None:
            return frozenset()
        return frozenset((value,)) if isinstance(value, int) else frozenset(value)

    @property
    def tokenizer(self) -> Any:
        """Expose the shared tokenizer to the paired output processor."""
        return self.hf_tokenizer

    def process(self, request: GenerateRequest) -> GenerateRequest:
        if isinstance(request.prompt, tuple):
            token_ids = request.prompt
        else:
            encoded = self.hf_tokenizer(request.prompt, add_special_tokens=True)
            token_ids = tuple(int(token_id) for token_id in encoded["input_ids"])
        if not token_ids:
            raise ValueError("prompt must contain at least one token")
        return replace(request, prompt=token_ids)
