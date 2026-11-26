"""Engine-side conversion from generated token IDs to text."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .protocol import OutputProcessor


class TokenizerOutputProcessor(OutputProcessor):
    """Incrementally detokenize generated IDs with a HF-compatible tokenizer."""

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        self.token_ids: dict[str, list[int]] = defaultdict(list)
        self.decoded_text: dict[str, str] = defaultdict(str)

    def process_token(self, request_id: str, token_id: int) -> str:
        token_ids = self.token_ids[request_id]
        token_ids.append(token_id)
        previous_text = self.decoded_text[request_id]
        decoded = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        text = decoded.removeprefix(previous_text)
        self.decoded_text[request_id] = previous_text + text
        return text

    def finish(self, request_id: str) -> None:
        self.token_ids.pop(request_id, None)
        self.decoded_text.pop(request_id, None)
