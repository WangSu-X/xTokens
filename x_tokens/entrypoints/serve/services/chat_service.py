"""Chat Completion API service layer."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from x_tokens.engine.llm_engine import LLMEngine
from x_tokens.engine.protocol import GenerateRequest
from x_tokens.entrypoints.serve.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from x_tokens.entrypoints.serve.services.request_processor import process_chat_request
from x_tokens.entrypoints.serve.services.response_processor import (
    collect_all_outputs,
    to_chat_completion_response,
)
from x_tokens.entrypoints.serve.sse_serializer import serialize_chat


class ChatCompletionService:
    """Generate chat completions via the LLM engine."""

    def __init__(self, engine: LLMEngine, *, tokenizer: Any | None = None) -> None:
        self.engine = engine
        self.tokenizer = tokenizer

    async def create_chat_completion(
        self, request: ChatCompletionRequest, *, request_id: str
    ) -> ChatCompletionResponse | AsyncIterator[str]:
        prompt, sampling_params = await asyncio.to_thread(
            process_chat_request, request, tokenizer=self.tokenizer
        )
        outputs_generator = self.engine.generate(
            GenerateRequest(
                request_id=request_id,
                model=request.model,
                prompt=prompt,
                sampling=sampling_params,
            )
        )

        if request.stream:
            return serialize_chat(
                request_id=request_id,
                model=request.model,
                outputs_generator=outputs_generator,
            )

        return to_chat_completion_response(
            request_id=request_id,
            model=request.model,
            outputs=await collect_all_outputs(outputs_generator),
        )
