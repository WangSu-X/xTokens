"""Serialize LLM engine events as Server-Sent Events."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator

from x_tokens.engine.protocol import EngineEvent, ErrorEvent, FinishedEvent
from x_tokens.entrypoints.serve.openai.protocol import (
    ChatCompletionStreamResponse,
    ChatCompletionStreamResponseChoice,
    ChatCompletionStreamResponseDelta,
    CompletionStreamResponse,
    CompletionStreamResponseChoice,
    UsageInfo,
)


def _encode_error(message: str) -> str:
    data = json.dumps({"error": {"message": message}}, separators=(",", ":"))
    return f"data: {data}\n\n"


async def serialize_completion(
    request_id: str,
    model: str,
    outputs_generator: AsyncIterator[EngineEvent],
) -> AsyncIterator[str]:
    async for event in outputs_generator:
        if isinstance(event, ErrorEvent):
            yield _encode_error(event.message)
            continue
        final = isinstance(event, FinishedEvent)
        chunk = CompletionStreamResponse(
            id=request_id,
            object="text_completion.chunk",
            created=int(time.time()),
            model=model,
            choices=[
                CompletionStreamResponseChoice(
                    index=0,
                    text="" if final else event.text,
                    finish_reason=event.finish_reason if final else None,
                )
            ],
            usage=(
                UsageInfo(
                    prompt_tokens=event.prompt_tokens,
                    completion_tokens=event.completion_tokens,
                    total_tokens=event.prompt_tokens + event.completion_tokens,
                )
                if final
                else None
            ),
        )
        data = json.dumps(chunk.model_dump(exclude_none=True), separators=(",", ":"))
        yield f"data: {data}\n\n"
    yield "data: [DONE]\n\n"


async def serialize_chat(
    request_id: str,
    model: str,
    outputs_generator: AsyncIterator[EngineEvent],
) -> AsyncIterator[str]:
    first_chunk = True
    async for event in outputs_generator:
        if isinstance(event, ErrorEvent):
            yield _encode_error(event.message)
            continue
        final = isinstance(event, FinishedEvent)
        if final:
            delta = ChatCompletionStreamResponseDelta()
        else:
            delta = ChatCompletionStreamResponseDelta(
                role="assistant" if first_chunk else None,
                content=event.text,
            )
            first_chunk = False
        chunk = ChatCompletionStreamResponse(
            id=request_id,
            object="chat.completion.chunk",
            created=int(time.time()),
            model=model,
            choices=[
                ChatCompletionStreamResponseChoice(
                    index=0,
                    delta=delta,
                    finish_reason=event.finish_reason if final else None,
                )
            ],
            usage=(
                UsageInfo(
                    prompt_tokens=event.prompt_tokens,
                    completion_tokens=event.completion_tokens,
                    total_tokens=event.prompt_tokens + event.completion_tokens,
                )
                if final
                else None
            ),
        )
        data = json.dumps(chunk.model_dump(exclude_none=True), separators=(",", ":"))
        yield f"data: {data}\n\n"
    yield "data: [DONE]\n\n"
