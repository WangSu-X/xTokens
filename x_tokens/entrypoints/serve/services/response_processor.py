"""Response processing and conversion for Serving layer."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator

from x_tokens.engine.protocol import EngineEvent, ErrorEvent, FinishedEvent, TokenEvent
from x_tokens.entrypoints.serve.openai.protocol import (
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseMessage,
    CompletionResponse,
    CompletionResponseChoice,
    UsageInfo,
)


def _extract_tokens_and_text(
    outputs: list[EngineEvent],
) -> tuple[str, int, int]:
    final_event = outputs[-1] if outputs else None
    full_text = "".join(
        event.text for event in outputs if isinstance(event, TokenEvent)
    )
    prompt_tokens = (
        final_event.prompt_tokens if isinstance(final_event, FinishedEvent) else 0
    )
    completion_tokens = (
        final_event.completion_tokens if isinstance(final_event, FinishedEvent) else 0
    )
    return full_text, prompt_tokens, completion_tokens


def to_completion_response(
    request_id: str,
    model: str,
    outputs: list[EngineEvent],
) -> CompletionResponse:
    full_text, prompt_tokens, completion_tokens = _extract_tokens_and_text(outputs)
    final_event = outputs[-1] if outputs else None

    return CompletionResponse(
        id=request_id,
        object="text_completion",
        created=int(time.time()),
        model=model,
        choices=[
            CompletionResponseChoice(
                index=0,
                text=full_text,
                finish_reason=(
                    final_event.finish_reason
                    if isinstance(final_event, FinishedEvent)
                    else None
                ),
            )
        ],
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


def to_chat_completion_response(
    request_id: str,
    model: str,
    outputs: list[EngineEvent],
) -> ChatCompletionResponse:
    full_text, prompt_tokens, completion_tokens = _extract_tokens_and_text(outputs)
    final_event = outputs[-1] if outputs else None

    return ChatCompletionResponse(
        id=request_id,
        object="chat.completion",
        created=int(time.time()),
        model=model,
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatCompletionResponseMessage(
                    role="assistant",
                    content=full_text,
                ),
                finish_reason=(
                    final_event.finish_reason
                    if isinstance(final_event, FinishedEvent)
                    else None
                ),
            )
        ],
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


async def collect_all_outputs(
    events: AsyncIterator[EngineEvent],
) -> list[EngineEvent]:
    outputs = []
    async for event in events:
        outputs.append(event)
        if isinstance(event, (FinishedEvent, ErrorEvent)):
            break
    return outputs
