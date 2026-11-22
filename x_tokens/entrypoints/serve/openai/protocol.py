"""Pydantic DTOs for the supported OpenAI-compatible API surface."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class StreamOptions(BaseModel):
    include_usage: bool = False


class CompletionRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: int = Field(default=16, ge=1)
    temperature: float = Field(default=1.0, ge=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=1)
    stop: str | list[str] | None = None
    ignore_eos: bool = False
    stream: bool = False
    stream_options: StreamOptions | None = None


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int | None = Field(default=None, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    temperature: float = Field(default=1.0, ge=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=1)
    stop: str | list[str] | None = None
    ignore_eos: bool = False
    stream: bool = False
    stream_options: StreamOptions | None = None

    @model_validator(mode="after")
    def validate_token_limits(self) -> ChatCompletionRequest:
        if self.max_tokens is not None and self.max_completion_tokens is not None:
            raise ValueError(
                "only one of max_tokens or max_completion_tokens may be set"
            )
        return self


# ============================================================================
# Response Types
# ============================================================================


class UsageInfo(BaseModel):
    """Token usage information."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionResponseChoice(BaseModel):
    """Choice in a completion response."""
    index: int
    text: str
    finish_reason: str | None = None


class CompletionResponse(BaseModel):
    """Completion API response (non-streaming)."""
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[CompletionResponseChoice]
    usage: UsageInfo


class CompletionStreamResponseChoice(BaseModel):
    """Choice in a completion stream response."""
    index: int
    text: str
    finish_reason: str | None = None


class CompletionStreamResponse(BaseModel):
    """Completion API response (streaming)."""
    id: str
    object: str = "text_completion.chunk"
    created: int
    model: str
    choices: list[CompletionStreamResponseChoice]
    usage: UsageInfo | None = None


class ChatCompletionResponseMessage(BaseModel):
    """Message in a chat completion response."""
    role: str
    content: str


class ChatCompletionResponseChoice(BaseModel):
    """Choice in a chat completion response."""
    index: int
    message: ChatCompletionResponseMessage
    finish_reason: str | None = None


class ChatCompletionResponse(BaseModel):
    """Chat completion API response (non-streaming)."""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionResponseChoice]
    usage: UsageInfo


class ChatCompletionStreamResponseDelta(BaseModel):
    """Delta in a chat completion stream response."""
    role: str | None = None
    content: str | None = None


class ChatCompletionStreamResponseChoice(BaseModel):
    """Choice in a chat completion stream response."""
    index: int
    delta: ChatCompletionStreamResponseDelta
    finish_reason: str | None = None


class ChatCompletionStreamResponse(BaseModel):
    """Chat completion API response (streaming)."""
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionStreamResponseChoice]
    usage: UsageInfo | None = None
