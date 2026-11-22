"""Engine protocol contracts and request/event metadata."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol


class FinishReason(StrEnum):
    STOP = "stop"
    LENGTH = "length"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class SamplingParams:
    """Canonical generation parameters accepted by the engine layer."""

    max_tokens: int = 16
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int | None = None
    stop: tuple[str, ...] = ()
    ignore_eos: bool = False


@dataclass(frozen=True, slots=True)
class GenerateRequest:
    """A protocol-independent text generation request."""

    request_id: str
    model: str
    prompt: str | tuple[int, ...]
    sampling: SamplingParams
    priority: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class InputProcessor(Protocol):
    """Turn user-facing prompts into the token IDs accepted by EngineCore."""

    def process(self, request: GenerateRequest) -> GenerateRequest: ...


class OutputProcessor(Protocol):
    """Maintain request-local detokenization state outside EngineCore."""

    def process_token(self, request_id: str, token_id: int) -> str: ...

    def finish(self, request_id: str) -> None: ...


@dataclass(frozen=True, slots=True)
class TokenEvent:
    request_id: str
    token_id: int
    text: str


@dataclass(frozen=True, slots=True)
class FinishedEvent:
    request_id: str
    finish_reason: FinishReason
    prompt_tokens: int
    completion_tokens: int


@dataclass(frozen=True, slots=True)
class ErrorEvent:
    request_id: str
    message: str
    retryable: bool = False


EngineEvent = TokenEvent | FinishedEvent | ErrorEvent


@dataclass(frozen=True, slots=True)
class CoreTokenEvent:
    """An unnormalized token event emitted by EngineCore."""

    request_id: str
    token_id: int


@dataclass(frozen=True, slots=True)
class CoreFinishedEvent:
    request_id: str
    finish_reason: FinishReason
    prompt_tokens: int
    completion_tokens: int


@dataclass(frozen=True, slots=True)
class CoreErrorEvent:
    request_id: str
    message: str
    retryable: bool = False


CoreEvent = CoreTokenEvent | CoreFinishedEvent | CoreErrorEvent


def encode_core_request(request: GenerateRequest) -> dict[str, Any]:
    sampling = request.sampling
    return {
        "request_id": request.request_id,
        "model": request.model,
        "prompt": (
            list(request.prompt) if isinstance(request.prompt, tuple) else request.prompt
        ),
        "prompt_is_tokens": isinstance(request.prompt, tuple),
        "sampling": {
            "max_tokens": sampling.max_tokens,
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "top_k": sampling.top_k,
            "stop": list(sampling.stop),
            "ignore_eos": sampling.ignore_eos,
        },
        "priority": request.priority,
        "metadata": request.metadata,
    }


def decode_core_request(data: dict[str, Any]) -> GenerateRequest:
    prompt = tuple(data["prompt"]) if data["prompt_is_tokens"] else data["prompt"]
    sampling = data["sampling"]
    return GenerateRequest(
        request_id=data["request_id"],
        model=data["model"],
        prompt=prompt,
        sampling=SamplingParams(
            max_tokens=sampling["max_tokens"],
            temperature=sampling["temperature"],
            top_p=sampling["top_p"],
            top_k=sampling["top_k"],
            stop=tuple(sampling["stop"]),
            ignore_eos=sampling["ignore_eos"],
        ),
        priority=data["priority"],
        metadata=data["metadata"],
    )


def encode_core_event(event: CoreEvent) -> dict[str, Any]:
    if isinstance(event, CoreTokenEvent):
        return {
            "type": "token",
            "request_id": event.request_id,
            "token_id": event.token_id,
        }
    if isinstance(event, CoreFinishedEvent):
        return {
            "type": "finished",
            "request_id": event.request_id,
            "finish_reason": event.finish_reason.value,
            "prompt_tokens": event.prompt_tokens,
            "completion_tokens": event.completion_tokens,
        }
    if isinstance(event, CoreErrorEvent):
        return {
            "type": "error",
            "request_id": event.request_id,
            "message": event.message,
            "retryable": event.retryable,
        }
    raise TypeError(f"unsupported Core event: {type(event)!r}")


def decode_core_event(data: dict[str, Any]) -> CoreEvent:
    if data["type"] == "token":
        return CoreTokenEvent(data["request_id"], data["token_id"])
    if data["type"] == "finished":
        return CoreFinishedEvent(
            data["request_id"],
            FinishReason(data["finish_reason"]),
            data["prompt_tokens"],
            data["completion_tokens"],
        )
    if data["type"] == "error":
        return CoreErrorEvent(data["request_id"], data["message"], data["retryable"])
    raise ValueError(f"unsupported Core event type: {data['type']}")


@dataclass(frozen=True, slots=True)
class EngineHealth:
    ready: bool
    detail: str = ""


class LLMEngineProtocol(Protocol):
    def generate(self, request: GenerateRequest) -> AsyncIterator[EngineEvent]: ...

    async def abort(self, request_id: str) -> None: ...

    async def health(self) -> EngineHealth: ...

    async def close(self) -> None: ...
