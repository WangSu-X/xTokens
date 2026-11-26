"""Engine protocol contracts and request/event metadata."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal, Protocol

if TYPE_CHECKING:
    from x_tokens.config import XTokensConfig


class FinishReason(StrEnum):
    STOP = "stop"
    LENGTH = "length"
    CANCELLED = "cancelled"


class EngineCoreRequestType(StrEnum):
    ADD = "add_request"
    ABORT = "abort_requests"
    CLOSE = "close"


class RequestStatus(StrEnum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"
    ABORTED = "aborted"


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


@dataclass(slots=True)
class EngineCoreRequest:
    request_id: str
    sampling_params: SamplingParams
    prompt_token_ids: tuple[int, ...]
    status: RequestStatus = RequestStatus.WAITING
    stop_reason: FinishReason | None = None
    output_token_ids: list[int] = field(default_factory=list)
    num_computed_tokens: int = 0
    block_hashes: list[str] = field(default_factory=list)

    @classmethod
    def from_input(
        cls,
        request_id: str,
        sampling_params: SamplingParams,
        prompt: str | tuple[int, ...],
        *,
        model: str,
        served_model_name: str,
    ) -> EngineCoreRequest:
        if model != served_model_name:
            raise ValueError("requested model is not loaded by this EngineCore")
        if not isinstance(prompt, tuple):
            raise TypeError("EngineCore requires preprocessed prompt token IDs")
        return cls(request_id, sampling_params, prompt)

    @property
    def all_token_ids(self) -> tuple[int, ...]:
        return self.prompt_token_ids + tuple(self.output_token_ids)

    @property
    def prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def completion_tokens(self) -> int:
        return len(self.output_token_ids)

    @property
    def needs_prefill(self) -> bool:
        return self.num_computed_tokens < self.prompt_tokens

    @property
    def num_uncomputed_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens - self.num_computed_tokens


type EngineCoreInput = (
    tuple[Literal[EngineCoreRequestType.ADD], EngineCoreRequest]
    | tuple[Literal[EngineCoreRequestType.ABORT], tuple[str, ...]]
    | tuple[Literal[EngineCoreRequestType.CLOSE], None]
)


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


CoreRequestOutput = CoreTokenEvent | CoreFinishedEvent | CoreErrorEvent


def encode_core_request(
    request: GenerateRequest, *, include_prompt: bool = True
) -> dict[str, Any]:
    sampling = request.sampling
    return {
        "request_id": request.request_id,
        "model": request.model,
        "prompt": (
            list(request.prompt)
            if isinstance(request.prompt, tuple)
            else request.prompt
        )
        if include_prompt
        else None,
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


def decode_core_request(
    data: dict[str, Any],
    *,
    served_model_name: str,
    token_frame: bytes | None = None,
) -> EngineCoreRequest:
    if token_frame is not None:
        prompt = tuple(memoryview(token_frame).cast("q"))
    elif data["prompt_is_tokens"]:
        prompt = tuple(data["prompt"])
    else:
        prompt = data["prompt"]
    sampling = data["sampling"]
    return EngineCoreRequest.from_input(
        request_id=data["request_id"],
        model=data["model"],
        served_model_name=served_model_name,
        prompt=prompt,
        sampling_params=SamplingParams(
            max_tokens=sampling["max_tokens"],
            temperature=sampling["temperature"],
            top_p=sampling["top_p"],
            top_k=sampling["top_k"],
            stop=tuple(sampling["stop"]),
            ignore_eos=sampling["ignore_eos"],
        ),
    )


def encode_core_event(request_output: CoreRequestOutput) -> dict[str, Any]:
    if isinstance(request_output, CoreTokenEvent):
        return {
            "type": "token",
            "request_id": request_output.request_id,
            "token_id": request_output.token_id,
        }
    if isinstance(request_output, CoreFinishedEvent):
        return {
            "type": "finished",
            "request_id": request_output.request_id,
            "finish_reason": request_output.finish_reason.value,
            "prompt_tokens": request_output.prompt_tokens,
            "completion_tokens": request_output.completion_tokens,
        }
    if isinstance(request_output, CoreErrorEvent):
        return {
            "type": "error",
            "request_id": request_output.request_id,
            "message": request_output.message,
            "retryable": request_output.retryable,
        }
    raise TypeError(f"unsupported Core request output: {type(request_output)!r}")


def decode_core_event(data: dict[str, Any]) -> CoreRequestOutput:
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


class EngineCoreClient(ABC):
    """Transport boundary used by LLMEngine to drive EngineCore."""

    @staticmethod
    def make_client(
        config: XTokensConfig,
        *,
        pad_token_id: int,
        eos_token_ids: frozenset[int],
    ) -> EngineCoreClient:
        from .core_client import InProcessClient, MultiProcessClient

        mode = config.server_config.core_client_mode
        if mode == "in_proc":
            return InProcessClient(config, pad_token_id, eos_token_ids)
        if mode == "multi_proc":
            return MultiProcessClient(config, pad_token_id, eos_token_ids)
        raise ValueError(f"unsupported EngineCore client mode: {mode}")

    @abstractmethod
    def add_request(self, request: GenerateRequest) -> None: ...

    @abstractmethod
    async def get_output(self) -> tuple[CoreRequestOutput, ...]: ...

    @abstractmethod
    def abort_requests(self, request_ids: tuple[str, ...]) -> None: ...

    @abstractmethod
    def health(self) -> EngineHealth: ...

    @abstractmethod
    def close(self) -> None: ...


class LLMEngineProtocol(Protocol):
    def generate(self, request: GenerateRequest) -> AsyncIterator[EngineEvent]: ...

    async def abort(self, request_id: str) -> None: ...

    async def health(self) -> EngineHealth: ...

    async def close(self) -> None: ...
