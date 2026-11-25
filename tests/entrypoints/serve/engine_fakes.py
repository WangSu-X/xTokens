from __future__ import annotations

from dataclasses import dataclass

from x_tokens.engine.protocol import (
    CoreErrorEvent,
    CoreFinishedEvent,
    CoreRequestOutput,
    CoreTokenEvent,
    EngineCoreClient,
    EngineHealth,
    FinishReason,
    GenerateRequest,
    InputProcessor,
    OutputProcessor,
    SamplingParams,
)


@dataclass
class FakeEngineConfig:
    error_on_request_id: str | None = None


class FakeEngineCore:
    def __init__(self, config: FakeEngineConfig | None = None) -> None:
        self.config = config or FakeEngineConfig()
        self.requests: dict[str, int] = {}
        self.closed = False

    def add_request(self, request: GenerateRequest) -> None:
        self.requests[request.request_id] = 0

    def step_fn(self) -> tuple[tuple[CoreRequestOutput, ...], bool]:
        request_outputs: list[CoreRequestOutput] = []
        for request_id, step in tuple(self.requests.items()):
            if request_id == self.config.error_on_request_id:
                request_outputs.append(CoreErrorEvent(request_id, "fake core error"))
                del self.requests[request_id]
            elif step == 0:
                request_outputs.append(CoreTokenEvent(request_id, 1))
                self.requests[request_id] = 1
            else:
                request_outputs.append(
                    CoreFinishedEvent(request_id, FinishReason.LENGTH, 1, step)
                )
                del self.requests[request_id]
        return tuple(request_outputs), bool(request_outputs)

    def post_step(self) -> None:
        pass

    def abort_requests(self, request_ids: tuple[str, ...]) -> None:
        for request_id in request_ids:
            self.requests.pop(request_id, None)

    def health(self) -> EngineHealth:
        return EngineHealth(not self.closed, "fake core")

    def close(self) -> None:
        self.closed = True
        self.requests.clear()


class FakeCoreClient(EngineCoreClient):
    def __init__(self, core: FakeEngineCore) -> None:
        self.core = core

    def add_request(self, request: GenerateRequest) -> None:
        self.core.add_request(request)

    async def get_output(self) -> tuple[CoreRequestOutput, ...]:
        request_outputs, _ = self.core.step_fn()
        self.core.post_step()
        return request_outputs

    def abort_requests(self, request_ids: tuple[str, ...]) -> None:
        self.core.abort_requests(request_ids)

    def health(self) -> EngineHealth:
        return self.core.health()

    def close(self) -> None:
        self.core.close()


class FakeInputProcessor(InputProcessor):
    pad_token_id = 0
    eos_token_ids = frozenset((2,))
    tokenizer = object()

    def process(self, request: GenerateRequest) -> GenerateRequest:
        if isinstance(request.prompt, tuple):
            prompt = request.prompt
        else:
            prompt = tuple(ord(char) for char in request.prompt) or (0,)
        sampling = request.sampling or SamplingParams()
        return GenerateRequest(
            request.request_id, request.model, prompt, sampling, request.priority
        )


class FakeOutputProcessor(OutputProcessor):
    def process_token(self, request_id: str, token_id: int) -> str:
        del request_id
        return chr(ord("A") + token_id % 26)

    def finish(self, request_id: str) -> None:
        del request_id
