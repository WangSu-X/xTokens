"""Serving-facing LLM engine facade."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from x_tokens.config import XTokensConfig
from x_tokens.executor.base import ExecutorFactory

from .core_client import EngineCoreClient
from .input_processor import TokenizerInputProcessor
from .output_processor import TokenizerOutputProcessor
from .protocol import (
    CoreErrorEvent,
    CoreEvent,
    CoreFinishedEvent,
    CoreTokenEvent,
    EngineEvent,
    EngineHealth,
    ErrorEvent,
    FinishedEvent,
    GenerateRequest,
    InputProcessor,
    LLMEngineProtocol,
    OutputProcessor,
    TokenEvent,
)


class LLMEngine(LLMEngineProtocol):
    """Stable facade used by entrypoints regardless of Core transport."""

    def __init__(
        self,
        config: XTokensConfig | None = None,
        *,
        input_processor: InputProcessor | None = None,
        output_processor: OutputProcessor | None = None,
        executor_factory: ExecutorFactory | None = None,
        core_client: EngineCoreClient | None = None,
    ) -> None:
        if input_processor is None:
            if config is None:
                raise ValueError("config is required to construct input processing")
            input_processor = TokenizerInputProcessor.from_config(
                config.model_config.model_name,
                local_files_only=config.executor_config.local_files_only,
            )
        if output_processor is None:
            output_processor = TokenizerOutputProcessor(input_processor.tokenizer)
        if core_client is None:
            if config is None:
                raise ValueError("config is required to construct an EngineCoreClient")
            if executor_factory is None:
                from x_tokens.executor.custom_factory import (
                    create_registered_model_executor,
                )

                executor_factory = create_registered_model_executor
            core_client = EngineCoreClient.make_client(
                config,
                pad_token_id=input_processor.pad_token_id,
                eos_token_ids=input_processor.eos_token_ids,
                executor_factory=executor_factory,
            )
        self.core_client = core_client
        self.input_processor = input_processor
        self.output_processor = output_processor
        self.running_request_ids: set[str] = set()
        self.request_outputs: dict[str, asyncio.Queue[CoreEvent | BaseException]] = {}
        self.output_task: asyncio.Task[None] | None = None
        self.closed = False

    async def generate(self, request: GenerateRequest) -> AsyncIterator[EngineEvent]:
        if request.request_id in self.running_request_ids:
            raise ValueError(f"request_id is already active: {request.request_id}")
        self.running_request_ids.add(request.request_id)
        request_outputs: asyncio.Queue[CoreEvent | BaseException] = asyncio.Queue()
        self.request_outputs[request.request_id] = request_outputs
        done = False
        try:
            processed_request = await asyncio.to_thread(
                self.input_processor.process, request
            )
            self.core_client.add_request(processed_request)
            self._ensure_output_task()
            while True:
                req_output = await request_outputs.get()
                if isinstance(req_output, BaseException):
                    raise req_output
                done = isinstance(req_output, (CoreErrorEvent, CoreFinishedEvent))
                yield self.parse_output(req_output)
                if done:
                    break
        finally:
            if not done and not self.closed:
                self.core_client.abort_requests((request.request_id,))
            self.running_request_ids.discard(request.request_id)
            self.request_outputs.pop(request.request_id, None)
            self.output_processor.finish(request.request_id)
            await self._stop_output_task()

    def _ensure_output_task(self) -> None:
        if self.output_task is None or self.output_task.done():
            self.output_task = asyncio.create_task(self._read_core_outputs())

    async def _read_core_outputs(self) -> None:
        try:
            while self.running_request_ids and not self.closed:
                req_outputs = await self.core_client.get_output()
                for event in req_outputs:
                    request_outputs = self.request_outputs.get(event.request_id)
                    if request_outputs is not None:
                        request_outputs.put_nowait(event)
                if req_outputs:
                    await asyncio.sleep(0)
                else:
                    await asyncio.sleep(0.001)
        except asyncio.CancelledError:
            raise
        except Exception as error:  # noqa: BLE001 - wake every active request.
            for request_id in tuple(self.running_request_ids):
                request_outputs = self.request_outputs.get(request_id)
                if request_outputs is not None:
                    request_outputs.put_nowait(error)

    async def _stop_output_task(self, *, force: bool = False) -> None:
        if self.output_task is None or (self.running_request_ids and not force):
            return
        output_task = self.output_task
        self.output_task = None
        output_task.cancel()
        await asyncio.gather(output_task, return_exceptions=True)

    def parse_output(self, event: CoreEvent) -> EngineEvent:
        if isinstance(event, CoreTokenEvent):
            text = self.output_processor.process_token(
                event.request_id, event.token_id
            )
            return TokenEvent(event.request_id, event.token_id, text)
        if isinstance(event, CoreFinishedEvent):
            return FinishedEvent(
                event.request_id,
                event.finish_reason,
                event.prompt_tokens,
                event.completion_tokens,
            )
        if isinstance(event, CoreErrorEvent):
            return ErrorEvent(event.request_id, event.message, event.retryable)
        raise TypeError(f"unsupported Core event: {type(event)!r}")

    async def abort(self, request_id: str) -> None:
        self.core_client.abort_requests((request_id,))

    async def health(self) -> EngineHealth:
        return self.core_client.health()

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.core_client.abort_requests(tuple(self.running_request_ids))
        for request_outputs in self.request_outputs.values():
            request_outputs.put_nowait(RuntimeError("LLMEngine is closed"))
        await self._stop_output_task(force=True)
        self.core_client.close()
