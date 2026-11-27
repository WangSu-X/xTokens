"""Serving-facing LLM engine facade."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from x_tokens.config import XTokensConfig

from .input_processor import TokenizerInputProcessor
from .output_processor import TokenizerOutputProcessor
from .protocol import (
    CoreErrorEvent,
    CoreFinishedEvent,
    CoreRequestOutput,
    CoreTokenEvent,
    EngineCoreClient,
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
            core_client = EngineCoreClient.make_client(
                config,
                pad_token_id=input_processor.pad_token_id,
                eos_token_ids=input_processor.eos_token_ids,
            )
        self.core_client = core_client
        self.input_processor = input_processor
        self.output_processor = output_processor
        self.running_request_ids: set[str] = set()
        self.request_output_queues: dict[
            str, asyncio.Queue[CoreRequestOutput | BaseException]
        ] = {}
        self.output_task: asyncio.Task[None] | None = None
        self.closed = False

    async def generate(self, request: GenerateRequest) -> AsyncIterator[EngineEvent]:
        if request.request_id in self.running_request_ids:
            raise ValueError(f"request_id is already active: {request.request_id}")
        self.running_request_ids.add(request.request_id)
        request_output_queue: asyncio.Queue[CoreRequestOutput | BaseException] = (
            asyncio.Queue()
        )
        self.request_output_queues[request.request_id] = request_output_queue
        done = False
        try:
            processed_request = await asyncio.to_thread(
                self.input_processor.process, request
            )
            self.core_client.add_request(processed_request)
            self._ensure_output_task()
            while True:
                request_output = await request_output_queue.get()
                if isinstance(request_output, BaseException):
                    raise request_output
                done = isinstance(request_output, (CoreErrorEvent, CoreFinishedEvent))
                yield self.parse_output(request_output)
                if done:
                    break
        finally:
            if not done and not self.closed:
                self.core_client.abort_requests((request.request_id,))
            self.running_request_ids.discard(request.request_id)
            self.request_output_queues.pop(request.request_id, None)
            self.output_processor.finish(request.request_id)
            await self._stop_output_task()

    def _ensure_output_task(self) -> None:
        if self.output_task is None or self.output_task.done():
            self.output_task = asyncio.create_task(self._read_core_outputs())

    async def _read_core_outputs(self) -> None:
        try:
            while self.running_request_ids and not self.closed:
                core_outputs = await self.core_client.get_output()
                for request_output in core_outputs:
                    request_output_queue = self.request_output_queues.get(
                        request_output.request_id
                    )
                    if request_output_queue is not None:
                        request_output_queue.put_nowait(request_output)
                if not core_outputs:
                    await asyncio.sleep(0.001)
        except asyncio.CancelledError:
            raise
        except Exception as error:  # noqa: BLE001 - wake every active request.
            for request_id in tuple(self.running_request_ids):
                request_output_queue = self.request_output_queues.get(request_id)
                if request_output_queue is not None:
                    request_output_queue.put_nowait(error)

    async def _stop_output_task(self, *, force: bool = False) -> None:
        if self.output_task is None or (self.running_request_ids and not force):
            return
        output_task = self.output_task
        self.output_task = None
        output_task.cancel()
        await asyncio.gather(output_task, return_exceptions=True)

    def parse_output(self, request_output: CoreRequestOutput) -> EngineEvent:
        """Convert a Core output to an Engine event with decoded text."""
        if isinstance(request_output, CoreTokenEvent):
            text = self.output_processor.process_token(
                request_output.request_id, request_output.token_id
            )
            return TokenEvent(request_output.request_id, request_output.token_id, text)
        if isinstance(request_output, CoreFinishedEvent):
            return FinishedEvent(
                request_output.request_id,
                request_output.finish_reason,
                request_output.prompt_tokens,
                request_output.completion_tokens,
            )
        if isinstance(request_output, CoreErrorEvent):
            return ErrorEvent(
                request_output.request_id,
                request_output.message,
                request_output.retryable,
            )
        raise TypeError(f"unsupported Core request output: {type(request_output)!r}")

    async def abort(self, request_id: str) -> None:
        self.core_client.abort_requests((request_id,))

    async def health(self) -> EngineHealth:
        return self.core_client.health()

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.core_client.abort_requests(tuple(self.running_request_ids))
        for request_output_queue in self.request_output_queues.values():
            request_output_queue.put_nowait(RuntimeError("LLMEngine is closed"))
        await self._stop_output_task(force=True)
        self.core_client.close()
