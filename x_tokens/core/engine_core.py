"""EngineCore scheduling and its optional child-process runtime."""

from __future__ import annotations

import logging
import threading
import time
import traceback
from concurrent.futures import Future
from dataclasses import dataclass, field
from queue import Empty, Queue
from typing import TYPE_CHECKING, Any

from ..engine.protocol import (
    CoreErrorEvent,
    CoreEvent,
    CoreFinishedEvent,
    CoreTokenEvent,
    FinishReason,
    GenerateRequest,
    decode_core_request,
    encode_core_event,
)
from ..executor.base import Executor, ExecutorFactory
from ..logger import init_logger
from .config import EngineCoreConfig
from .scheduler import ScheduledRequest, Scheduler

if TYPE_CHECKING:
    from multiprocessing.connection import Connection

    from ..config import XTokensConfig

logger = init_logger(__name__)


@dataclass(slots=True)
class EngineCoreOutputs:
    """Outputs produced by one Core step, grouped by output channel."""

    outputs: dict[int, list[CoreEvent]] = field(default_factory=dict)

    def add(self, event: CoreEvent, channel: int = 0) -> None:
        self.outputs.setdefault(channel, []).append(event)

    def get(
        self, channel: int, default: tuple[CoreEvent, ...] = ()
    ) -> tuple[CoreEvent, ...]:
        events = self.outputs.get(channel)
        return tuple(events) if events is not None else default


class EngineCore:
    """Own scheduler and executor state for synchronous or loop-driven execution."""

    def __init__(
        self,
        config: EngineCoreConfig,
        *,
        executor: Executor,
        scheduler: Scheduler | None = None,
    ) -> None:
        self._config = config
        self._executor = executor
        if scheduler is not None:
            self._scheduler = scheduler
        else:
            self._scheduler = Scheduler(
                max_num_seqs=config.max_num_seqs,
                max_model_len=config.max_model_len,
                kv_cache_config=config.kv_cache_config,
                model_fingerprint=config.model_aliases[0],
            )
        self._pending_outputs = EngineCoreOutputs()
        self._shutdown_requested = False
        self._closed = False

    @classmethod
    def from_config(
        cls,
        config: XTokensConfig,
        pad_token_id: int,
        eos_token_ids: frozenset[int],
        executor_factory: ExecutorFactory,
    ) -> EngineCore:
        executor = executor_factory(config, pad_token_id, eos_token_ids)
        try:
            core_config = EngineCoreConfig.from_config(config, executor.num_gpu_blocks)
            return cls(core_config, executor=executor)
        except Exception:
            cls._shutdown_executor(executor)
            raise

    @classmethod
    def run_process(
        cls,
        startup_connection: Connection,
        command_endpoint: str,
        client_identity: bytes,
        output_endpoint: str,
        config: XTokensConfig,
        pad_token_id: int,
        eos_token_ids: frozenset[int],
        executor_factory: ExecutorFactory,
    ) -> None:
        """Spawn entry point: construct the executor and Core in the child."""
        with startup_connection:
            try:
                core = cls.from_config(
                    config, pad_token_id, eos_token_ids, executor_factory
                )
                core._serve(
                    startup_connection, command_endpoint, client_identity, output_endpoint
                )
            except Exception:
                if startup_connection.closed:
                    raise
                startup_connection.send(("error", traceback.format_exc()))

    def _serve(
        self,
        startup_connection: Connection,
        command_endpoint: str,
        client_identity: bytes,
        output_endpoint: str,
    ) -> None:
        self._input_queue: Queue[tuple[str, Any]] = Queue()
        self._output_queue: Queue[tuple[CoreEvent, ...] | Exception | None] = Queue()
        self._input_stop = threading.Event()
        input_ready: Future[None] = Future()
        output_ready: Future[None] = Future()
        input_thread = threading.Thread(
            target=self._receive_commands,
            args=(command_endpoint, client_identity, input_ready),
            name="xtokens-engine-core-input",
            daemon=True,
        )
        output_thread = threading.Thread(
            target=self._send_outputs,
            args=(output_endpoint, output_ready),
            name="xtokens-engine-core-output",
            daemon=True,
        )
        started_threads: list[threading.Thread] = []
        try:
            for thread in (input_thread, output_thread):
                thread.start()
                started_threads.append(thread)
            input_ready.result(timeout=10)
            output_ready.result(timeout=10)
            startup_connection.send(("ready", ""))
            startup_connection.close()
            try:
                self.run_busy_loop()
            except Exception as error:
                self._output_queue.put(error)
                raise
        finally:
            self._input_stop.set()
            self._output_queue.put(None)
            for thread in started_threads:
                thread.join(timeout=2)
            self.close()

    def _receive_commands(
        self, endpoint: str, client_identity: bytes, ready: Future[None]
    ) -> None:
        try:
            import zmq

            with zmq.Context() as context, context.socket(zmq.DEALER) as socket:
                socket.setsockopt(zmq.IDENTITY, client_identity)
                socket.setsockopt(zmq.LINGER, 0)
                socket.setsockopt(zmq.SNDTIMEO, 1000)
                socket.connect(endpoint)
                socket.send(b"READY")
                poller = zmq.Poller()
                poller.register(socket, zmq.POLLIN)
                ready.set_result(None)
                while not self._input_stop.is_set():
                    if not poller.poll(100):
                        continue
                    message = socket.recv_json()
                    command = message["command"]
                    if command == "add_request":
                        payload = decode_core_request(message["request"])
                    elif command == "abort_requests":
                        payload = tuple(message["request_ids"])
                    elif command == "close":
                        self._input_queue.put(("close", None))
                        return
                    else:
                        self._output_queue.put(
                            RuntimeError(f"unknown command: {command}")
                        )
                        continue
                    self._input_queue.put((command, payload))
        except Exception as error:
            if not ready.done():
                ready.set_exception(error)
            else:
                logger.exception("EngineCore input thread failed")
                self._output_queue.put(error)
                self._input_queue.put(("close", None))

    def _send_outputs(self, endpoint: str, ready: Future[None]) -> None:
        try:
            import zmq

            with zmq.Context() as context, context.socket(zmq.PUSH) as socket:
                socket.setsockopt(zmq.LINGER, 1000)
                socket.setsockopt(zmq.SNDTIMEO, 1000)
                socket.connect(endpoint)
                ready.set_result(None)
                while (output := self._output_queue.get()) is not None:
                    if isinstance(output, Exception):
                        socket.send_json({"error": str(output)})
                    else:
                        socket.send_json(
                            {"events": [encode_core_event(event) for event in output]}
                        )
        except Exception as error:
            if not ready.done():
                ready.set_exception(error)
            else:
                logger.exception("EngineCore output thread failed")
                self._input_queue.put(("close", None))

    def add_request(self, request: GenerateRequest) -> None:
        if self._closed:
            self._pending_outputs.add(
                CoreErrorEvent(request.request_id, "EngineCore is closed")
            )
            return
        try:
            self._validate_request(request)
            self._scheduler.add_request(request, request.prompt)
        except Exception as error:  # noqa: BLE001 - duplicate requests are isolated.
            logger.warning(
                "engine request rejected: request_id=%s error=%s",
                request.request_id,
                error,
            )
            self._pending_outputs.add(CoreErrorEvent(request.request_id, str(error)))

    def abort_requests(self, request_ids: tuple[str, ...]) -> None:
        for request_id in request_ids:
            self._scheduler.abort(request_id)

    @property
    def has_work(self) -> bool:
        return self._scheduler.has_work or bool(self._pending_outputs.outputs)

    def run_busy_loop(self) -> None:
        """Process commands and continuously step while scheduler work exists."""
        while self._handle_shutdown():
            self._process_input_queue()
            self._process_engine_step()

    def _handle_shutdown(self) -> bool:
        """Apply a pending shutdown and return whether the loop should continue."""
        if self._shutdown_requested:
            self.close()
        return not self._closed

    def _process_input_queue(self) -> None:
        """Wait for work when idle, then handle any queued commands."""
        while not self._shutdown_requested and not self.has_work:
            self._handle_command(*self._input_queue.get())

        while not self._shutdown_requested:
            try:
                command, payload = self._input_queue.get_nowait()
            except Empty:
                return
            self._handle_command(command, payload)

    def _process_engine_step(self) -> None:
        """Run one step and enqueue its events unless shutdown was requested."""
        if self._shutdown_requested:
            return

        outputs, model_executed = self.step_fn()
        events = outputs.get(0)
        if events:
            self._output_queue.put(events)
        self.post_step(model_executed=model_executed)
        if not model_executed and self.has_work:
            time.sleep(0.001)

    def _handle_command(self, command: str, payload: Any) -> None:
        if command == "add_request":
            self.add_request(payload)
        elif command == "abort_requests":
            self.abort_requests(payload)
        elif command == "close":
            self._shutdown_requested = True
        else:
            raise ValueError(f"unsupported EngineCore command: {command}")

    def step_fn(self) -> tuple[EngineCoreOutputs, bool]:
        """Run at most one scheduled model step in the caller's thread."""
        outputs = self._take_pending_outputs()
        if self._closed:
            return outputs, False
        scheduler_output = self._scheduler.schedule()
        if not scheduler_output.requests:
            return outputs, False
        debug_enabled = logger.isEnabledFor(logging.DEBUG)
        step_started = time.perf_counter() if debug_enabled else 0.0
        try:
            model_output = self._executor.execute_model(scheduler_output)
            token_ids = self._executor.sample_tokens(model_output, scheduler_output)
        except Exception as error:
            logger.exception(
                "model step failed: batch_size=%d request_ids=%s",
                len(scheduler_output.requests),
                tuple(request.request_id for request in scheduler_output.requests),
            )
            for request in self._scheduler.fail_batch(scheduler_output):
                outputs.add(CoreErrorEvent(request.request_id, str(error)))
            return outputs, True
        if debug_enabled:
            logger.debug(
                "model step finished: batch_size=%d duration_ms=%.2f request_ids=%s",
                len(scheduler_output.requests),
                (time.perf_counter() - step_started) * 1000,
                tuple(request.request_id for request in scheduler_output.requests),
            )
        scheduler_updates = self._scheduler.update_from_output(
            scheduler_output,
            token_ids,
            eos_token_ids=self._executor.eos_token_ids,
        )
        for scheduler_update in scheduler_updates:
            request = scheduler_update.request
            is_eos = scheduler_update.token_id in self._executor.eos_token_ids
            if not is_eos or request.request.sampling.ignore_eos:
                outputs.add(
                    CoreTokenEvent(
                        request.request_id,
                        scheduler_update.token_id,
                    )
                )
            if scheduler_update.finish_reason is not None:
                outputs.add(
                    self._finished_event(request, scheduler_update.finish_reason)
                )
        return outputs, True

    def post_step(self, *, model_executed: bool) -> None:
        """Lifecycle hook reserved for post-forward maintenance."""
        del model_executed

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._scheduler.abort_all()
        self._shutdown_executor(self._executor)
        logger.info("engine core closed")

    @staticmethod
    def _shutdown_executor(executor: Executor) -> None:
        shutdown = getattr(executor, "shutdown", None)
        if callable(shutdown):
            shutdown()

    def _validate_request(self, request: GenerateRequest) -> None:
        if not self._config.accepts_model(request.model):
            raise ValueError("requested model is not loaded by this EngineCore")
        if not isinstance(request.prompt, tuple):
            raise TypeError("EngineCore requires preprocessed prompt token IDs")
        sampling = request.sampling
        if sampling.max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if sampling.temperature < 0:
            raise ValueError("temperature must not be negative")
        if not 0 < sampling.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if sampling.top_k is not None and sampling.top_k < 1:
            raise ValueError("top_k must be positive")
        if sampling.stop:
            raise ValueError("stop strings are not supported by EngineCore")

    def _finished_event(
        self, request: ScheduledRequest, finish_reason: FinishReason
    ) -> CoreFinishedEvent:
        return CoreFinishedEvent(
            request.request_id,
            finish_reason,
            request.prompt_tokens,
            request.completion_tokens,
        )

    def _take_pending_outputs(self) -> EngineCoreOutputs:
        outputs = self._pending_outputs
        self._pending_outputs = EngineCoreOutputs()
        return outputs
