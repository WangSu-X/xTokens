"""EngineCore scheduling and its optional child-process runtime."""

from __future__ import annotations

import json
import logging
import threading
import time
import traceback
from collections.abc import Iterator
from concurrent.futures import Future
from dataclasses import dataclass, field, replace
from queue import Empty, Queue
from typing import TYPE_CHECKING

from x_tokens.core.config import EngineCoreConfig
from x_tokens.core.scheduler import Scheduler
from x_tokens.engine.protocol import (
    CoreErrorEvent,
    CoreFinishedEvent,
    CoreRequestOutput,
    CoreTokenEvent,
    EngineCoreInput,
    EngineCoreRequest,
    EngineCoreRequestType,
    FinishReason,
    decode_core_request,
    encode_core_event,
)
from x_tokens.executor.protocol import Executor
from x_tokens.logger import init_logger

if TYPE_CHECKING:
    from multiprocessing.connection import Connection

logger = init_logger(__name__)


@dataclass(slots=True)
class EngineCoreOutputs:
    """Outputs produced by one Core step."""

    outputs: list[CoreRequestOutput] = field(default_factory=list)

    def add(self, request_output: CoreRequestOutput) -> None:
        """Add a request output to this batch."""
        self.outputs.append(request_output)

    def get(self) -> tuple[CoreRequestOutput, ...]:
        """Return all outputs as a tuple."""
        return tuple(self.outputs)


class EngineCore:
    """Own scheduler and executor state for synchronous or loop-driven execution."""

    def __init__(self, config: EngineCoreConfig) -> None:
        from x_tokens.executor import ModelExecutor

        self.config = config
        xtokens_config = config.xtokens_config
        executor_class = ModelExecutor.get_class(xtokens_config.parallel_config)
        self.executor = executor_class(
            xtokens_config,
            pad_token_id=config.pad_token_id,
            eos_token_ids=config.eos_token_ids,
        )
        try:
            num_gpu_blocks = self.executor.num_gpu_blocks
            if num_gpu_blocks < 1:
                raise RuntimeError("executor did not initialize KV cache")
            kv_cache_config = replace(
                xtokens_config.kv_cache_config,
                num_gpu_blocks=num_gpu_blocks,
                memory_bytes=None,
            )
            model_config = xtokens_config.model_config
            scheduler_config = xtokens_config.scheduler_config
            self.scheduler = Scheduler(
                max_num_seqs=scheduler_config.max_num_seqs,
                max_prefill_batch_tokens=scheduler_config.max_prefill_batch_tokens,
                enable_chunked_prefill=scheduler_config.enable_chunked_prefill,
                prefill_chunk_alignment=scheduler_config.prefill_chunk_alignment,
                max_model_len=model_config.max_model_len,
                kv_cache_config=kv_cache_config,
                model_fingerprint=model_config.served_model_name,
            )
        except Exception:
            self._shutdown_executor(self.executor)
            raise
        self.pending_outputs = EngineCoreOutputs()
        self.close_requested = False
        self.closed = False

    @classmethod
    def run_process(
        cls,
        startup_connection: Connection,
        input_endpoint: str,
        client_identity: bytes,
        output_endpoint: str,
        config: EngineCoreConfig,
    ) -> None:
        """Spawn entry point: construct the executor and Core in the child."""
        with startup_connection:
            try:
                core = cls(config)
                core._serve(
                    startup_connection,
                    input_endpoint,
                    client_identity,
                    output_endpoint,
                )
            except Exception:
                if startup_connection.closed:
                    raise
                startup_connection.send(("error", traceback.format_exc()))

    def _serve(
        self,
        startup_connection: Connection,
        input_endpoint: str,
        client_identity: bytes,
        output_endpoint: str,
    ) -> None:
        self._input_queue: Queue[EngineCoreInput] = Queue()
        self._output_queue: Queue[tuple[CoreRequestOutput, ...] | Exception | None] = (
            Queue()
        )
        self._input_stop = threading.Event()
        input_ready: Future[None] = Future()
        output_ready: Future[None] = Future()
        input_thread = threading.Thread(
            target=self._process_input_sockets,
            args=(input_endpoint, client_identity, input_ready),
            name="process_input_sockets",
            daemon=True,
        )
        output_thread = threading.Thread(
            target=self._process_output_sockets,
            args=(output_endpoint, output_ready),
            name="process_output_sockets",
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

    def _process_input_sockets(
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
                    header, *frames = socket.recv_multipart()
                    message = json.loads(header)
                    request_type_value = message["request_type"]
                    try:
                        request_type = EngineCoreRequestType(request_type_value)
                    except ValueError:
                        self._output_queue.put(
                            RuntimeError(f"unknown request type: {request_type_value}")
                        )
                        continue
                    if request_type is EngineCoreRequestType.ADD:
                        request_data = message["request"]
                        try:
                            request = decode_core_request(
                                request_data,
                                served_model_name=self.config.served_model_name,
                                token_frame=frames[0] if frames else None,
                            )
                        except (TypeError, ValueError) as error:
                            self._output_queue.put(
                                (CoreErrorEvent(request_data["request_id"], str(error)),)
                            )
                            continue
                        self._input_queue.put((request_type, request))
                    elif request_type is EngineCoreRequestType.ABORT:
                        request_ids = tuple(message["request_ids"])
                        self._input_queue.put((request_type, request_ids))
                    else:
                        self._input_queue.put((request_type, None))
        except Exception as error:
            self._handle_thread_error(error, ready)

    def _process_output_sockets(self, endpoint: str, ready: Future[None]) -> None:
        try:
            import zmq

            with zmq.Context() as context, context.socket(zmq.PUSH) as socket:
                socket.setsockopt(zmq.LINGER, 1000)
                socket.setsockopt(zmq.SNDTIMEO, 1000)
                socket.connect(endpoint)
                ready.set_result(None)
                while (queued_output := self._output_queue.get()) is not None:
                    if isinstance(queued_output, Exception):
                        socket.send_json({"error": str(queued_output)})
                    else:
                        socket.send_json(
                            {
                                "events": [
                                    encode_core_event(request_output)
                                    for request_output in queued_output
                                ]
                            }
                        )
        except Exception as error:
            self._handle_thread_error(error, ready)

    def _handle_thread_error(self, error: Exception, ready: Future[None]) -> None:
        if not ready.done():
            ready.set_exception(error)
        else:
            logger.exception("EngineCore thread failed")
            self._output_queue.put(error)
            self._input_queue.put((EngineCoreRequestType.CLOSE, None))

    def add_request(self, request: EngineCoreRequest) -> None:
        """Add a request to the scheduler after validation."""
        try:
            self._validate_request(request)
            self.scheduler.add_request(request)
        except (ValueError, RuntimeError) as error:
            self.reject_request(request.request_id, str(error))

    def reject_request(self, request_id: str, message: str) -> None:
        """Reject a request and emit an error event."""
        logger.warning(
            "engine request rejected: request_id=%s error=%s", request_id, message
        )
        self.pending_outputs.add(CoreErrorEvent(request_id, message))

    def abort_requests(self, request_ids: tuple[str, ...]) -> None:
        """Cancel running or waiting requests."""
        for request_id in request_ids:
            self.scheduler.abort(request_id)

    @property
    def has_work(self) -> bool:
        """Check if scheduler or pending outputs have work."""
        return self.scheduler.has_work or bool(self.pending_outputs.outputs)

    def run_busy_loop(self) -> None:
        """Process incoming requests and step while scheduler work exists."""
        while not self.close_requested and not self.closed:
            self._process_input()
            self._process_engine_step()
        self.close()

    def _process_input(self) -> None:
        for queued_request in self._iter_input_requests():
            match queued_request:
                case EngineCoreRequestType.ADD, request:
                    self.add_request(request)
                case EngineCoreRequestType.ABORT, request_ids:
                    self.abort_requests(request_ids)
                case EngineCoreRequestType.CLOSE, _:
                    self.close_requested = True
                    return

    def _iter_input_requests(self) -> Iterator[EngineCoreInput]:
        while not self.has_work:
            try:
                yield self._input_queue.get(timeout=5)
            except Empty:
                return

        # The busy loop is the input queue's only consumer.
        while not self._input_queue.empty():
            yield self._input_queue.get_nowait()

    def _process_engine_step(self) -> None:
        if self.close_requested:
            return
        outputs, model_executed = self.step_fn()
        request_outputs = outputs.get()
        if request_outputs:
            self._output_queue.put(request_outputs)
        if not model_executed and self.has_work:
            time.sleep(0.001)

    def step_fn(self) -> tuple[EngineCoreOutputs, bool]:
        """Run at most one scheduled model step in the caller's thread."""
        outputs = self.pending_outputs
        self.pending_outputs = EngineCoreOutputs()
        scheduler_output = self.scheduler.schedule()
        if not scheduler_output.running_requests:
            return outputs, False
        try:
            model_output = self.executor.execute_step(scheduler_output)
            scheduler_updates = self.scheduler.update_from_output(
                scheduler_output,
                model_output.sampled_token_ids,
                eos_token_ids=self.executor.eos_token_ids,
            )
        except Exception as error:
            logger.exception(
                "model step failed: batch_size=%d request_ids=%s",
                len(scheduler_output.running_requests),
                tuple(
                    request.request_id for request in scheduler_output.running_requests
                ),
            )
            for request in self.scheduler.fail_batch(scheduler_output):
                outputs.add(CoreErrorEvent(request.request_id, str(error)))
            return outputs, True
        for scheduler_update in scheduler_updates:
            request = scheduler_update.request
            finish_reason = scheduler_update.finish_reason
            if finish_reason is not FinishReason.STOP:
                # Don't emit token event when finish_reason is STOP
                outputs.add(
                    CoreTokenEvent(
                        request.request_id,
                        scheduler_update.token_id,
                    )
                )
            if finish_reason is not None:
                outputs.add(self._finished_event(request, finish_reason))
        return outputs, True

    def close(self) -> None:
        """Shut down scheduler and executor."""
        if self.closed:
            return
        self.closed = True
        self.scheduler.abort_all()
        self._shutdown_executor(self.executor)
        logger.info("engine core closed")

    @staticmethod
    def _shutdown_executor(executor: Executor) -> None:
        shutdown = getattr(executor, "shutdown", None)
        if callable(shutdown):
            shutdown()

    def _validate_request(self, request: EngineCoreRequest) -> None:
        sampling = request.sampling_params
        validations = [
            (sampling.max_tokens < 1, "max_tokens must be positive"),
            (sampling.temperature < 0, "temperature must not be negative"),
            (not 0 < sampling.top_p <= 1, "top_p must be in (0, 1]"),
            (sampling.top_k is not None and sampling.top_k < 1, "top_k must be positive"),
            (sampling.stop, "stop strings are not supported by EngineCore"),
        ]
        for condition, error_msg in validations:
            if condition:
                raise ValueError(error_msg)

    def _finished_event(
        self, request: EngineCoreRequest, finish_reason: FinishReason
    ) -> CoreFinishedEvent:
        return CoreFinishedEvent(
            request.request_id,
            finish_reason,
            request.prompt_tokens,
            request.completion_tokens,
        )


