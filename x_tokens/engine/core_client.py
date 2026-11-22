"""EngineCore client contract, factory, and transport implementations."""

from __future__ import annotations

import asyncio
import json
import multiprocessing
import tempfile
import threading
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from x_tokens.config import XTokensConfig
from x_tokens.engine.protocol import (
    CoreEvent,
    EngineHealth,
    GenerateRequest,
    decode_core_event,
    encode_core_request,
)
from x_tokens.executor.base import ExecutorFactory


class EngineCoreClient(ABC):
    """Transport boundary used by LLMEngine to drive EngineCore."""

    @staticmethod
    def make_client(
        config: XTokensConfig,
        *,
        pad_token_id: int,
        eos_token_ids: frozenset[int],
        executor_factory: ExecutorFactory,
    ) -> EngineCoreClient:
        mode = config.server_config.core_client_mode
        if mode == "in_proc":
            return InProcessClient(
                config, pad_token_id, eos_token_ids, executor_factory
            )
        if mode == "multi_proc":
            return MultiProcessClient(
                config, pad_token_id, eos_token_ids, executor_factory
            )
        raise ValueError(f"unsupported EngineCore client mode: {mode}")

    @abstractmethod
    def add_request(self, request: GenerateRequest) -> None: ...

    @abstractmethod
    async def get_output(self) -> tuple[CoreEvent, ...]: ...

    @abstractmethod
    def abort_requests(self, request_ids: tuple[str, ...]) -> None: ...

    @abstractmethod
    def health(self) -> EngineHealth: ...

    @abstractmethod
    def close(self) -> None: ...


class InProcessClient(EngineCoreClient):
    """Drive EngineCore directly in the caller's process and thread."""

    def __init__(
        self,
        config: XTokensConfig,
        pad_token_id: int,
        eos_token_ids: frozenset[int],
        executor_factory: ExecutorFactory,
    ) -> None:
        from x_tokens.core.engine_core import EngineCore

        self.engine_core = EngineCore.from_config(
            config, pad_token_id, eos_token_ids, executor_factory
        )
        self.closed = False

    def add_request(self, request: GenerateRequest) -> None:
        self.engine_core.add_request(request)

    async def get_output(self) -> tuple[CoreEvent, ...]:
        if self.closed:
            return ()
        outputs, model_executed = self.engine_core.step_fn()
        self.engine_core.post_step(model_executed=model_executed)
        return outputs.get(0)

    def abort_requests(self, request_ids: tuple[str, ...]) -> None:
        self.engine_core.abort_requests(request_ids)

    def health(self) -> EngineHealth:
        return EngineHealth(
            not self.closed,
            "In-process EngineCore is ready"
            if not self.closed
            else "In-process EngineCore is closed",
        )

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.engine_core.close()


class MultiProcessClient(EngineCoreClient):
    """Submit commands and asynchronously receive outputs from EngineCore."""

    def __init__(
        self,
        config: XTokensConfig,
        pad_token_id: int,
        eos_token_ids: frozenset[int],
        executor_factory: ExecutorFactory,
    ) -> None:
        from x_tokens.core.engine_core import EngineCore

        try:
            import zmq
        except ImportError as error:
            raise RuntimeError(
                "MultiProcessClient requires pyzmq; install the project dependencies"
            ) from error

        self.zmq = zmq
        self.temp_dir = tempfile.TemporaryDirectory(prefix="xtokens-core-")
        command_endpoint = f"ipc://{Path(self.temp_dir.name) / 'commands.sock'}"
        output_endpoint = f"ipc://{Path(self.temp_dir.name) / 'outputs.sock'}"
        self.client_identity = uuid.uuid4().hex.encode()
        self.context = zmq.Context()
        self.command_socket = self.context.socket(zmq.ROUTER)
        self.command_socket.setsockopt(zmq.LINGER, 0)
        self.command_socket.setsockopt(zmq.ROUTER_MANDATORY, 1)
        timeout_s = config.server_config.request_timeout_s or 600.0
        timeout_ms = int(timeout_s * 1000)
        self.command_socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self.command_socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.command_socket.bind(command_endpoint)
        self.closed = False
        self.transport_closed = False
        self.process: multiprocessing.Process | None = None
        self.process_started = threading.Event()
        self.receiver_stop = threading.Event()
        self.receiver_ready = threading.Event()
        self.output_queue: asyncio.Queue[
            tuple[CoreEvent, ...] | BaseException
        ] = asyncio.Queue()
        self.output_lock = threading.Lock()
        self.output_loop: asyncio.AbstractEventLoop | None = None
        self.pending_outputs: list[tuple[CoreEvent, ...] | BaseException] = []
        self.output_thread = threading.Thread(
            target=self._receive_outputs,
            args=(output_endpoint,),
            name="xtokens-engine-core-output",
            daemon=True,
        )
        self.output_thread.start()
        if not self.receiver_ready.wait(timeout=5):
            self._cleanup_transport()
            raise TimeoutError("EngineCore output receiver failed to initialize")

        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe(duplex=False)
        self.process = context.Process(
            target=EngineCore.run_process,
            args=(
                child,
                command_endpoint,
                self.client_identity,
                output_endpoint,
                config,
                pad_token_id,
                eos_token_ids,
                executor_factory,
            ),
            name=f"xtokens-engine-core-{uuid.uuid4().hex[:8]}",
        )
        try:
            self.process.start()
        except Exception:
            parent.close()
            child.close()
            self._cleanup_transport()
            raise
        self.process_started.set()
        child.close()
        startup_timeout_s = max(timeout_s, config.parallel_config.timeout_s)
        if not parent.poll(startup_timeout_s):
            parent.close()
            self._terminate_process()
            raise TimeoutError("EngineCore process startup timed out")
        try:
            status, detail = parent.recv()
        except EOFError as error:
            parent.close()
            self._terminate_process()
            raise RuntimeError("EngineCore process exited during startup") from error
        parent.close()
        if status != "ready":
            self._terminate_process()
            raise RuntimeError(f"EngineCore process startup failed:\n{detail}")
        try:
            identity, handshake = self.command_socket.recv_multipart()
        except self.zmq.error.Again as error:
            self._terminate_process()
            raise TimeoutError("EngineCore command channel startup timed out") from error
        if identity != self.client_identity or handshake != b"READY":
            self._terminate_process()
            raise RuntimeError("EngineCore command channel handshake failed")
        self.closed = False

    def _send_command(self, command: str, **payload: Any) -> None:
        if self.closed:
            raise RuntimeError("MultiProcessClient is closed")
        if self.process is None or not self.process.is_alive():
            raise RuntimeError("EngineCore process is not running")
        try:
            self.command_socket.send_multipart(
                (
                    self.client_identity,
                    json.dumps({"command": command, **payload}).encode(),
                )
            )
        except self.zmq.error.Again as error:
            self.closed = True
            self._terminate_process()
            raise TimeoutError(
                f"timed out sending {command} to EngineCore process"
            ) from error

    def add_request(self, request: GenerateRequest) -> None:
        self._send_command("add_request", request=encode_core_request(request))

    async def get_output(self) -> tuple[CoreEvent, ...]:
        loop = asyncio.get_running_loop()
        with self.output_lock:
            if self.output_loop is None:
                self.output_loop = loop
                for output in self.pending_outputs:
                    self.output_queue.put_nowait(output)
                self.pending_outputs.clear()
            elif self.output_loop is not loop:
                raise RuntimeError("MultiProcessClient cannot span event loops")
        output = await self.output_queue.get()
        if isinstance(output, BaseException):
            raise output
        return output

    def abort_requests(self, request_ids: tuple[str, ...]) -> None:
        if (
            request_ids
            and not self.closed
            and self.process is not None
            and self.process.is_alive()
        ):
            self._send_command("abort_requests", request_ids=list(request_ids))

    def health(self) -> EngineHealth:
        ready = not self.closed and self.process is not None and self.process.is_alive()
        return EngineHealth(
            ready,
            "Multi-process EngineCore is ready"
            if ready
            else "Multi-process EngineCore is not running",
        )

    def close(self) -> None:
        if self.closed:
            return
        try:
            if self.process is not None and self.process.is_alive():
                self._send_command("close")
        finally:
            self.closed = True
            if self.process is not None:
                self.process.join(timeout=5)
            if self.process is not None and self.process.is_alive():
                self._terminate_process()
            self._cleanup_transport()

    def _terminate_process(self) -> None:
        if self.process is not None and self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5)
        self._cleanup_transport()

    def _cleanup_transport(self) -> None:
        if self.transport_closed:
            return
        self.transport_closed = True
        self.receiver_stop.set()
        self.command_socket.close(0)
        self.context.term()
        if self.output_thread.is_alive():
            self.output_thread.join(timeout=1)
        self.temp_dir.cleanup()

    def _publish_output(self, output: tuple[CoreEvent, ...] | BaseException) -> None:
        with self.output_lock:
            loop = self.output_loop
            if loop is None:
                self.pending_outputs.append(output)
                return
        try:
            loop.call_soon_threadsafe(self.output_queue.put_nowait, output)
        except RuntimeError:
            pass

    def _receive_outputs(self, endpoint: str) -> None:
        import zmq as zmq_module

        context = zmq_module.Context()
        socket = context.socket(zmq_module.PULL)
        socket.setsockopt(zmq_module.LINGER, 0)
        try:
            socket.bind(endpoint)
            self.receiver_ready.set()
            poller = zmq_module.Poller()
            poller.register(socket, zmq_module.POLLIN)
            while not self.receiver_stop.is_set():
                if poller.poll(100):
                    message = socket.recv_json()
                    if "error" in message:
                        self._publish_output(RuntimeError(message["error"]))
                    else:
                        events = tuple(
                            decode_core_event(event) for event in message["events"]
                        )
                        if events:
                            self._publish_output(events)
                if (
                    not self.closed
                    and self.process_started.is_set()
                    and self.process is not None
                    and not self.process.is_alive()
                ):
                    self._publish_output(
                        RuntimeError("EngineCore process exited unexpectedly")
                    )
                    return
        except Exception as error:  # noqa: BLE001 - forwarded to async consumers.
            self.receiver_ready.set()
            self._publish_output(error)
        finally:
            socket.close(0)
            context.term()
