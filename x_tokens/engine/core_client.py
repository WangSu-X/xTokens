"""In-process and multiprocess EngineCore client transports."""

from __future__ import annotations

import asyncio
import json
import multiprocessing
import tempfile
import uuid
from array import array
from pathlib import Path
from typing import Any

from x_tokens.config import XTokensConfig
from x_tokens.engine.protocol import (
    CoreRequestOutput,
    EngineCoreClient,
    EngineCoreRequest,
    EngineCoreRequestType,
    EngineHealth,
    GenerateRequest,
    decode_core_event,
    encode_core_request,
)
from x_tokens.executor.protocol import ExecutorFactory


class InProcessClient(EngineCoreClient):
    """Drive EngineCore directly in the caller's process and thread."""

    def __init__(
        self,
        config: XTokensConfig,
        pad_token_id: int,
        eos_token_ids: frozenset[int],
        executor_factory: ExecutorFactory | None = None,
    ) -> None:
        from x_tokens.core import EngineCore, EngineCoreConfig

        self.engine_core = EngineCore(
            EngineCoreConfig(config, pad_token_id, eos_token_ids),
            executor_factory=executor_factory,
        )
        self.closed = False

    def add_request(self, request: GenerateRequest) -> None:
        try:
            core_request = EngineCoreRequest.from_input(
                request.request_id,
                request.sampling,
                request.prompt,
                model=request.model,
                served_model_name=self.engine_core.config.served_model_name,
            )
        except (TypeError, ValueError) as error:
            self.engine_core.reject_request(request.request_id, str(error))
            return
        self.engine_core.add_request(core_request)

    async def get_output(self) -> tuple[CoreRequestOutput, ...]:
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
    """Submit requests and asynchronously receive outputs from EngineCore."""

    def __init__(
        self,
        config: XTokensConfig,
        pad_token_id: int,
        eos_token_ids: frozenset[int],
        executor_factory: ExecutorFactory | None = None,
    ) -> None:
        from x_tokens.core import EngineCore, EngineCoreConfig

        try:
            import zmq
            import zmq.asyncio
        except ImportError as error:
            raise RuntimeError(
                "MultiProcessClient requires pyzmq; install the project dependencies"
            ) from error

        self.zmq = zmq
        self.temp_dir = tempfile.TemporaryDirectory(prefix="xtokens-core-")
        input_endpoint = f"ipc://{Path(self.temp_dir.name) / 'inputs.sock'}"
        output_endpoint = f"ipc://{Path(self.temp_dir.name) / 'outputs.sock'}"
        self.client_identity = uuid.uuid4().hex.encode()
        self.context = zmq.Context()
        self.output_context = zmq.asyncio.Context()
        self.input_socket = self.context.socket(zmq.ROUTER)
        self.output_socket = self.output_context.socket(zmq.PULL)
        self.output_socket.setsockopt(zmq.LINGER, 0)
        self.input_socket.setsockopt(zmq.LINGER, 0)
        self.input_socket.setsockopt(zmq.ROUTER_MANDATORY, 1)
        timeout_s = config.server_config.request_timeout_s or 600.0
        timeout_ms = int(timeout_s * 1000)
        self.input_socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self.input_socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.closed = False
        self.transport_closed = False
        self.process: multiprocessing.Process | None = None
        self.output_loop: asyncio.AbstractEventLoop | None = None
        try:
            self.input_socket.bind(input_endpoint)
            self.output_socket.bind(output_endpoint)
        except BaseException:
            self._cleanup_transport()
            raise

        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe(duplex=False)
        self.process = context.Process(
            target=EngineCore.run_process,
            args=(
                child,
                input_endpoint,
                self.client_identity,
                output_endpoint,
                EngineCoreConfig(config, pad_token_id, eos_token_ids),
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
            identity, handshake = self.input_socket.recv_multipart()
        except self.zmq.error.Again as error:
            self._terminate_process()
            raise TimeoutError("EngineCore input channel startup timed out") from error
        if identity != self.client_identity or handshake != b"READY":
            self._terminate_process()
            raise RuntimeError("EngineCore input channel handshake failed")
        self.closed = False

    def _send_request(
        self,
        request_type: EngineCoreRequestType,
        *,
        token_frame: bytes | None = None,
        **request_fields: Any,
    ) -> None:
        if self.closed:
            raise RuntimeError("MultiProcessClient is closed")
        if self.process is None or not self.process.is_alive():
            raise RuntimeError("EngineCore process is not running")
        try:
            frames = [
                self.client_identity,
                json.dumps(
                    {"request_type": request_type.value, **request_fields}
                ).encode(),
            ]
            if token_frame is not None:
                frames.append(token_frame)
            self.input_socket.send_multipart(frames)
        except self.zmq.error.Again as error:
            self.closed = True
            self._terminate_process()
            raise TimeoutError(
                f"timed out sending {request_type} to EngineCore process"
            ) from error

    def add_request(self, request: GenerateRequest) -> None:
        # Endpoints are local IPC peers, so native signed int64 frames share a byte order.
        tokens = (
            array("q", request.prompt).tobytes()
            if isinstance(request.prompt, tuple)
            else None
        )
        self._send_request(
            EngineCoreRequestType.ADD,
            request=encode_core_request(request, include_prompt=tokens is None),
            token_frame=tokens,
        )

    async def get_output(self) -> tuple[CoreRequestOutput, ...]:
        loop = asyncio.get_running_loop()
        if self.output_loop is None:
            self.output_loop = loop
        elif self.output_loop is not loop:
            raise RuntimeError("MultiProcessClient cannot span event loops")
        while not self.closed:
            if not await self.output_socket.poll(timeout=100):
                if self.process is None or not self.process.is_alive():
                    raise RuntimeError("EngineCore process exited unexpectedly")
                continue
            output_message = json.loads(
                await self.output_socket.recv(flags=self.zmq.DONTWAIT)
            )
            if "error" in output_message:
                raise RuntimeError(output_message["error"])
            return tuple(
                decode_core_event(encoded_output)
                for encoded_output in output_message["events"]
            )
        raise RuntimeError("MultiProcessClient is closed")

    def abort_requests(self, request_ids: tuple[str, ...]) -> None:
        if (
            request_ids
            and not self.closed
            and self.process is not None
            and self.process.is_alive()
        ):
            self._send_request(
                EngineCoreRequestType.ABORT, request_ids=list(request_ids)
            )

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
                self._send_request(EngineCoreRequestType.CLOSE)
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
        self.input_socket.close(0)
        self.output_socket.close(0)
        self.context.term()
        self.output_context.term()
        self.temp_dir.cleanup()
