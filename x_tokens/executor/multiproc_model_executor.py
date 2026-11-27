"""Spawn-based multiprocess ModelExecutor backend."""

from __future__ import annotations

import multiprocessing
import signal
import socket
import traceback
from multiprocessing.connection import Connection

from x_tokens.worker import GPUWorker, GPUWorkerConfig

from .model_executor import ModelExecutor
from .worker_handle import ProcessWorkerHandle


class MultiprocModelExecutor(ModelExecutor):
    def _init_executor(self) -> None:
        context = multiprocessing.get_context("spawn")
        init_method = f"tcp://127.0.0.1:{_free_port()}"
        handles: list[ProcessWorkerHandle] = []
        try:
            for rank in range(self.topology.world_size):
                parent_connection, child_connection = context.Pipe()
                worker_config = GPUWorkerConfig(
                    xtokens_config=self.config,
                    topology=self.topology,
                    rank=rank,
                    distributed_init_method=init_method,
                    output_rank=self.output_rank,
                    pad_token_id=self.pad_token_id,
                )
                process = context.Process(
                    target=_worker_entry,
                    args=(child_connection, worker_config),
                    name=f"xtokens-worker-{rank}",
                )
                process.start()
                child_connection.close()
                handles.append(ProcessWorkerHandle(rank, process, parent_connection))
            for handle in handles:
                if not handle.connection.poll(self.parallel_config.timeout_s):
                    raise TimeoutError(f"worker rank {handle.rank} startup timed out")
                status, payload = handle.connection.recv()
                if status != "ready":
                    raise RuntimeError(
                        f"worker rank {handle.rank} failed during startup:\n{payload}"
                    )
            self.worker_handles = tuple(handles)
        except Exception:
            _terminate(handles)
            raise

    def shutdown(self) -> None:
        handles = [
            handle
            for handle in self.worker_handles
            if isinstance(handle, ProcessWorkerHandle)
        ]
        if not handles:
            return
        try:
            for handle in handles:
                if handle.process.is_alive():
                    handle.send("shutdown", ())
            for handle in handles:
                if handle.process.is_alive():
                    handle.recv(self.parallel_config.timeout_s)
        finally:
            for handle in handles:
                handle.process.join(timeout=1)
            _terminate(handles)
            self.worker_handles = ()


def _worker_entry(
    connection: Connection,
    config: GPUWorkerConfig,
) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    worker = GPUWorker(config)
    try:
        worker.init_device()
        worker.load_model()
        connection.send(("ready", None))
        while True:
            method, args = connection.recv()
            try:
                result = getattr(worker, method)(*args)
                connection.send(("ok", result))
            except Exception:  # noqa: BLE001 - serialized to executor process.
                connection.send(("error", traceback.format_exc()))
                break
            if method == "shutdown":
                break
    except Exception:  # noqa: BLE001 - serialized to executor process.
        try:
            connection.send(("startup_error", traceback.format_exc()))
        except (BrokenPipeError, EOFError):
            pass
    finally:
        worker.shutdown()
        connection.close()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        return int(server.getsockname()[1])


def _terminate(handles: list[ProcessWorkerHandle]) -> None:
    for handle in handles:
        if handle.process.is_alive():
            handle.process.terminate()
    for handle in handles:
        handle.process.join(timeout=1)
        handle.connection.close()
