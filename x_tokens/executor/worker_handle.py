"""Backend-neutral worker control-plane handles."""

from __future__ import annotations

from multiprocessing.connection import Connection
from typing import Any

from x_tokens.worker.protocol import ModelWorker

from .protocol import WorkerHandle


class LocalWorkerHandle(WorkerHandle):
    def __init__(self, rank: int, worker: ModelWorker) -> None:
        self.rank = rank
        self.worker = worker
        self.pending: tuple[bool, Any] | None = None

    def send(self, method: str, args: tuple[Any, ...]) -> None:
        if self.pending is not None:
            raise RuntimeError("local worker already has a pending command")
        try:
            self.pending = (True, getattr(self.worker, method)(*args))
        except Exception as error:  # noqa: BLE001 - re-raised through RPC contract.
            self.pending = (False, error)

    def recv(self, timeout_s: float) -> Any:
        del timeout_s
        if self.pending is None:
            raise RuntimeError("local worker has no pending command")
        success, value = self.pending
        self.pending = None
        if not success:
            raise value
        return value

    def check_health(self) -> None:
        self.worker.check_health()


class ProcessWorkerHandle(WorkerHandle):
    def __init__(self, rank: int, process: Any, connection: Connection) -> None:
        self.rank = rank
        self.process = process
        self.connection = connection
        self.pending = False

    def send(self, method: str, args: tuple[Any, ...]) -> None:
        self.check_health()
        if self.pending:
            raise RuntimeError(f"worker rank {self.rank} already has a command")
        self.connection.send((method, args))
        self.pending = True

    def recv(self, timeout_s: float) -> Any:
        if not self.pending:
            raise RuntimeError(f"worker rank {self.rank} has no pending command")
        if not self.connection.poll(timeout_s):
            raise TimeoutError(f"worker rank {self.rank} RPC timed out")
        self.pending = False
        status, payload = self.connection.recv()
        if status != "ok":
            raise RuntimeError(f"worker rank {self.rank} failed:\n{payload}")
        return payload

    def check_health(self) -> None:
        if not self.process.is_alive():
            raise RuntimeError(
                f"worker rank {self.rank} exited with code {self.process.exitcode}"
            )
