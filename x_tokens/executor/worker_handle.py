"""Backend-neutral worker control-plane handles."""

from __future__ import annotations

from multiprocessing.connection import Connection
from typing import Any, Protocol

from x_tokens.worker.base import ModelWorker


class WorkerHandle(Protocol):
    rank: int

    def send(self, method: str, args: tuple[Any, ...]) -> None: ...

    def recv(self, timeout_s: float) -> Any: ...

    def check_health(self) -> None: ...


class LocalWorkerHandle:
    def __init__(self, rank: int, worker: ModelWorker) -> None:
        self.rank = rank
        self.worker = worker
        self._pending: tuple[bool, Any] | None = None

    def send(self, method: str, args: tuple[Any, ...]) -> None:
        if self._pending is not None:
            raise RuntimeError("local worker already has a pending command")
        try:
            self._pending = (True, getattr(self.worker, method)(*args))
        except Exception as error:  # noqa: BLE001 - re-raised through RPC contract.
            self._pending = (False, error)

    def recv(self, timeout_s: float) -> Any:
        del timeout_s
        if self._pending is None:
            raise RuntimeError("local worker has no pending command")
        success, value = self._pending
        self._pending = None
        if not success:
            raise value
        return value

    def check_health(self) -> None:
        self.worker.check_health()


class ProcessWorkerHandle:
    def __init__(self, rank: int, process: Any, connection: Connection) -> None:
        self.rank = rank
        self.process = process
        self.connection = connection
        self._pending = False

    def send(self, method: str, args: tuple[Any, ...]) -> None:
        self.check_health()
        if self._pending:
            raise RuntimeError(f"worker rank {self.rank} already has a command")
        self.connection.send((method, args))
        self._pending = True

    def recv(self, timeout_s: float) -> Any:
        if not self._pending:
            raise RuntimeError(f"worker rank {self.rank} has no pending command")
        if not self.connection.poll(timeout_s):
            raise TimeoutError(f"worker rank {self.rank} RPC timed out")
        self._pending = False
        status, payload = self.connection.recv()
        if status != "ok":
            raise RuntimeError(f"worker rank {self.rank} failed:\n{payload}")
        return payload

    def check_health(self) -> None:
        if not self.process.is_alive():
            raise RuntimeError(
                f"worker rank {self.rank} exited with code {self.process.exitcode}"
            )
