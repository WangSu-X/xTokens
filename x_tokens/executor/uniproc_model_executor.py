"""In-process ModelExecutor backend."""

from __future__ import annotations

from x_tokens.worker import GPUWorker, GPUWorkerConfig

from .model_executor import ModelExecutor
from .worker_handle import LocalWorkerHandle


class UniProcModelExecutor(ModelExecutor):
    def _init_executor(self) -> None:
        if self.parallel_config.distributed_executor_backend != "uni":
            raise ValueError("UniProcModelExecutor requires the uni backend")
        worker = GPUWorker(
            GPUWorkerConfig(
                self.config,
                self.topology,
                rank=0,
                distributed_init_method="env://",
                output_rank=self.output_rank,
                pad_token_id=self.pad_token_id,
            )
        )
        try:
            worker.init_device()
            worker.load_model()
        except Exception:
            worker.shutdown()
            raise
        self.worker_handles = (LocalWorkerHandle(0, worker),)

    def shutdown(self) -> None:
        for handle in self.worker_handles:
            if isinstance(handle, LocalWorkerHandle):
                handle.worker.shutdown()
        self.worker_handles = ()
