"""In-process ModelExecutor backend."""

from __future__ import annotations

from x_tokens.worker import GPUWorker, GPUWorkerConfig

from .model_executor import ModelExecutor, RunnerGroup
from .worker_handle import LocalWorkerHandle


class UniProcModelExecutor(ModelExecutor):
    def _init_executor(self) -> None:
        if self.parallel_config.distributed_executor_backend != "uni":
            raise ValueError("UniProcModelExecutor requires the uni backend")
        spec = self.topology.runner_group_specs[0]
        worker = GPUWorker(
            GPUWorkerConfig(
                self.parallel_config,
                self.topology,
                rank=0,
                distributed_init_method="env://",
                output_rank=spec.output_rank,
            ),
            self.runner_factory,
        )
        worker.init_device()
        try:
            worker.load_model()
        except Exception:
            worker.shutdown()
            raise
        self.runner_groups = (RunnerGroup(spec, (LocalWorkerHandle(0, worker),)),)

    def shutdown(self) -> None:
        if not self.runner_groups:
            return
        group = self.runner_groups[0]
        for handle in group.worker_handles:
            if isinstance(handle, LocalWorkerHandle):
                handle.worker.shutdown()
        self.runner_groups = ()
