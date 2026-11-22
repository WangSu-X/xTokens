"""Group-scoped communication interfaces and torch.distributed adapters."""

from __future__ import annotations

from typing import Any, Protocol


class CollectiveGroup(Protocol):
    rank_in_group: int
    world_size: int

    def all_reduce(self, tensor: Any) -> Any: ...

    def all_gather(self, tensor: Any, dim: int = 0) -> Any: ...

    def broadcast(self, tensor: Any, src: int = 0) -> Any: ...

    def send(self, tensor: Any, dst: int) -> None: ...

    def recv(self, tensor: Any, src: int) -> None: ...

    def barrier(self) -> None: ...


class SingletonCollectiveGroup(CollectiveGroup):
    """No-op implementation preserving the group API for size-one meshes."""

    rank_in_group = 0
    world_size = 1

    def all_reduce(self, tensor: Any) -> Any:
        return tensor

    def all_gather(self, tensor: Any, dim: int = 0) -> Any:
        del dim
        return tensor

    def broadcast(self, tensor: Any, src: int = 0) -> Any:
        if src != 0:
            raise ValueError("singleton group source rank must be zero")
        return tensor

    def send(self, tensor: Any, dst: int) -> None:
        del tensor, dst
        raise RuntimeError("send is invalid for a singleton group")

    def recv(self, tensor: Any, src: int) -> None:
        del tensor, src
        raise RuntimeError("recv is invalid for a singleton group")

    def barrier(self) -> None:
        return


class TorchDistributedCollectiveGroup(CollectiveGroup):
    """A process-local adapter around a torch distributed process group."""

    def __init__(
        self,
        process_group: Any,
        ranks: tuple[int, ...],
        global_rank: int,
        *,
        torch_module: Any,
    ) -> None:
        self._process_group = process_group
        self._ranks = ranks
        self._torch = torch_module
        self.rank_in_group = ranks.index(global_rank)
        self.world_size = len(ranks)

    def all_reduce(self, tensor: Any) -> Any:
        with self._torch.inference_mode(False):
            output = tensor.clone()
            self._torch.distributed.all_reduce(output, group=self._process_group)
        return output

    def all_gather(self, tensor: Any, dim: int = 0) -> Any:
        with self._torch.inference_mode(False):
            input_tensor = tensor.clone()
            outputs = [
                self._torch.empty_like(input_tensor) for _ in range(self.world_size)
            ]
            self._torch.distributed.all_gather(
                outputs, input_tensor, group=self._process_group
            )
            return self._torch.cat(outputs, dim=dim)

    def broadcast(self, tensor: Any, src: int = 0) -> Any:
        self._validate_group_rank(src)
        with self._torch.inference_mode(False):
            output = tensor.clone()
            self._torch.distributed.broadcast(
                output, src=self._ranks[src], group=self._process_group
            )
        return output

    def send(self, tensor: Any, dst: int) -> None:
        self._validate_group_rank(dst)
        self._torch.distributed.send(
            tensor, dst=self._ranks[dst], group=self._process_group
        )

    def recv(self, tensor: Any, src: int) -> None:
        self._validate_group_rank(src)
        self._torch.distributed.recv(
            tensor, src=self._ranks[src], group=self._process_group
        )

    def barrier(self) -> None:
        self._torch.distributed.barrier(group=self._process_group)

    def _validate_group_rank(self, rank: int) -> None:
        if rank < 0 or rank >= self.world_size:
            raise ValueError(f"group rank {rank} is outside size {self.world_size}")
