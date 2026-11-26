"""Group-scoped collective communication contract."""

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
