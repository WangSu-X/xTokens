"""Materialize process-local communication groups from a pure topology."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .communication import (
    CollectiveGroup,
    SingletonCollectiveGroup,
    TorchDistributedCollectiveGroup,
)
from .topology import ParallelGroupKind, ParallelTopology, RankCoordinates


@dataclass(frozen=True, slots=True)
class ParallelContext:
    coordinates: RankCoordinates
    device: Any
    _groups: Mapping[ParallelGroupKind, CollectiveGroup]

    @classmethod
    def create(
        cls,
        coordinates: RankCoordinates,
        device: Any,
        groups: Mapping[ParallelGroupKind, CollectiveGroup],
    ) -> ParallelContext:
        return cls(coordinates, device, MappingProxyType(dict(groups)))

    def group(self, kind: ParallelGroupKind) -> CollectiveGroup:
        try:
            return self._groups[kind]
        except KeyError as error:
            raise RuntimeError(
                f"parallel group {kind.value} is not initialized"
            ) from error


def materialize_parallel_context(
    topology: ParallelTopology,
    rank: int,
    *,
    torch_module: Any,
) -> ParallelContext:
    """Create groups in deterministic order and retain this rank's handles."""

    assignment = topology.assignment(rank)
    groups: dict[ParallelGroupKind, CollectiveGroup] = {}
    for spec in topology.group_specs:
        if len(spec.ranks) == 1:
            process_group = None
        elif spec.kind is ParallelGroupKind.WORLD:
            process_group = torch_module.distributed.group.WORLD
        else:
            process_group = torch_module.distributed.new_group(ranks=list(spec.ranks))
        if rank not in spec.ranks:
            continue
        if len(spec.ranks) == 1:
            groups[spec.kind] = SingletonCollectiveGroup()
        else:
            groups[spec.kind] = TorchDistributedCollectiveGroup(
                process_group,
                spec.ranks,
                rank,
                torch_module=torch_module,
            )
    return ParallelContext.create(
        assignment.coordinates,
        torch_module.device(assignment.device),
        groups,
    )
