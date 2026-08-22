"""Distributed topology and process-local communication groups."""

from .communication import CollectiveGroup, SingletonCollectiveGroup
from .parallel_context import ParallelContext, materialize_parallel_context
from .topology import (
    ParallelGroupKind,
    ParallelGroupSpec,
    ParallelTopology,
    RankAssignment,
    RankCoordinates,
    RunnerGroupSpec,
)

__all__ = [
    "CollectiveGroup",
    "ParallelContext",
    "ParallelGroupKind",
    "ParallelGroupSpec",
    "ParallelTopology",
    "RankAssignment",
    "RankCoordinates",
    "RunnerGroupSpec",
    "SingletonCollectiveGroup",
    "materialize_parallel_context",
]
