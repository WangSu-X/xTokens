"""Distributed topology and process-local communication groups."""

from .communication import SingletonCollectiveGroup
from .parallel_context import ParallelContext, materialize_parallel_context
from .protocol import CollectiveGroup
from .topology import (
    ParallelGroupKind,
    ParallelGroupSpec,
    ParallelTopology,
    RankAssignment,
    RankCoordinates,
)

__all__ = [
    "CollectiveGroup",
    "ParallelContext",
    "ParallelGroupKind",
    "ParallelGroupSpec",
    "ParallelTopology",
    "RankAssignment",
    "RankCoordinates",
    "SingletonCollectiveGroup",
    "materialize_parallel_context",
]
