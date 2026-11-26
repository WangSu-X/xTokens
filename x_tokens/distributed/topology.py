"""Pure-data construction of model-parallel rank topology."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum

from x_tokens.config import ParallelConfig


class ParallelGroupKind(StrEnum):
    WORLD = "world"
    TENSOR = "tp"
    PIPELINE = "pp"
    CONTEXT = "cp"
    EXPERT = "ep"


@dataclass(frozen=True, slots=True)
class RankCoordinates:
    global_rank: int
    local_rank: int
    tp_rank: int
    pp_rank: int
    cp_rank: int


@dataclass(frozen=True, slots=True)
class RankAssignment:
    coordinates: RankCoordinates
    device: str


@dataclass(frozen=True, slots=True)
class ParallelGroupSpec:
    kind: ParallelGroupKind
    group_index: int
    ranks: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ParallelTopology:
    world_size: int
    rank_assignments: tuple[RankAssignment, ...]
    group_specs: tuple[ParallelGroupSpec, ...]

    @classmethod
    def build(cls, config: ParallelConfig) -> ParallelTopology:
        assignments = tuple(_assignments(config))
        specs: list[ParallelGroupSpec] = [
            ParallelGroupSpec(
                ParallelGroupKind.WORLD,
                0,
                tuple(range(config.world_size)),
            )
        ]
        specs.extend(_axis_groups(config, ParallelGroupKind.TENSOR))
        specs.extend(_axis_groups(config, ParallelGroupKind.PIPELINE))
        specs.extend(_axis_groups(config, ParallelGroupKind.CONTEXT))
        if config.enable_expert_parallel:
            specs.extend(_expert_groups(config))
        return cls(config.world_size, assignments, tuple(specs))

    def assignment(self, rank: int) -> RankAssignment:
        if rank < 0 or rank >= self.world_size:
            raise ValueError(f"rank {rank} is outside world_size {self.world_size}")
        return self.rank_assignments[rank]


def _rank(config: ParallelConfig, *, pp: int, cp: int, tp: int) -> int:
    return (pp * config.context_parallel_size + cp) * config.tensor_parallel_size + tp


def _assignments(config: ParallelConfig) -> Iterator[RankAssignment]:
    for rank, device in enumerate(config.devices):
        stage_rank, tp_rank = divmod(rank, config.tensor_parallel_size)
        pp_rank, cp_rank = divmod(stage_rank, config.context_parallel_size)
        yield RankAssignment(
            RankCoordinates(rank, rank, tp_rank, pp_rank, cp_rank),
            device,
        )


def _axis_groups(
    config: ParallelConfig, kind: ParallelGroupKind
) -> list[ParallelGroupSpec]:
    if kind is ParallelGroupKind.TENSOR:
        groups = [
            tuple(
                _rank(config, pp=pp, cp=cp, tp=tp)
                for tp in range(config.tensor_parallel_size)
            )
            for pp in range(config.pipeline_parallel_size)
            for cp in range(config.context_parallel_size)
        ]
    elif kind is ParallelGroupKind.PIPELINE:
        groups = [
            tuple(
                _rank(config, pp=pp, cp=cp, tp=tp)
                for pp in range(config.pipeline_parallel_size)
            )
            for cp in range(config.context_parallel_size)
            for tp in range(config.tensor_parallel_size)
        ]
    elif kind is ParallelGroupKind.CONTEXT:
        groups = [
            tuple(
                _rank(config, pp=pp, cp=cp, tp=tp)
                for cp in range(config.context_parallel_size)
            )
            for pp in range(config.pipeline_parallel_size)
            for tp in range(config.tensor_parallel_size)
        ]
    else:
        raise ValueError(f"unsupported mesh axis: {kind}")
    return [ParallelGroupSpec(kind, index, ranks) for index, ranks in enumerate(groups)]


def _expert_groups(config: ParallelConfig) -> list[ParallelGroupSpec]:
    ep_size = config.expert_parallel_size or config.model_parallel_size
    groups: list[tuple[int, ...]] = []
    for pp in range(config.pipeline_parallel_size):
        stage_ranks = [
            _rank(config, pp=pp, cp=cp, tp=tp)
            for cp in range(config.context_parallel_size)
            for tp in range(config.tensor_parallel_size)
        ]
        groups.extend(
            tuple(stage_ranks[offset : offset + ep_size])
            for offset in range(0, len(stage_ranks), ep_size)
        )
    return [
        ParallelGroupSpec(ParallelGroupKind.EXPERT, index, ranks)
        for index, ranks in enumerate(groups)
    ]
