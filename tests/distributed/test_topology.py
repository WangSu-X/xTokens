from __future__ import annotations

import pytest

from x_tokens.config import ParallelConfig
from x_tokens.distributed import ParallelGroupKind, ParallelTopology


def mp_config(**overrides: object) -> ParallelConfig:
    values: dict[str, object] = {
        "distributed_executor_backend": "mp",
        "devices": tuple(f"cpu:{index}" for index in range(8)),
        "tensor_parallel_size": 2,
        "pipeline_parallel_size": 2,
        "context_parallel_size": 1,
        "data_parallel_size": 2,
        "collective_backend": "gloo",
    }
    values.update(overrides)
    return ParallelConfig(**values)  # type: ignore[arg-type]


def groups(
    topology: ParallelTopology, kind: ParallelGroupKind
) -> tuple[tuple[int, ...], ...]:
    return tuple(spec.ranks for spec in topology.group_specs if spec.kind is kind)


def test_builds_dp_pp_cp_tp_rank_mesh_and_runner_groups() -> None:
    topology = ParallelTopology.build(mp_config())

    assert topology.world_size == 8
    assert groups(topology, ParallelGroupKind.TENSOR) == (
        (0, 1),
        (2, 3),
        (4, 5),
        (6, 7),
    )
    assert groups(topology, ParallelGroupKind.PIPELINE) == (
        (0, 2),
        (1, 3),
        (4, 6),
        (5, 7),
    )
    assert groups(topology, ParallelGroupKind.DATA) == (
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    )
    assert topology.assignment(7).coordinates.tp_rank == 1
    assert topology.assignment(7).coordinates.pp_rank == 1
    assert topology.assignment(7).coordinates.dp_rank == 1
    assert topology.runner_group_specs[0].ranks == (0, 1, 2, 3)
    assert topology.runner_group_specs[0].output_rank == 2
    assert topology.runner_group_specs[1].ranks == (4, 5, 6, 7)
    assert topology.runner_group_specs[1].output_rank == 6


def test_builds_context_and_expert_overlay_groups() -> None:
    config = mp_config(
        devices=tuple(f"cpu:{index}" for index in range(4)),
        pipeline_parallel_size=1,
        context_parallel_size=2,
        data_parallel_size=1,
        enable_expert_parallel=True,
        expert_parallel_size=4,
    )
    topology = ParallelTopology.build(config)

    assert groups(topology, ParallelGroupKind.CONTEXT) == ((0, 2), (1, 3))
    assert groups(topology, ParallelGroupKind.EXPERT) == ((0, 1, 2, 3),)


def test_parallel_config_rejects_ambiguous_or_invalid_layouts() -> None:
    with pytest.raises(ValueError, match="uni executor"):
        ParallelConfig(devices=("cuda:0", "cuda:1"), tensor_parallel_size=2)
    with pytest.raises(ValueError, match="exactly one"):
        mp_config(devices=("cpu:0",))
    with pytest.raises(ValueError, match="duplicates"):
        mp_config(devices=("cpu",) * 8)
    with pytest.raises(ValueError, match="requires"):
        ParallelConfig(expert_parallel_size=1)
