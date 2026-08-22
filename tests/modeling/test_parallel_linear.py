from __future__ import annotations

import torch
import torch.nn.functional as F

from x_tokens.modeling.layers import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)


class FakeGroup:
    def __init__(self, rank: int, world_size: int) -> None:
        self.rank_in_group = rank
        self.world_size = world_size

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    def all_gather(self, tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        del dim
        return tensor


def test_column_and_row_shards_reconstruct_dense_linear() -> None:
    torch.manual_seed(0)
    inputs = torch.randn(3, 6)
    weight = torch.randn(8, 6)
    bias = torch.randn(8)

    column_outputs = []
    row_outputs = []
    for rank in range(2):
        group = FakeGroup(rank, 2)
        column = ColumnParallelLinear(6, 8, group=group, bias=False)
        column.load_weight(weight)
        column_outputs.append(column(inputs))

        row = RowParallelLinear(
            6,
            8,
            group=group,
            bias=True,
            input_is_parallel=False,
            reduce_results=False,
        )
        row.load_weight(weight)
        row.bias.data.copy_(bias)
        row_outputs.append(row(inputs))

    assert torch.allclose(torch.cat(column_outputs, dim=-1), F.linear(inputs, weight))
    assert torch.allclose(sum(row_outputs), F.linear(inputs, weight, bias))


def test_merged_column_shards_each_projection_independently() -> None:
    gate = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    up = torch.arange(36, dtype=torch.float32).reshape(6, 6) + 100
    layers = [
        MergedColumnParallelLinear(6, (4, 6), group=FakeGroup(rank, 2), bias=False)
        for rank in range(2)
    ]
    for layer in layers:
        layer.load_weight(gate, 0)
        layer.load_weight(up, 1)

    assert torch.equal(layers[0].weight[:2], gate[:2])
    assert torch.equal(layers[0].weight[2:], up[:3])
    assert torch.equal(layers[1].weight[:2], gate[2:])
    assert torch.equal(layers[1].weight[2:], up[3:])


def test_qkv_loader_replicates_kv_heads_when_tp_is_larger() -> None:
    q = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    k = torch.arange(16, dtype=torch.float32).reshape(2, 8) + 100
    v = k + 100
    layers = [
        QKVParallelLinear(
            8,
            head_dim=2,
            total_num_heads=4,
            total_num_kv_heads=1,
            group=FakeGroup(rank, 2),
            bias=False,
        )
        for rank in range(2)
    ]
    for layer in layers:
        layer.load_weight(q, "q")
        layer.load_weight(k, "k")
        layer.load_weight(v, "v")

    assert torch.equal(layers[0].weight[:4], q[:4])
    assert torch.equal(layers[1].weight[:4], q[4:])
    assert torch.equal(layers[0].weight[4:6], k)
    assert torch.equal(layers[1].weight[4:6], k)
    assert torch.equal(layers[0].weight[6:], v)
    assert torch.equal(layers[1].weight[6:], v)
