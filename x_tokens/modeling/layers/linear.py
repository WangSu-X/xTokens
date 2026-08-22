"""Group-scoped tensor-parallel linear layers."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from x_tokens.distributed import CollectiveGroup


def _require_divisible(value: int, divisor: int, name: str) -> None:
    if value % divisor:
        raise ValueError(f"{name}={value} must be divisible by group size {divisor}")


class ReplicatedLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        initialize: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, **factory_kwargs)
        )
        self.bias = (
            nn.Parameter(torch.empty(out_features, **factory_kwargs)) if bias else None
        )
        if initialize:
            self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.linear(inputs, self.weight, self.bias)

    def load_weight(self, loaded_weight: torch.Tensor) -> None:
        if loaded_weight.shape != self.weight.shape:
            raise ValueError(
                f"weight shape mismatch: expected {self.weight.shape}, "
                f"got {loaded_weight.shape}"
            )
        self.weight.data.copy_(loaded_weight)


class ColumnParallelLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        group: CollectiveGroup,
        bias: bool,
        gather_output: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        initialize: bool = True,
    ) -> None:
        super().__init__()
        _require_divisible(out_features, group.world_size, "out_features")
        self.in_features = in_features
        self.out_features = out_features
        self.output_size_per_rank = out_features // group.world_size
        self.group = group
        self.gather_output = gather_output
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(
            torch.empty(self.output_size_per_rank, in_features, **factory_kwargs)
        )
        self.bias = (
            nn.Parameter(torch.empty(self.output_size_per_rank, **factory_kwargs))
            if bias
            else None
        )
        if initialize:
            self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = F.linear(inputs, self.weight, self.bias)
        return self.group.all_gather(output, dim=-1) if self.gather_output else output

    def load_weight(self, loaded_weight: torch.Tensor) -> None:
        expected = (self.out_features, self.in_features)
        if tuple(loaded_weight.shape) != expected:
            raise ValueError(
                f"weight shape mismatch: expected {expected}, "
                f"got {tuple(loaded_weight.shape)}"
            )
        start = self.group.rank_in_group * self.output_size_per_rank
        self.weight.data.copy_(
            loaded_weight.narrow(0, start, self.output_size_per_rank)
        )


class MergedColumnParallelLinear(ColumnParallelLinear):
    def __init__(
        self,
        in_features: int,
        output_sizes: tuple[int, ...],
        *,
        group: CollectiveGroup,
        bias: bool,
        gather_output: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        initialize: bool = True,
    ) -> None:
        if not output_sizes:
            raise ValueError("output_sizes must not be empty")
        for index, size in enumerate(output_sizes):
            _require_divisible(size, group.world_size, f"output_sizes[{index}]")
        self.output_sizes = output_sizes
        super().__init__(
            in_features,
            sum(output_sizes),
            group=group,
            bias=bias,
            gather_output=gather_output,
            device=device,
            dtype=dtype,
            initialize=initialize,
        )

    def load_weight(self, loaded_weight: torch.Tensor, shard_id: int) -> None:
        if shard_id < 0 or shard_id >= len(self.output_sizes):
            raise ValueError(f"invalid packed shard ID: {shard_id}")
        global_size = self.output_sizes[shard_id]
        expected = (global_size, self.in_features)
        if tuple(loaded_weight.shape) != expected:
            raise ValueError(
                f"weight shape mismatch: expected {expected}, "
                f"got {tuple(loaded_weight.shape)}"
            )
        local_sizes = tuple(size // self.group.world_size for size in self.output_sizes)
        target_offset = sum(local_sizes[:shard_id])
        source_offset = self.group.rank_in_group * local_sizes[shard_id]
        self.weight.data.narrow(0, target_offset, local_sizes[shard_id]).copy_(
            loaded_weight.narrow(0, source_offset, local_sizes[shard_id])
        )


class QKVParallelLinear(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        total_num_heads: int,
        total_num_kv_heads: int,
        *,
        group: CollectiveGroup,
        bias: bool,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        initialize: bool = True,
    ) -> None:
        super().__init__()
        _require_divisible(total_num_heads, group.world_size, "total_num_heads")
        if total_num_kv_heads >= group.world_size:
            _require_divisible(
                total_num_kv_heads, group.world_size, "total_num_kv_heads"
            )
        elif group.world_size % total_num_kv_heads:
            raise ValueError(
                "group size must be divisible by total_num_kv_heads when KV "
                "heads are replicated"
            )
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        self.group = group
        self.local_num_heads = total_num_heads // group.world_size
        self.local_num_kv_heads = max(1, total_num_kv_heads // group.world_size)
        self.local_output_sizes = (
            self.local_num_heads * head_dim,
            self.local_num_kv_heads * head_dim,
            self.local_num_kv_heads * head_dim,
        )
        local_output_size = sum(self.local_output_sizes)
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(
            torch.empty(local_output_size, hidden_size, **factory_kwargs)
        )
        self.bias = (
            nn.Parameter(torch.empty(local_output_size, **factory_kwargs))
            if bias
            else None
        )
        if initialize:
            self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.hidden_size)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.linear(inputs, self.weight, self.bias)

    def load_weight(self, loaded_weight: torch.Tensor, shard_id: str) -> None:
        shard_index = {"q": 0, "k": 1, "v": 2}.get(shard_id)
        if shard_index is None:
            raise ValueError(f"invalid QKV shard ID: {shard_id}")
        total_heads = (
            self.total_num_heads if shard_id == "q" else self.total_num_kv_heads
        )
        expected = (total_heads * self.head_dim, self.hidden_size)
        if tuple(loaded_weight.shape) != expected:
            raise ValueError(
                f"weight shape mismatch: expected {expected}, "
                f"got {tuple(loaded_weight.shape)}"
            )
        local_size = self.local_output_sizes[shard_index]
        target_offset = sum(self.local_output_sizes[:shard_index])
        if shard_id != "q" and self.total_num_kv_heads < self.group.world_size:
            replicas = self.group.world_size // self.total_num_kv_heads
            head_index = self.group.rank_in_group // replicas
            source_offset = head_index * self.head_dim
        else:
            source_offset = self.group.rank_in_group * local_size
        self.weight.data.narrow(0, target_offset, local_size).copy_(
            loaded_weight.narrow(0, source_offset, local_size)
        )


class RowParallelLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        group: CollectiveGroup,
        bias: bool,
        input_is_parallel: bool = True,
        reduce_results: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        initialize: bool = True,
    ) -> None:
        super().__init__()
        _require_divisible(in_features, group.world_size, "in_features")
        self.in_features = in_features
        self.out_features = out_features
        self.input_size_per_rank = in_features // group.world_size
        self.group = group
        self.input_is_parallel = input_is_parallel
        self.reduce_results = reduce_results
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(
            torch.empty(out_features, self.input_size_per_rank, **factory_kwargs)
        )
        self.bias = (
            nn.Parameter(torch.empty(out_features, **factory_kwargs)) if bias else None
        )
        if initialize:
            self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not self.input_is_parallel:
            inputs = inputs.narrow(
                -1,
                self.group.rank_in_group * self.input_size_per_rank,
                self.input_size_per_rank,
            )
        output = F.linear(inputs, self.weight, None)
        if self.reduce_results:
            output = self.group.all_reduce(output)
        if self.bias is not None and (
            self.reduce_results or self.group.rank_in_group == 0
        ):
            output = output + self.bias
        return output

    def load_weight(self, loaded_weight: torch.Tensor) -> None:
        expected = (self.out_features, self.in_features)
        if tuple(loaded_weight.shape) != expected:
            raise ValueError(
                f"weight shape mismatch: expected {expected}, "
                f"got {tuple(loaded_weight.shape)}"
            )
        start = self.group.rank_in_group * self.input_size_per_rank
        self.weight.data.copy_(loaded_weight.narrow(1, start, self.input_size_per_rank))


def set_parameter_data(parameter: nn.Parameter, value: Any) -> None:
    """Copy a tensor-like value while keeping the registered Parameter object."""

    tensor = torch.as_tensor(value, device=parameter.device, dtype=parameter.dtype)
    if tensor.shape != parameter.shape:
        raise ValueError(
            f"parameter shape mismatch: expected {parameter.shape}, got {tensor.shape}"
        )
    parameter.data.copy_(tensor)
