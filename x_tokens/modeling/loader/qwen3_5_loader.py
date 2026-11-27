"""Load rank-local dense and block-scaled FP8 Qwen3.5 text weights."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import TYPE_CHECKING

import torch
from safetensors import safe_open
from torch import nn

from x_tokens.distributed import ParallelGroupKind
from x_tokens.modeling.layers import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)

from .safetensors_loader import WeightLoadReport, _load_manifest, _safe_checkpoint_path

if TYPE_CHECKING:
    from x_tokens.modeling.models.qwen3_5 import Qwen3_5ForCausalLM


_DENSE_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def _copy_projection(
    target: torch.Tensor,
    source: torch.Tensor,
    bounds: tuple[slice, ...],
    scales: torch.Tensor | None,
    block_size: tuple[int, int] | None,
) -> None:
    """Copy weight tensor into target, handling FP8 block-scaled quantization.

    For FP8-quantized weights:
    1. Align requested bounds to tile boundaries (tile_rows × tile_columns)
    2. Load and dequantize the aligned region to float32
    3. Scale by per-tile scale factors (inverse scales stored in checkpoint)
    4. Extract the originally requested slice

    For dense weights (bfloat16/float16/float32):
    1. Direct copy via bounds indexing

    This two-stage process (align→dequantize→scale→slice) ensures that:
    - FP8 tiles are decoded together (tiles are atomic quantization units)
    - Split boundaries within tiles are handled correctly
    - Per-rank tensor parallel shards receive properly scaled weights

    Args:
        target: Destination tensor (rank-local parameter shard)
        source: Source tensor from checkpoint (quantized or dense)
        bounds: Slice tuple [row_slice, col_slice] indicating which portion to copy
        scales: Per-tile scale factors (1D or 2D, only for FP8 source)
        block_size: (tile_rows, tile_columns) FP8 quantization block size

    Raises:
        ValueError: If FP8 matrix lacks required scales or has shape mismatch
    """
    if source.dtype == torch.float8_e4m3fn:
        if scales is None or block_size is None or source.ndim != 2:
            raise ValueError("FP8 matrix requires per-block weight_scale_inv")
        rows, columns = source.shape
        tile_rows, tile_columns = block_size
        expected_scales = (
            math.ceil(rows / tile_rows),
            math.ceil(columns / tile_columns),
        )
        if tuple(scales.shape) != expected_scales:
            raise ValueError(f"FP8 scale shape mismatch: expected {expected_scales}")

        # Decode only the local shard. Align to tiles first to support split
        # boundaries that fall within an FP8 tile.
        row_start, row_end, _ = bounds[0].indices(rows)
        col_start, col_end, _ = bounds[1].indices(columns)
        first_row, last_row = row_start // tile_rows, math.ceil(row_end / tile_rows)
        first_col, last_col = (
            col_start // tile_columns,
            math.ceil(col_end / tile_columns),
        )
        aligned_row, aligned_col = first_row * tile_rows, first_col * tile_columns
        weights = (
            source[
                aligned_row : last_row * tile_rows,
                aligned_col : last_col * tile_columns,
            ]
            .contiguous()
            .to(device=target.device)
            .float()
        )
        weight_scales = scales[first_row:last_row, first_col:last_col].to(
            device=target.device, dtype=torch.float32
        )
        height, width = weights.shape
        if height % tile_rows or width % tile_columns:
            expanded = weight_scales.repeat_interleave(tile_rows, 0).repeat_interleave(
                tile_columns, 1
            )
            weights.mul_(expanded[:height, :width])
        else:
            weights.view(
                last_row - first_row, tile_rows, last_col - first_col, tile_columns
            ).mul_(weight_scales[:, None, :, None])
        target.copy_(
            weights[
                row_start - aligned_row : row_end - aligned_row,
                col_start - aligned_col : col_end - aligned_col,
            ]
        )
    else:
        if source.dtype not in _DENSE_DTYPES or scales is not None:
            raise ValueError(
                f"unsupported Qwen3.5 weight dtype or scales: {source.dtype}"
            )
        target.copy_(source[bounds])


def _copy_parameter(
    name: str,
    parameter: torch.Tensor,
    module: nn.Module,
    source: torch.Tensor,
    rank: int,
    tp: int,
    scales: torch.Tensor | None,
    block_size: tuple[int, int] | None,
    conv_segments: tuple[int, int, int],
) -> None:
    """Copy and shard checkpoint weights into rank-local parameter.

    Handles multiple layer types with different sharding strategies:

    - VocabParallelEmbedding: Load full embedding (must be unquantized)
    - ColumnParallelLinear: Shard by output dimension (axis=0)
    - RowParallelLinear: Shard by input dimension (axis=1)
    - MergedColumnParallelLinear: Multiple output sizes (e.g., QKV combined)
    - 1D Convolution: Multiple segments (K, K, V for gated delta net)
    - Scalar parameters (A_log, dt_bias): Per-head sharding

    For tensor parallel (TP) execution:
    1. Identify sharding axis and segments from module type
    2. Validate source shape matches total (unsharded) parameter size
    3. Calculate this rank's slice within each segment
    4. Copy with FP8 dequantization if quantized

    Args:
        name: Parameter name (e.g., "model.layers.0.self_attn.q_proj.weight")
        parameter: Local rank's parameter tensor (pre-allocated, uninitialized)
        module: nn.Module containing the parameter (determines sharding strategy)
        source: Full parameter from checkpoint (all ranks' shards concatenated)
        rank: This rank's index in tensor parallel group (0 to tp-1)
        tp: Tensor parallel world size
        scales: Per-tile scale factors if source is FP8-quantized
        block_size: (tile_rows, tile_columns) for FP8 quantization
        conv_segments: (K_width, K_width, V_width) for Conv1d sharding

    Raises:
        ValueError: If embedding is quantized, or shape mismatch, or unsupported module
    """
    if isinstance(module, VocabParallelEmbedding):
        if scales is not None or source.dtype not in _DENSE_DTYPES:
            raise ValueError(
                "Qwen3.5 embedding and LM head weights must be unquantized"
            )
        module.load_weight(source)
        return

    axis = 0
    segments: tuple[int, ...] = ()
    if isinstance(module, MergedColumnParallelLinear):
        segments = module.output_sizes
    elif isinstance(module, ColumnParallelLinear) or name.endswith(
        (".A_log", ".dt_bias")
    ):
        segments = (parameter.shape[0] * tp,)
    elif isinstance(module, RowParallelLinear):
        axis = 1
        segments = (parameter.shape[1] * tp,)
    elif name.endswith(".conv1d.weight"):
        segments = conv_segments

    expected = list(parameter.shape)
    if segments:
        expected[axis] = sum(segments)
    if tuple(source.shape) != tuple(expected):
        raise ValueError(
            f"{name} shape mismatch: expected {tuple(expected)}, got {tuple(source.shape)}"
        )
    if not segments:
        _copy_projection(
            parameter, source, (slice(None),) * source.ndim, scales, block_size
        )
        return
    offset = 0
    destination_offset = 0
    shards = []
    for segment in segments:
        size = segment // tp
        start = offset + rank * size
        bounds = [slice(None)] * source.ndim
        bounds[axis] = slice(start, start + size)
        shards.append(tuple(bounds))
        offset += segment
    for segment, bounds in zip(segments, shards, strict=True):
        size = segment // tp
        _copy_projection(
            parameter.narrow(axis, destination_offset, size),
            source,
            bounds,
            scales,
            block_size,
        )
        destination_offset += size


@torch.no_grad()
def load_qwen3_5_weights(model: Qwen3_5ForCausalLM) -> WeightLoadReport:
    """Load rank-local Qwen3.5 weights from safetensors checkpoint with FP8 support.

    Load flow:
    1. Build parameter manifest from checkpoint (safetensors files)
    2. Validate that all model parameters have corresponding checkpoint weights
    3. Flag extra/missing parameters (optional strict mode)
    4. Group parameters by checkpoint file for efficient I/O
    5. For each file:
       a. Load checkpoint shard into CPU memory
       b. Load per-tile scales if present (for FP8 quantized weights)
       c. For each parameter: determine sharding, copy with _copy_parameter()
    6. Return load statistics (bytes, tensor count, rank info)

    FP8 Quantization:
    - Weights may be stored as float8_e4m3fn (8-bit) with per-tile scales
    - Scales stored as separate tensors: {param_name}_scale_inv
    - Tiles are atomic units (cannot split across rank boundaries)
    - On load: dequantize tile-aligned regions, then slice to rank's shard

    Tensor Parallelism:
    - Parameters are sharded by output/input dimension based on layer type
    - Each rank loads its local shard (size ÷ tp)
    - Collective barrier after load ensures all ranks finish before training

    Args:
        model: Qwen3_5ForCausalLM instance with load_config and distributed context

    Returns:
        WeightLoadReport: Statistics including:
        - rank, world_size: Rank coordinates
        - loaded_source_tensors: Count (parameters + scales)
        - loaded_parameters: Count of model parameters initialized
        - local_parameter_bytes: Memory consumption
        - missing_parameters, unexpected_tensors: Validation results

    Raises:
        ValueError: If strict_weights=True and mismatches detected
        FileNotFoundError: If checkpoint files missing
    """
    root = model.load_config.model_path
    manifest = _load_manifest(root)
    prefix = (
        "model.language_model"
        if "model.language_model.embed_tokens.weight" in manifest
        else "model"
    )
    weights = {
        name.replace("model.", f"{prefix}.", 1): (name, parameter)
        for name, parameter in model.named_parameters()
    }
    missing = tuple(sorted(weights.keys() - manifest.keys()))
    expected_scales = {f"{name}_scale_inv" for name in weights}
    permitted_extras = {"lm_head.weight"} if model.config.tie_word_embeddings else set()
    unexpected = tuple(
        sorted(
            key
            for key in manifest.keys()
            - weights.keys()
            - expected_scales
            - permitted_extras
            if not key.startswith(
                ("model.visual.", "visual.", "mtp.", "model.mtp.", f"{prefix}.mtp.")
            )
        )
    )
    if model.load_config.strict_weights and (missing or unexpected):
        raise ValueError(
            f"Qwen3.5 checkpoint validation failed: missing={missing[:8]}, unexpected={unexpected[:8]}"
        )

    group = model.context.group(ParallelGroupKind.TENSOR)
    config = model.config
    key_width = config.linear_num_key_heads * config.linear_key_head_dim
    value_width = config.linear_num_value_heads * config.linear_value_head_dim
    conv_segments = (key_width, key_width, value_width)
    loaded = 0
    by_file: dict[str, list[str]] = defaultdict(list)
    for key in sorted(weights.keys() & manifest.keys()):
        by_file[manifest[key]].append(key)
    for filename, keys in by_file.items():
        with safe_open(
            _safe_checkpoint_path(root, filename), framework="pt", device="cpu"
        ) as shard:
            for key in keys:
                name, parameter = weights[key]
                module_name, _, _ = name.rpartition(".")
                module = model.get_submodule(module_name)
                scale_key = f"{key}_scale_inv"
                scale_file = manifest.get(scale_key)
                scales = None
                if scale_file == filename:
                    scales = shard.get_tensor(scale_key)
                elif scale_file is not None:
                    with safe_open(
                        _safe_checkpoint_path(root, scale_file),
                        framework="pt",
                        device="cpu",
                    ) as scale_shard:
                        scales = scale_shard.get_tensor(scale_key)
                _copy_parameter(
                    name,
                    parameter,
                    module,
                    shard.get_tensor(key),
                    group.rank_in_group,
                    group.world_size,
                    scales,
                    config.weight_block_size,
                    conv_segments,
                )
                loaded += 1 + int(scales is not None)
    parameters = tuple(model.parameters())
    return WeightLoadReport(
        rank=model.context.coordinates.global_rank,
        world_size=model.context.group(ParallelGroupKind.WORLD).world_size,
        loaded_source_tensors=loaded,
        loaded_parameters=len(weights) - len(missing),
        local_parameter_bytes=sum(p.numel() * p.element_size() for p in parameters),
        missing_parameters=missing,
        unexpected_tensors=unexpected,
    )
