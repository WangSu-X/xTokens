"""Strict streaming SafeTensors loader for registered model architectures."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from safetensors import safe_open

from x_tokens.distributed import ParallelGroupKind

from .source import ResolvedModelSource

if TYPE_CHECKING:
    from x_tokens.modeling.models.qwen3_moe import Qwen3MoeForCausalLM


_LAYER_KEY = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
_LEGACY_EXPERT_KEY = re.compile(
    r"^mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"
)


@dataclass(frozen=True, slots=True)
class WeightLoadReport:
    rank: int
    world_size: int
    loaded_source_tensors: int
    loaded_parameters: int
    local_parameter_bytes: int
    missing_parameters: tuple[str, ...]
    unexpected_tensors: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SafeTensorsManifest:
    weight_map: dict[str, str]

    @property
    def keys(self) -> frozenset[str]:
        return frozenset(self.weight_map)


class SafeTensorsWeightLoader:
    def __init__(self, source: ResolvedModelSource, *, strict: bool = True) -> None:
        self.source = source
        self.strict = strict
        self.root = Path(source.path)
        self.manifest = _load_manifest(self.root)

    def load(self, model: Qwen3MoeForCausalLM) -> WeightLoadReport:
        expert_layout = _detect_expert_layout(model, self.manifest.keys)
        expected = _expected_source_keys(model, expert_layout, self.manifest.keys)
        missing = tuple(sorted(expected - self.manifest.keys))
        unexpected = tuple(sorted(self.manifest.keys - expected))
        if self.strict and (missing or unexpected):
            raise ValueError(
                "strict checkpoint validation failed: "
                f"missing={missing[:8]}, unexpected={unexpected[:8]}"
            )
        keys_to_load = expected & self.manifest.keys
        keys_by_file: dict[str, list[str]] = defaultdict(list)
        for key in sorted(keys_to_load):
            keys_by_file[self.manifest.weight_map[key]].append(key)
        loaded = 0
        for filename, keys in keys_by_file.items():
            checkpoint_path = _safe_checkpoint_path(self.root, filename)
            with safe_open(checkpoint_path, framework="pt", device="cpu") as shard:
                available = set(shard.keys())
                absent = [key for key in keys if key not in available]
                if absent:
                    raise ValueError(
                        f"checkpoint shard {filename} is missing indexed keys: {absent[:8]}"
                    )
                for key in keys:
                    _load_qwen3_weight(model, key, shard.get_tensor(key), expert_layout)
                    loaded += 1
        parameters = tuple(model.parameters())
        return WeightLoadReport(
            rank=model.context.coordinates.global_rank,
            world_size=model.context.group(ParallelGroupKind.WORLD).world_size,
            loaded_source_tensors=loaded,
            loaded_parameters=len(parameters),
            local_parameter_bytes=sum(
                parameter.numel() * parameter.element_size() for parameter in parameters
            ),
            missing_parameters=missing,
            unexpected_tensors=unexpected,
        )


def _load_manifest(root: Path) -> SafeTensorsManifest:
    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        with index_path.open(encoding="utf-8") as index_file:
            index = json.load(index_file)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("SafeTensors index must contain a non-empty weight_map")
        normalized = {str(key): str(value) for key, value in weight_map.items()}
        for filename in set(normalized.values()):
            _safe_checkpoint_path(root, filename)
        return SafeTensorsManifest(normalized)
    checkpoint_files = sorted(root.glob("*.safetensors"))
    if len(checkpoint_files) != 1:
        raise ValueError(
            "model directory without an index must contain exactly one SafeTensors file"
        )
    checkpoint = checkpoint_files[0]
    with safe_open(checkpoint, framework="pt", device="cpu") as shard:
        checkpoint_keys = shard.keys()
        return SafeTensorsManifest({key: checkpoint.name for key in checkpoint_keys})


def _safe_checkpoint_path(root: Path, filename: str) -> Path:
    path = (root / filename).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(
            f"checkpoint path escapes model directory: {filename}"
        ) from error
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint shard does not exist: {path}")
    return path


def _detect_expert_layout(
    model: Qwen3MoeForCausalLM, manifest_keys: frozenset[str]
) -> str:
    sparse_layers = [
        index
        for index in range(model.config.num_hidden_layers)
        if model.config.is_sparse_layer(index)
    ]
    if not sparse_layers:
        return "none"
    layer = sparse_layers[0]
    fused = f"model.layers.{layer}.mlp.experts.gate_up_proj" in manifest_keys
    legacy = f"model.layers.{layer}.mlp.experts.0.gate_proj.weight" in manifest_keys
    if fused == legacy:
        raise ValueError(
            "checkpoint must use exactly one expert schema: fused or per-expert"
        )
    return "fused" if fused else "legacy"


def _expected_source_keys(
    model: Qwen3MoeForCausalLM,
    expert_layout: str,
    manifest_keys: frozenset[str],
) -> frozenset[str]:
    config = model.config
    keys = {"model.embed_tokens.weight", "model.norm.weight"}
    if not config.tie_word_embeddings or "lm_head.weight" in manifest_keys:
        keys.add("lm_head.weight")
    for layer_index in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer_index}"
        keys.update(
            {
                f"{prefix}.self_attn.q_proj.weight",
                f"{prefix}.self_attn.k_proj.weight",
                f"{prefix}.self_attn.v_proj.weight",
                f"{prefix}.self_attn.o_proj.weight",
                f"{prefix}.self_attn.q_norm.weight",
                f"{prefix}.self_attn.k_norm.weight",
                f"{prefix}.input_layernorm.weight",
                f"{prefix}.post_attention_layernorm.weight",
            }
        )
        if config.is_sparse_layer(layer_index):
            keys.add(f"{prefix}.mlp.gate.weight")
            if expert_layout == "fused":
                keys.add(f"{prefix}.mlp.experts.gate_up_proj")
                keys.add(f"{prefix}.mlp.experts.down_proj")
            elif expert_layout == "legacy":
                for expert_index in range(config.num_experts):
                    expert_prefix = f"{prefix}.mlp.experts.{expert_index}"
                    keys.update(
                        {
                            f"{expert_prefix}.gate_proj.weight",
                            f"{expert_prefix}.up_proj.weight",
                            f"{expert_prefix}.down_proj.weight",
                        }
                    )
        else:
            keys.update(
                {
                    f"{prefix}.mlp.gate_proj.weight",
                    f"{prefix}.mlp.up_proj.weight",
                    f"{prefix}.mlp.down_proj.weight",
                }
            )
    return frozenset(keys)


def _load_qwen3_weight(
    model: Qwen3MoeForCausalLM,
    key: str,
    tensor: torch.Tensor,
    expert_layout: str,
) -> None:
    if key == "model.embed_tokens.weight":
        model.embed_tokens.load_weight(tensor)
        return
    if key == "lm_head.weight":
        model.lm_head.load_weight(tensor)
        return
    if key == "model.norm.weight":
        _copy_parameter(model.norm.weight, tensor, key)
        return
    match = _LAYER_KEY.match(key)
    if match is None:
        raise ValueError(f"unmapped Qwen3-MoE checkpoint key: {key}")
    layer_index = int(match.group(1))
    suffix = match.group(2)
    layer = model.layers[layer_index]
    direct = {
        "self_attn.q_norm.weight": layer.self_attn.q_norm.weight,
        "self_attn.k_norm.weight": layer.self_attn.k_norm.weight,
        "input_layernorm.weight": layer.input_layernorm.weight,
        "post_attention_layernorm.weight": layer.post_attention_layernorm.weight,
    }
    if suffix in direct:
        _copy_parameter(direct[suffix], tensor, key)
    elif suffix == "self_attn.q_proj.weight":
        layer.self_attn.qkv_proj.load_weight(tensor, "q")
    elif suffix == "self_attn.k_proj.weight":
        layer.self_attn.qkv_proj.load_weight(tensor, "k")
    elif suffix == "self_attn.v_proj.weight":
        layer.self_attn.qkv_proj.load_weight(tensor, "v")
    elif suffix == "self_attn.o_proj.weight":
        layer.self_attn.o_proj.load_weight(tensor)
    elif suffix == "mlp.gate.weight":
        _copy_parameter(layer.mlp.gate.weight, tensor, key)
    elif suffix == "mlp.gate_proj.weight":
        layer.mlp.gate_up_proj.load_weight(tensor, 0)
    elif suffix == "mlp.up_proj.weight":
        layer.mlp.gate_up_proj.load_weight(tensor, 1)
    elif suffix == "mlp.down_proj.weight":
        layer.mlp.down_proj.load_weight(tensor)
    elif expert_layout == "fused" and suffix == "mlp.experts.gate_up_proj":
        _load_fused_gate_up(layer.mlp.experts, tensor, key)
    elif expert_layout == "fused" and suffix == "mlp.experts.down_proj":
        _load_fused_down(layer.mlp.experts, tensor, key)
    else:
        expert_match = _LEGACY_EXPERT_KEY.match(suffix)
        if expert_layout != "legacy" or expert_match is None:
            raise ValueError(f"unmapped Qwen3-MoE checkpoint key: {key}")
        _load_legacy_expert(
            layer.mlp.experts,
            int(expert_match.group(1)),
            expert_match.group(2),
            tensor,
            key,
        )


def _load_fused_gate_up(experts: Any, tensor: torch.Tensor, key: str) -> None:
    intermediate = experts.local_intermediate_size * experts.group.world_size
    expected = (experts.num_experts, 2 * intermediate, experts.hidden_size)
    if tuple(tensor.shape) != expected:
        raise ValueError(
            f"{key} shape mismatch: expected {expected}, got {tuple(tensor.shape)}"
        )
    source_start = experts.group.rank_in_group * experts.local_intermediate_size
    local = experts.local_intermediate_size
    experts.gate_up_proj.data[:, :local].copy_(
        tensor[:, source_start : source_start + local]
    )
    experts.gate_up_proj.data[:, local:].copy_(
        tensor[:, intermediate + source_start : intermediate + source_start + local]
    )


def _load_fused_down(experts: Any, tensor: torch.Tensor, key: str) -> None:
    intermediate = experts.local_intermediate_size * experts.group.world_size
    expected = (experts.num_experts, experts.hidden_size, intermediate)
    if tuple(tensor.shape) != expected:
        raise ValueError(
            f"{key} shape mismatch: expected {expected}, got {tuple(tensor.shape)}"
        )
    source_start = experts.group.rank_in_group * experts.local_intermediate_size
    experts.down_proj.data.copy_(
        tensor[:, :, source_start : source_start + experts.local_intermediate_size]
    )


def _load_legacy_expert(
    experts: Any,
    expert_index: int,
    projection: str,
    tensor: torch.Tensor,
    key: str,
) -> None:
    if expert_index < 0 or expert_index >= experts.num_experts:
        raise ValueError(f"{key} has invalid expert index")
    intermediate = experts.local_intermediate_size * experts.group.world_size
    source_start = experts.group.rank_in_group * experts.local_intermediate_size
    local = experts.local_intermediate_size
    if projection in {"gate_proj", "up_proj"}:
        expected = (intermediate, experts.hidden_size)
        if tuple(tensor.shape) != expected:
            raise ValueError(
                f"{key} shape mismatch: expected {expected}, got {tuple(tensor.shape)}"
            )
        target_start = 0 if projection == "gate_proj" else local
        experts.gate_up_proj.data[
            expert_index, target_start : target_start + local
        ].copy_(tensor[source_start : source_start + local])
    else:
        expected = (experts.hidden_size, intermediate)
        if tuple(tensor.shape) != expected:
            raise ValueError(
                f"{key} shape mismatch: expected {expected}, got {tuple(tensor.shape)}"
            )
        experts.down_proj.data[expert_index].copy_(
            tensor[:, source_start : source_start + local]
        )


def _copy_parameter(
    parameter: torch.nn.Parameter, tensor: torch.Tensor, key: str
) -> None:
    if parameter.shape != tensor.shape:
        raise ValueError(
            f"{key} shape mismatch: expected {tuple(parameter.shape)}, "
            f"got {tuple(tensor.shape)}"
        )
    parameter.data.copy_(tensor)
