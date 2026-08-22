from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from transformers import Qwen3MoeConfig as HFQwen3MoeConfig
from transformers import Qwen3MoeForCausalLM as HFQwen3MoeForCausalLM

from x_tokens.config import ParallelConfig
from x_tokens.core.scheduler import ScheduledRequest, SchedulerOutput
from x_tokens.distributed import ParallelTopology, materialize_parallel_context
from x_tokens.engine.types import GenerateRequest, SamplingParams
from x_tokens.executor import MultiprocModelExecutor
from x_tokens.modeling import ModelInput, ModelLoadConfig
from x_tokens.modeling.loader import resolve_model_source
from x_tokens.modeling.models import Qwen3MoeForCausalLM, Qwen3MoeRunnerFactory
from x_tokens.modeling.models.qwen3_moe import resolve_torch_dtype
from x_tokens.modeling.registry import resolve_model_registration


def tiny_hf_model() -> HFQwen3MoeForCausalLM:
    torch.manual_seed(7)
    config = HFQwen3MoeConfig(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=32,
        moe_intermediate_size=4,
        num_experts=2,
        num_experts_per_tok=1,
        norm_topk_prob=True,
        torch_dtype="float32",
        tie_word_embeddings=False,
    )
    config.architectures = ["Qwen3MoeForCausalLM"]
    model = HFQwen3MoeForCausalLM(config)
    model.eval()
    return model


def save_tiny_checkpoint(
    directory: Path,
    model: HFQwen3MoeForCausalLM,
    *,
    legacy_experts: bool,
    sharded: bool = False,
) -> None:
    model.config.save_pretrained(directory)
    state = model.state_dict()
    if not legacy_experts:
        save_file(dict(state), directory / "model.safetensors")
        return
    converted: dict[str, torch.Tensor] = {}
    for key, tensor in state.items():
        if key.endswith("mlp.experts.gate_up_proj"):
            intermediate = tensor.shape[1] // 2
            prefix = key.removesuffix("experts.gate_up_proj")
            for expert in range(tensor.shape[0]):
                converted[f"{prefix}experts.{expert}.gate_proj.weight"] = tensor[
                    expert, :intermediate
                ].clone()
                converted[f"{prefix}experts.{expert}.up_proj.weight"] = tensor[
                    expert, intermediate:
                ].clone()
        elif key.endswith("mlp.experts.down_proj"):
            prefix = key.removesuffix("experts.down_proj")
            for expert in range(tensor.shape[0]):
                converted[f"{prefix}experts.{expert}.down_proj.weight"] = tensor[
                    expert
                ].clone()
        else:
            converted[key] = tensor.clone()
    if not sharded:
        save_file(converted, directory / "model.safetensors")
        return
    items = list(converted.items())
    split = len(items) // 2
    shards = (items[:split], items[split:])
    weight_map: dict[str, str] = {}
    for index, shard_items in enumerate(shards, start=1):
        filename = f"model-{index:05d}-of-00002.safetensors"
        save_file(dict(shard_items), directory / filename)
        weight_map.update({key: filename for key, _ in shard_items})
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}),
        encoding="utf-8",
    )


def load_custom_model(directory: Path) -> Qwen3MoeForCausalLM:
    load_config = ModelLoadConfig(str(directory), dtype="float32")
    parallel_config = ParallelConfig(devices=("cpu",), collective_backend="gloo")
    source = resolve_model_source(load_config)
    resolved = resolve_model_registration(source, parallel_config)
    context = materialize_parallel_context(
        ParallelTopology.build(parallel_config), 0, torch_module=torch
    )
    model = Qwen3MoeForCausalLM(
        resolved.config,
        context,
        resolve_torch_dtype(load_config, resolved.config),
        source,
        load_config,
    )
    model.load_model()
    model.eval()
    return model


@pytest.mark.parametrize(
    ("legacy_experts", "sharded"),
    [(False, False), (True, False), (True, True)],
)
def test_qwen3_moe_checkpoint_schema_and_logits_match_hf(
    tmp_path: Path, legacy_experts: bool, sharded: bool
) -> None:
    hf_model = tiny_hf_model()
    save_tiny_checkpoint(
        tmp_path,
        hf_model,
        legacy_experts=legacy_experts,
        sharded=sharded,
    )
    custom_model = load_custom_model(tmp_path)
    input_ids = torch.tensor([[0, 0, 1], [4, 5, 6]])
    attention_mask = torch.tensor([[0, 0, 1], [1, 1, 1]])
    position_ids = attention_mask.cumsum(dim=-1) - 1

    with torch.inference_mode():
        expected = hf_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            logits_to_keep=1,
        ).logits[:, -1, :]
        actual = custom_model(
            ModelInput(input_ids, attention_mask, position_ids)
        ).logits

    assert custom_model.load_report is not None
    assert custom_model.load_report.missing_parameters == ()
    assert custom_model.load_report.unexpected_tensors == ()
    assert torch.allclose(actual, expected, atol=2e-5, rtol=2e-5)


def test_qwen3_moe_tp2_executor_loads_rank_shards_and_samples(
    tmp_path: Path,
) -> None:
    hf_model = tiny_hf_model()
    save_tiny_checkpoint(tmp_path, hf_model, legacy_experts=False)
    load_config = ModelLoadConfig(str(tmp_path), dtype="float32")
    parallel_config = ParallelConfig(
        distributed_executor_backend="mp",
        devices=("cpu:0", "cpu:1"),
        tensor_parallel_size=2,
        collective_backend="gloo",
        timeout_s=20,
    )
    source = resolve_model_source(load_config)
    resolved = resolve_model_registration(source, parallel_config)
    executor = MultiprocModelExecutor(
        parallel_config,
        Qwen3MoeRunnerFactory(source, resolved.config, load_config, pad_token_id=0),
    )
    request = GenerateRequest(
        "tiny-qwen",
        "tiny",
        (1, 2, 3),
        SamplingParams(max_tokens=1, temperature=0),
    )
    batch = SchedulerOutput((ScheduledRequest(request, (1, 2, 3)),))
    with torch.inference_mode():
        expected = int(
            hf_model(
                input_ids=torch.tensor([[1, 2, 3]]),
                attention_mask=torch.ones((1, 3), dtype=torch.long),
                use_cache=False,
                logits_to_keep=1,
            )
            .logits[0, -1]
            .argmax()
        )
    try:
        output = executor.execute_model(batch)
        assert executor.sample_tokens(output, batch) == (expected,)
    finally:
        executor.shutdown()
