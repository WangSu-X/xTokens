from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import save_file
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

from x_tokens.config import (
    ExecutorConfig,
    KVCacheConfig,
    ModelConfig,
    ParallelConfig,
    XTokensConfig,
)
from x_tokens.core.kv_cache_manager import KVCacheManager
from x_tokens.core.protocol import SchedulerOutput, SchedulerRequest
from x_tokens.distributed import ParallelTopology
from x_tokens.engine.protocol import EngineCoreRequest, SamplingParams
from x_tokens.executor import ModelExecutor, MultiprocModelExecutor
from x_tokens.modeling.loader.qwen3_5_loader import _copy_projection
from x_tokens.modeling.models.qwen3_5 import (
    Qwen3_5Attention,
    Qwen3_5DecoderLayer,
    Qwen3_5GatedDeltaNet,
    Qwen3_5MLP,
    Qwen3_5Model,
)
from x_tokens.worker import GPUWorker, GPUWorkerConfig
from x_tokens.worker.model_runner import (
    GPUModelRunner,
    ModelRunnerInput,
    ModelRunnerRequest,
    SamplingMetadata,
)


@pytest.fixture
def tiny_qwen3_5() -> Qwen3_5ForCausalLM:
    torch.manual_seed(9)
    hf_config = Qwen3_5TextConfig(
        architectures=["Qwen3_5ForCausalLM"],
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        linear_key_head_dim=4,
        linear_value_head_dim=6,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_conv_kernel_dim=3,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.5,
            "mrope_section": [1, 0, 0],
        },
    )
    return Qwen3_5ForCausalLM(hf_config).eval()


def save_checkpoint(
    path: Path, model: Qwen3_5ForCausalLM, *, conditional: bool
) -> None:
    parameters = dict(model.state_dict())
    if conditional:
        config = {
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "text_config": model.config.to_dict(),
        }
        (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
        parameters = {
            name.replace("model.", "model.language_model.", 1): parameter
            for name, parameter in parameters.items()
        }
        parameters["model.visual.extra"] = torch.zeros(1)
        parameters["mtp.extra"] = torch.zeros(1)
    else:
        model.config.save_pretrained(path)
    save_file(parameters, path / "model.safetensors")


@pytest.mark.parametrize("conditional", [False, True], ids=["text", "conditional"])
def test_hybrid_prefill_apc_and_decode_match_transformers(
    tmp_path: Path, tiny_qwen3_5: Qwen3_5ForCausalLM, conditional: bool
) -> None:
    hf_model = tiny_qwen3_5
    save_checkpoint(tmp_path, hf_model, conditional=conditional)

    parallel = ParallelConfig(devices=("cpu",), collective_backend="gloo")
    config = XTokensConfig(
        model_config=ModelConfig(model=str(tmp_path)),
        executor_config=ExecutorConfig(dtype="float32", local_files_only=True),
        parallel_config=parallel,
    )
    worker = GPUWorker(
        GPUWorkerConfig(config, ParallelTopology.build(parallel), 0, "env://", 0, 0)
    )
    with patch.object(Path, "open", autospec=True, side_effect=Path.open) as open_file:
        worker.init_device()
        worker.load_model()
    checkpoint_path = tmp_path.resolve()
    assert (
        sum(
            call.args[0] == checkpoint_path / "config.json"
            for call in open_file.call_args_list
        )
        == 1
    )
    runner = worker.model_runner
    assert isinstance(runner, GPUModelRunner)
    model = runner.model
    assert model.load_config.model_path == checkpoint_path
    assert model.load_report.missing_parameters == ()
    assert model.load_report.unexpected_tensors == ()
    assert isinstance(model.model, Qwen3_5Model)
    linear_layer, full_layer = model.model.layers
    assert isinstance(linear_layer, Qwen3_5DecoderLayer)
    assert isinstance(full_layer, Qwen3_5DecoderLayer)
    assert isinstance(linear_layer.linear_attn, Qwen3_5GatedDeltaNet)
    assert isinstance(full_layer.self_attn, Qwen3_5Attention)
    assert all(isinstance(layer.mlp, Qwen3_5MLP) for layer in model.model.layers)
    cache_config = KVCacheConfig(block_size=4, num_gpu_blocks=8)
    manager = KVCacheManager(
        cache_config, num_blocks=8, max_model_len=16, model_fingerprint="test"
    )
    spec = model.get_kv_cache_spec(cache_config.block_size)
    runner.initialize_kv_cache(num_blocks=8, block_size=cache_config.block_size)
    assert runner.kv_cache.allocated_bytes == 8 * spec.bytes_per_block

    @torch.inference_mode()
    def compare(requests: tuple[EngineCoreRequest, ...]) -> None:
        runner_requests = []
        for request in requests:
            cached = (
                manager.get_computed_blocks(request)
                if request.request_id not in manager.req_to_blocks
                else ()
            )
            count = (
                request.num_uncomputed_tokens - len(cached) * cache_config.block_size
            )
            assert (
                manager.allocate_slots(request, count, computed_blocks=cached)
                is not None
            )
            block_ids = manager.get_block_ids(request.request_id)
            runner_requests.append(
                ModelRunnerRequest(
                    request.all_token_ids[-count:],
                    request.num_computed_tokens,
                    block_ids,
                    None,
                )
            )
        actual = runner._forward(ModelRunnerInput(0, tuple(runner_requests))).logits
        for index, request in enumerate(requests):
            expected = hf_model(
                input_ids=torch.tensor([request.all_token_ids]),
                use_cache=False,
                logits_to_keep=1,
            ).logits[0, 0]
            torch.testing.assert_close(actual[index], expected, atol=2e-5, rtol=2e-5)
            end = len(request.all_token_ids)
            manager.cache_blocks(request, end)
            request.num_computed_tokens = end

    prompt = (1, 2, 3, 4, 5, 6, 7, 8, 9)
    cold = EngineCoreRequest("cold", SamplingParams(), prompt)
    compare((cold,))
    manager.free("cold")

    warm = EngineCoreRequest("warm", SamplingParams(), prompt)
    other = EngineCoreRequest("other", SamplingParams(), (1, 2, 3, 4, 20, 21, 22))
    assert len(manager.get_computed_blocks(warm)) == 2
    assert len(manager.get_computed_blocks(other)) == 1
    compare((warm, other))
    for warm_token, other_token in ((10, 23), (11, 24)):
        warm.output_token_ids.append(warm_token)
        other.output_token_ids.append(other_token)
        compare((warm, other))
    manager.free("other")

    chunked = EngineCoreRequest("chunked", SamplingParams(), (17, 18, 19, 20, 21))
    assert manager.allocate_slots(chunked, 3) is not None
    first_chunk = manager.get_block_ids(chunked.request_id)
    warm.output_token_ids.append(12)
    assert manager.allocate_slots(warm, 1) is not None
    warm_decode = manager.get_block_ids(warm.request_id)
    greedy = SamplingMetadata(0, 1, None)
    runner.execute_model(
        ModelRunnerInput(
            1,
            (
                ModelRunnerRequest(chunked.prompt_token_ids[:3], 0, first_chunk, None),
                ModelRunnerRequest(
                    (12,), warm.num_computed_tokens, warm_decode, greedy
                ),
            ),
        )
    )
    with torch.inference_mode():
        expected = (
            hf_model(
                input_ids=torch.tensor([warm.all_token_ids]),
                use_cache=False,
                logits_to_keep=1,
            )
            .logits.argmax(dim=-1)
            .item()
        )
    assert runner.sample_tokens(1) == (expected,)
    manager.cache_blocks(chunked, 3)
    chunked.num_computed_tokens = 3
    manager.cache_blocks(warm, len(warm.all_token_ids))
    warm.num_computed_tokens = len(warm.all_token_ids)
    compare((chunked,))
    worker.shutdown()


def test_qwen3_5_tp2_loads_and_samples(
    tmp_path: Path, tiny_qwen3_5: Qwen3_5ForCausalLM, monkeypatch
) -> None:
    save_checkpoint(tmp_path, tiny_qwen3_5, conditional=True)
    parallel = ParallelConfig(
        distributed_executor_backend="mp",
        devices=("cpu:0", "cpu:1"),
        tensor_parallel_size=2,
        collective_backend="gloo",
        timeout_s=20,
    )
    tokens = (1, 2, 3)
    with torch.inference_mode():
        expected = (
            tiny_qwen3_5(
                input_ids=torch.tensor([tokens]), use_cache=False, logits_to_keep=1
            )
            .logits.argmax(dim=-1)
            .item()
        )
    request = EngineCoreRequest("request", SamplingParams(temperature=0), tokens)
    batch = SchedulerOutput(
        (SchedulerRequest(request, (0,)),),
        {request.request_id: len(tokens)},
    )
    config = XTokensConfig(
        model_config=ModelConfig(model=str(tmp_path)),
        executor_config=ExecutorConfig(dtype="float32", local_files_only=True),
        parallel_config=parallel,
        kv_cache_config=KVCacheConfig(block_size=4, num_gpu_blocks=8),
    )
    executor_class = ModelExecutor.get_class(config.parallel_config)
    executor = executor_class(config, pad_token_id=0)
    assert isinstance(executor, MultiprocModelExecutor)
    try:
        collective_rpc = executor.collective_rpc
        methods = []

        def record_method(method, *args, **kwargs):
            methods.append(method)
            return collective_rpc(method, *args, **kwargs)

        monkeypatch.setattr(executor, "collective_rpc", record_method)
        assert executor.execute_step(batch) == (expected,)
        assert methods == ["execute_and_sample"]

        output = executor.execute_model(batch)
        assert executor.sample_tokens(output, batch) == (expected,)
        executor.check_health()
    finally:
        executor.shutdown()


def test_fp8_dequantization_slices_across_partial_tiles() -> None:
    source = torch.arange(35, dtype=torch.float32).reshape(5, 7).to(torch.float8_e4m3fn)
    scale = torch.tensor([[0.25, 0.5, 1.0], [1.25, 1.5, 2.0], [2.25, 2.5, 3.0]])
    target = torch.empty(4, 5)
    _copy_projection(target, source, (slice(1, 5), slice(2, 7)), scale, (2, 3))
    expected = (
        source.float() * scale.repeat_interleave(2, 0).repeat_interleave(3, 1)[:5, :7]
    )
    torch.testing.assert_close(target, expected[1:5, 2:7])
