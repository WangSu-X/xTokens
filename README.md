# xTokens

轻量、易用的 LLM 推理系统，提供 OpenAI-compatible API server。

## 架构

请求沿 `FastAPI → LLMEngine → EngineCore → Scheduler → ModelExecutor → GPUModelRunner → Model` 执行。

- 只有一个 `Scheduler`，统一负责排队、prefill/decode 调度和逻辑 KV cache 管理。
- 执行层只保留项目的 `Executor` / `ModelExecutor` 抽象，支持 uni/mp 执行和 TP。
- KV cache 为必需能力，支持分块存储、prefix reuse、释放和抢占后的重算。
- 支持 Qwen3.5 混合注意力文本模型，包括 Qwen3.8-27B。
- 全注意力使用 PyTorch SDPA，Gated DeltaNet 使用 xTokens 的 PyTorch 实现。

## 环境与验证

需要 Python 3.12 或更新版本，以及与 PyTorch 兼容的运行环境。GPU 推理需要 NVIDIA GPU。

```bash
uv sync --locked
source .venv/bin/activate
uv run pytest -q tests
uv run ruff check x_tokens tests
```

需要在 xTokens 的同一个虚拟环境中运行 AIPerf 时，启用可选的 `benchmark` 依赖组。

```bash
uv sync --locked --group benchmark
uv run --locked --group benchmark aiperf --version
```

## 启动 API server

`--model` 为模型目录或 Hub ID；`--served-model-name` 为 API 名称，默认等于 `--model`。

```bash
uv run python -m x_tokens \
  --model /path/to/Qwen3.8-27B-FP8 \
  --served-model-name Qwen3.8-27B \
  --dtype bfloat16 \
  --local-files-only \
  --distributed-executor-backend mp \
  --tensor-parallel-size 2 \
  --kv-cache-block-size 128 \
  --max-num-seqs 4 \
  --max-model-len 4096 \
  --host 127.0.0.1 \
  --port 8000
```

上例使用两卡 TP；单卡部署可删除 `--distributed-executor-backend` 和 `--tensor-parallel-size`。
按需通过 `--parallel-device cuda:0 --parallel-device cuda:1` 指定设备。

模型加载后按各 rank 可用显存确定 KV block 数，也可通过 `--num-gpu-blocks` 显式指定。
容量必须至少容纳一条 `max_model_len` 长度的序列；CPU 执行必须显式指定 block 数。
`--disable-prefix-caching` 仅关闭前缀复用，不关闭 KV cache。

Qwen3.8-27B 可直接指定 BF16 或 block-scaled FP8 checkpoint 的目录。FP8 线性权重在加载时按 TP rank
解量化为运行 dtype。多卡 TP 默认使用 NCCL，建议用 128-token KV block 减少线性注意力的状态块数。

`--kv-cache-memory-bytes` 可指定每 rank 缓存字节预算。`--max-prefill-batch-tokens` 可限制单批未命中 APC 的 prefill token 数。
`--enable-chunked-prefill` 可让多请求负载按 KV block 边界分批 prefill，同一批可包含 decode 任务。
独立请求仍一次完成 prefill。`--prefill-chunk-alignment` 可设置多个 prefill 竞争时的中间切分粒度，需为 KV block 大小的整数倍。

## 不兼容变更

- `NaiveScheduler`、`KVCacheScheduler` 统一为 `Scheduler`，移除 scheduler policy 选择。
- 删除 `NaiveHFExecutor`；设备通过 `ParallelConfig.devices` 配置。
- 删除扁平 `ServeConfig`，统一使用 `x_tokens.config.XTokensConfig` 及各子配置。
- `EngineCore` 根据 `EngineCoreConfig` 构建 Executor，再用 Executor 确定的 block 数初始化 Scheduler。
- `ModelForwardOutput` 只携带待采样状态的 `runner_step_id`，删除 HF logits、`past_key_values`。
- CLI 删除 `--hf-*`、`--executor-backend`；使用 `--model`、`--served-model-name`、
  `--dtype`、`--local-files-only`、`--max-num-seqs`、`--parallel-device`。
- 删除旧 HF server 启动脚本；Transformers 模型只在测试中作为数值参考，不作为运行时 fallback。

历史设计中已失效的兼容方案均已标注，当前接口以代码和本 README 为准。
