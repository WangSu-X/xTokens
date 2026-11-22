# xTokens

轻量、易用的 LLM 推理系统，提供 OpenAI-compatible API server。

## 架构

请求沿 `FastAPI → LLMEngine → EngineCore → Scheduler → ModelExecutor → GPUModelRunner → Model` 执行。

- 只有一个 `Scheduler`，统一负责排队、prefill/decode 调度和逻辑 KV cache 管理。
- 执行层只保留项目的 `Executor` / `ModelExecutor` 抽象，支持 uni/mp 执行和 TP。
- KV cache 为必需能力，支持分块存储、prefix reuse、释放和抢占后的重算。
- 自定义模型目前支持 Qwen3-MoE；保留 HF config、tokenizer、Hub 和 SafeTensors 加载能力，
  不提供 Hugging Face 模型执行后端或 fallback。
- Attention 当前通过聚合 KV 张量调用 PyTorch SDPA，尚未实现原生 paged attention kernel。

## 环境与验证

需要 Python 3.12 或更新版本，以及与 PyTorch 兼容的运行环境。GPU 推理需要 NVIDIA GPU。

```bash
uv sync --locked
source .venv/bin/activate
uv run pytest -q
uv run ruff check x_tokens tests
```

## 启动 API server

`--model` 为模型目录或 Hub ID；`--served-model-name` 为 API 名称，默认等于 `--model`。

```bash
uv run python -m x_tokens \
  --model /model \
  --served-model-name qwen3-30b-a3b \
  --dtype bfloat16 \
  --local-files-only \
  --distributed-executor-backend mp \
  --tensor-parallel-size 2 \
  --max-num-seqs 4 \
  --max-model-len 4096 \
  --host 127.0.0.1 \
  --port 8000
```

上例使用两卡 TP；单卡部署可删除 `--distributed-executor-backend` 和 `--tensor-parallel-size`。
按需通过 `--parallel-device cuda:0 --parallel-device cuda:1` 指定设备。两卡 Qwen3-MoE 服务也可使用
`scripts/xtokens_serve_qwen3_30b_a3b.sh`。

模型加载后按各 rank 可用显存确定 KV block 数，也可通过 `--num-gpu-blocks` 显式指定。
容量必须至少容纳一条 `max_model_len` 长度的序列；CPU 执行必须显式指定 block 数。
`--disable-prefix-caching` 仅关闭前缀复用，不关闭 KV cache。

## 不兼容变更

- `NaiveScheduler`、`KVCacheScheduler` 统一为 `Scheduler`，移除 scheduler policy 选择。
- 删除 `NaiveHFExecutor` 和 executor backend 选择；`ExecutorConfig` 只保留 `dtype`、
  `local_files_only`，设备使用 `ParallelConfig.devices`。
- 删除扁平 `ServeConfig`，统一使用 `x_tokens.config.XTokensConfig` 及各子配置。
- `EngineCoreConfig.kv_cache_config` 必填，创建 Core 前必须解析 `num_gpu_blocks`。
- `ModelForwardOutput` 只携带待采样状态的 `runner_step_id`，删除 HF logits、`past_key_values`。
- CLI 删除 `--hf-*`、`--executor-backend`；使用 `--model`、`--served-model-name`、
  `--dtype`、`--local-files-only`、`--max-num-seqs`、`--parallel-device`。
- 删除旧 HF server 启动脚本；Transformers 模型只在测试中作为数值参考，不作为运行时 fallback。

历史设计中已失效的兼容方案均已标注，当前接口以代码和本 README 为准。
