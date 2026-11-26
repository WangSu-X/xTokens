# xTokens Benchmark

| 用例 | 数据 | 默认负载 |
|---|---|---|
| `model` | SPEED-Bench | 1K、8K、16K 各跑 C1，采集 worker 侧 Prefill 和 Decode 耗时 |
| `serving` | SPEED-Bench | 1K、8K、16K 各跑 C4、C8、C16，固定输出 256 tokens |
| `mixed` | 同一份 SPEED-Bench | C8 按 1K、8K、16K 循环发请求，沿用分档输出策略 |
| `accuracy` | GSM8K | 同一批题目跑 C1 和 C16，逐题对比 |

## 准备和运行

在仓库根目录安装 benchmark 依赖并准备数据。需要已有的本地模型和 tokenizer，可用 `--model` 指定。下载 SPEED-Bench 时可通过 `--mirror` 指定 Hugging Face 镜像。

```bash
uv sync --locked --group benchmark
source .venv/bin/activate
python -m benchmarks.scripts.run prepare
python -m benchmarks.scripts.run all --gpus 6,7 --output benchmarks/results/baseline
python -m benchmarks.scripts.report benchmarks/results/baseline
```

运行时使用有 CUDA 和 AIPerf 0.14.0 的环境。需要指定驱动库路径时，可沿用本机的 `LD_LIBRARY_PATH` 环境变量设置。将 `all` 换成表中的用例名可单独运行。小样本验证可用 `--requests 16 --repeats 1 --warmup 1 --model-samples 3 --gsm-limit 32`。

`accuracy` 默认在一次运行中完成两档并发。分片时指定 `--gsm-shard-count N --gsm-shard-index I`，为每个进程分配独立端口和 GPU，并将结果写入同一个实验下的 `gsm-shard-I/`。

## 文件结构

- `benches/` 保存四个 benchmark 的配置。混跑继承 `serving.yaml` 的输出策略和轮次。共享的运行配置在 `runtime.yaml`。
- `docs/` 保存评测口径和数据集分析。 [评测计划](docs/plan.md) · [SPEED-Bench 分析](docs/dataset.md)
- `scripts/` 包含数据准备、执行和汇总脚本，以及 worker 探针和 AIPerf 适配器。分档和混跑共用 Serving runner。
- `data/` 保存原始文件、定长样本和数据清单。准备脚本使用公开 Parquet 中自带正文的样本，按 Qwen tokenizer 截断或补齐。每档默认 128 条。公开文件中的占位记录会被跳过。 [SPEED-Bench 数据说明](https://huggingface.co/datasets/nvidia/SPEED-Bench#dataset-format)
- `results/<experiment>/<suite>/` 保存实际配置、请求记录、校验和日志。实验根目录下的 `summary.json` 和 `summary.md` 汇总四项结果，也可读取旧版实验的 `<suite>/xtokens/` 路径。

Serving 关闭前缀缓存。客户端并发表示在途请求数。实际组批受 KV cache 与 Prefill token 预算限制。需要排查组批时，可在诊断运行中设 `XTOKENS_LOGGING_LEVEL=DEBUG` 并查看 `server.log` 的 `scheduler step`。报告中的混跑 P95 基于各档通过 token 校验的请求，失败率包含超时或缺失响应。
