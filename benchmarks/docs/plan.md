# xTokens Benchmark 计划

Serving 性能测试用 AIPerf 发请求并记录端到端指标。模型探针串行发请求，由 xTokens worker 采集模型级耗时。 [AIPerf](https://github.com/ai-dynamo/aiperf)

## 测试场景

### 1. 单并发模型性能

- 用 1K、8K、16K 三档输入跑 C1，建立单请求模型性能基线。
- 固定输入 token、Decode token 序列和步数。对比实验使用相同的 Prefill 分块配置，并记录实际分块大小。
- 用 worker 侧 CUDA event 分别测 Prefill 和逐步 Decode 耗时，记录计时覆盖的操作。

### 2. 分档 Serving 性能

- 使用 SPEED-Bench Throughput 的 1K、8K、16K 三档。按 Qwen tokenizer 和服务端实际计数校验输入分别为 1,024、8,192、16,384 tokens。数据清单记录来源版本、样本 ID 和长度调整方式。 [SPEED-Bench](https://huggingface.co/datasets/nvidia/SPEED-Bench)
- 每个长度分别跑 C4、C8、C16。同一档的各并发复用相同请求样本和输出策略，第一版沿用现有的固定 256-token 输出。
- 报告 TTFT、TPOT、吞吐和失败率。记录服务端的 `max_num_seqs` 和 KV cache 预算。用诊断运行的调度日志观察实际组批大小。
- 比较满足延迟目标的吞吐时，在用例配置中设定 TTFT 和 TPOT 的 SLO，并记录 goodput。

### 3. 混合请求 Serving 性能

- 复用分档压测的 SPEED-Bench 数据。第一版用 C8，按 1K、8K、16K 循环发请求，沿用分档测试的输出策略。用例配置记录循环顺序和比例。
- 按请求长度分别报告 P95 TTFT 和失败率。把混跑中的短请求与同并发下的纯短请求对照，观察长请求对短请求的影响。

### 4. GSM8K 正确性回归

- 使用 GSM8K `main/test`，用同一批题目、固定的 few-shot 示例、提示词模板和贪心生成设置分别跑 C1 和 C16。 [GSM8K](https://huggingface.co/datasets/openai/gsm8k)
- 报告两档的准确率和请求失败数，并逐题配对记录答案变化、由正确变错误的题目，以及并发下新增的截断和请求失败。

## 实验记录

每次实验保存模型与 xTokens 版本、GPU、运行参数、用例配置快照及数据清单。目录和运行命令见 [README](../README.md)。
