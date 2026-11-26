# NVIDIA SPEED-Bench 数据集分析

## 1. 数据信息

- **数据集名称**：SPEED-Bench v1.1
- **发布方**：NVIDIA
- **主页**：https://huggingface.co/datasets/nvidia/SPEED-Bench
- **数据格式**：Parquet
- **总记录数**：8,944
- **主要用途**：评估 Speculative Decoding，以及不同输入长度、任务熵和并发条件下的 LLM 推理性能。

数据集分为两个部分：

| Split | 数据量 | 用途 |
|---|---:|---|
| Throughput | 8,064 | 推理吞吐量测试 |
| Qualitative | 880 | 不同语义任务下的投机解码测试 |

## 2. 数据构成

### 2.1 Throughput

按照输入长度和任务可预测性构建测试数据，分为三类：

| 类型 | 任务示例 | 特征 |
|---|---|---|
| Low Entropy | Coding、Sorting | 输出可预测性高 |
| Mixed Entropy | Needle-in-a-Haystack、Exams | 混合型任务 |
| High Entropy | Creative Writing、Dialogue | 输出可预测性低 |

### 2.2 Qualitative

包含 11 个领域，每个领域 80 条，共 880 条：

Coding、Humanities、Math、Multilingual、QA、RAG、Reasoning、Roleplay、STEM、Summarization、Writing。

主要用于评估不同语义任务对 Draft Model 预测能力的影响，指标包括 Acceptance Rate（AR）和 Acceptance Length（AL）。

## 3. 数据分布

### 3.1 Input Length 分布

| Input Length | 总数 | Low | Mixed | High |
|---|---:|---:|---:|---:|
| 1K | 1,536 | 512 | 512 | 512 |
| 2K | 1,536 | 512 | 512 | 512 |
| 8K | 1,536 | 512 | 512 | 512 |
| 16K | 1,536 | 512 | 512 | 512 |
| 32K | 1,536 | 512 | 512 | 512 |
| 100K | 384 | 128 | 128 | 128 |
| **Total** | **8,064** | **2,688** | **2,688** | **2,688** |

特点：
- 1K–32K 每个长度均有 1,536 条。
- 100K 有 384 条。
- 三种 Entropy 均衡分布。
- 属于人为控制的分层采样，而非真实请求长度分布。

### 3.2 Turn 分布

**Throughput**
- 8,064 条记录，按单 Prompt 请求口径，每条对应一个请求。

**Qualitative**

| 类型 | 数量 |
|---|---:|
| Single-turn | 713 |
| Multi-turn | 167 |
| Total | 880 |

- Multi-turn 占比：18.98%。
- 最大 Turn 数：5。
- 展开后约 1,140 个 Requests。
- 官方未公布每个 Turn 数对应的具体样本数量。

### 3.3 APC Prefix 信息

数据集没有直接提供 Prefix Length、Shared Prefix Tokens、Prefix Group ID 或 Cache Hit Rate。

但 Qualitative 的 Multi-turn 存在天然的前缀复用机会：

| Turn | 完整请求上下文 |
|---|---|
| Turn 1 | System + User 1 |
| Turn 2 | System + User 1 + Assistant 1 + User 2 |
| Turn 3 | System + User 1 + Assistant 1 + User 2 + Assistant 2 + User 3 |

后续请求携带之前的对话历史，因此同一 Conversation 内存在 Prefix Reuse 机会；不同 Conversation 之间则没有明确的共享前缀保证。

如果用于 APC Benchmark，需要额外统计：

| 字段 | 含义 |
|---|---|
| conversation_id | 对话 ID |
| turn_id | 当前轮次 |
| input_tokens | 完整输入长度 |
| prefix_tokens | 共享前缀长度 |
| prefix_ratio | 前缀共享比例 |
| prefix_group_id | 前缀分组 ID |

计算公式：

- Prefix Ratio = Shared Prefix Tokens / Input Tokens
- New Tokens = Input Tokens - Shared Prefix Tokens

## 4. Benchmark 使用建议

| 测试目标 | 使用数据 | 重点指标 |
|---|---|---|
| Prefill Performance | Throughput | TTFT、Prefill TPS |
| Decode Performance | Throughput | TPOT、Output TPS |
| Speculative Decoding | Qualitative | Acceptance Rate、Acceptance Length |
| Long Context | Throughput | KV Cache、Prefill 性能 |
| APC | Qualitative Multi-turn + 自建数据 | Prefix Reuse、Cache Hit Rate |

**局限性：** SPEED-Bench 不提供真实请求到达时间、完整 Output Length 分布和 APC 前缀统计，因此需要额外构造 Arrival Pattern、Output Length 和 Prefix Reuse Workload。