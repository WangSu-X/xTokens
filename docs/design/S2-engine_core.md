# 设计文档：Token-ID EngineCore

> 当前 `EngineCore` 根据 `EngineCoreConfig` 构建 Executor，再按实际 KV cache 容量构建 `Scheduler`。文中涉及 `NaiveScheduler`、`KVCacheScheduler`、scheduler factory 和无 cache 分支的内容属于历史方案。当前测试位于 `tests/engine/test_engine_core.py`。

## 架构位置

本设计属于 xTokens 三层分布式架构的 **P1 - EngineCore** 层（见 S0-arch.md）：

- **P0 - API Server**：OpenAI HTTP 服务、请求转换（见 S1-serve_api.md）
- **P1 - EngineCore**：单线程事件循环、请求调度、KV cache 管理、执行协调、事件汇聚
- **P2 - ModelRunner**：SPMD 并行执行、模型 forward、采样（见 S4-custom_model.md）

本设计涉及的核心组件：`EngineCore`、`Scheduler`（与 `KVCacheManager`，见 S8-kv_cache.md）、`Executor` 协议、`NaiveScheduler` 默认实现、`EngineCoreRequest` 和 `EngineCoreOutputs` 数据结构。

## 摘要

`EngineCore` 负责校验 token ID 请求，由自身创建的 `Scheduler` 管理请求生命周期，并由 `Executor` 执行模型推理与采样。Core 根据执行结果生成请求输出。

## 背景

`EngineCore` 是 xTokens 的同步推理后端。Serve 与 `LLMEngine` 已在 Core 外完成 chat rendering、tokenization 和输出文本解码；因此 Core 不应依赖 FastAPI、OpenAI DTO、tokenizer 或 Hugging Face 模型实现。

当前需要一个可替换 executor 的最小 Core：它接收稳定的 token-ID 请求，维护请求生命周期和 batch 调度，并将 executor 的下一个 token ID 转为 Core events。

## 目标

- 仅接收包含 prompt token IDs 的 `EngineCoreRequest`。
- 负责模型别名和 sampling 参数校验、FIFO 准入、运行状态、取消和关闭。
- 每次 `step_fn()` 至多执行一次 batch forward，并输出 `CoreTokenEvent`、`CoreFinishedEvent` 或 `CoreErrorEvent`。
- 仅依赖 `Executor` protocol，使 `NaiveHFExecutor` 可以替换为其他执行后端。

不包含 tokenization、chat template、detokenization、HTTP/SSE、KV cache 或 Core worker/IPC。

## 非目标

- 不处理文本 tokenization、chat template、detokenization、HTTP/SSE 或 OpenAI DTO。
- 不实现 KV cache、prefix cache、优先级、抢占、chunked prefill 或异步 Core worker。
- 不把具体模型框架类型暴露到 Core contract。

## 整体设计

```mermaid
flowchart LR
    Engine["P0: LLMEngine"] --> Client[EngineCoreClient]
    Client --> Core["P1: EngineCore"]
    Core --> Scheduler["P1: Scheduler protocol"]
    Scheduler -. default implementation .-> Naive["P1: NaiveScheduler + KVCacheManager"]
    Core --> Executor["P1/P2 boundary: Executor protocol"]
    Executor -. implementation .-> Exec["P2: ModelExecutor"]
```

`EngineCoreConfig` 保存 `XTokensConfig` 和 tokenizer 提供的 pad/EOS token ID。Core 构造时先通过注册模型工厂创建 Executor，再把实际的 `num_gpu_blocks` 交给 Scheduler。Scheduler 初始化失败时，Core 关闭已创建的 Executor。

单进程客户端在本进程创建 Core，多进程客户端把配置交给 Core 子进程。可选的 `executor_factory` 用于替换默认工厂。

## 详细设计

### 核心流程

1. Engine 预处理文本为 token IDs。多进程 Core 的 `process_input_sockets` 线程读取 ZMQ 消息中的 `request_type`，并按请求类型处理。收到 ADD 时，通过 `decode_core_request()` 校验模型与 prompt 并构造 `EngineCoreRequest`。单进程客户端直接构造同一类型。
2. `EngineCore.add_request()` 校验采样参数，Scheduler 校验准入条件并将同一个请求实例加入 waiting 队列。请求校验失败时返回 `CoreErrorEvent`。
3. `step_fn()` 先取出暂存事件，再由 `Scheduler.schedule()` 将请求填入 running batch；默认实现以 FIFO 纳入请求，上限为 `max_num_seqs`。
4. Core 调用 `Executor.execute_model(SchedulerOutput)` 获取 `ModelForwardOutput`。
5. Core 将 forward 输出交给 `Executor.sample_tokens(output, SchedulerOutput)`，每个 running request 得到一个 token ID。
6. Scheduler 追加 token 并依据请求级 `ignore_eos`、EOS、`max_tokens` 或 `max_model_len` 计算完成状态。普通请求的 EOS 产生 `STOP` 且不发送 token event；ignore-EOS 请求把 EOS 作为普通 `CoreTokenEvent`，继续生成到长度限制。完成请求额外产生 `CoreFinishedEvent`。
7. forward 或采样失败时 Core 调用 `fail_batch()`，向仍在该 batch 中的请求分别产生 `CoreErrorEvent`。
8. 每次 `schedule()` 在 DEBUG 级别记录 batch size、waiting/running 数量、admitted 数量和 request IDs；Core 在 DEBUG 级别记录模型 step 耗时。executor 或采样异常记录带 traceback 的 ERROR 日志。

```mermaid
sequenceDiagram
    participant E as LLMEngine
    participant I as EngineCoreClient
    participant C as EngineCore
    participant S as Scheduler
    participant X as Executor

    E->>I: add_request(GenerateRequest)
    I->>C: add_request(EngineCoreRequest)
    C->>S: add_request(EngineCoreRequest)
    C->>C: step_fn()
    C->>S: schedule()
    S-->>C: SchedulerOutput
    C->>X: execute_model(SchedulerOutput)
    X-->>C: ModelForwardOutput
    C->>X: sample_tokens(output, SchedulerOutput)
    X-->>C: token IDs
    C->>S: update_from_output()
    C-->>E: EngineCoreOutputs
```

`abort_requests()` 可从 waiting 或 running 移除请求。多进程 Core 收到 CLOSE 后结束输入处理，由循环入口的 `_handle_shutdown()` 调用 `close()`，取消剩余的请求并关闭 Executor。

### 接口与数据结构

```python
class Executor(Protocol):
    @property
    def eos_token_ids(self) -> frozenset[int]: ...
    def execute_model(self, batch: SchedulerOutput) -> ModelForwardOutput: ...
    def sample_tokens(
        self, output: ModelForwardOutput, batch: SchedulerOutput
    ) -> tuple[int, ...]: ...

class Scheduler(Protocol):
    @property
    def has_work(self) -> bool: ...
    def add_request(self, request: EngineCoreRequest) -> None: ...
    def schedule(self) -> SchedulerOutput: ...
    def update_from_output(
        self, batch: SchedulerOutput, token_ids: tuple[int, ...], *,
        eos_token_ids: frozenset[int]
    ) -> tuple[SchedulerUpdate, ...]: ...
    def fail_batch(self, batch: SchedulerOutput) -> tuple[EngineCoreRequest, ...]: ...
    def abort(self, request_id: str) -> bool: ...
    def abort_all(self) -> tuple[EngineCoreRequest, ...]: ...

@dataclass(frozen=True, slots=True)
class ModelForwardOutput:
    logits: Any
    past_key_values: Any | None = None

class EngineCore:
    def add_request(self, request: EngineCoreRequest) -> None: ...
    def abort_requests(self, request_ids: tuple[str, ...]) -> None: ...
    def step_fn(self) -> tuple[EngineCoreOutputs, bool]: ...
    def post_step(self, *, model_executed: bool) -> None: ...
    def close(self) -> None: ...
```

| 类型 | 职责 |
| --- | --- |
| `EngineCoreConfig` | 构建 Core 所需的运行配置和 tokenizer ID。 |
| `Scheduler` | 调度、输出更新、失败清理和取消的接口；`update_from_output()` 返回 `SchedulerUpdate`，由 Core 转换为 `EngineCoreOutputs`。 |
| `NaiveScheduler` | `Scheduler` 的 FIFO、无 KV cache 默认实现。 |
| `EngineCoreRequest` | Core 内唯一的请求实体，持有 prompt 和输出 token IDs、采样参数、调度和终止状态、已计算 token 数及完整前缀块 hash。 |
| `EngineCoreOutputs` | 按 output channel 组织的本 step Core events；当前使用 channel `0`。 |

### 状态图

```mermaid
stateDiagram-v2
    [*] --> Waiting: add_request
    Waiting --> Running: schedule
    Running --> Running: non-terminal token
    Running --> Running: EOS and ignore_eos=true
    Running --> Finished: EOS and ignore_eos=false
    Running --> Finished: max_tokens / max_model_len
    Waiting --> Aborted: abort_requests
    Running --> Aborted: abort_requests
    Running --> Failed: executor error
    Finished --> [*]
    Aborted --> [*]
    Failed --> [*]
```

### 兼容性与迁移

`EngineCore` keeps a token-ID input and Core-event output boundary. Existing callers continue to use
the default `NaiveScheduler`; custom schedulers can be injected through `EngineCore` or `InprocClient`.
`SchedulerOutput` 表示 `Scheduler.schedule()` 的调度结果，`SchedulerUpdate` 表示单个请求的
生命周期更新。

## 测试与评估

- `tests/engine/test_scheduler.py` 验证 FIFO、并发上限、完成与取消。
- `tests/engine/test_hf_core.py` 用 `FakeExecutor` 验证 token-ID 边界、batch event 顺序、错误隔离和文本 prompt 拒绝。
- `tests/engine/test_inproc_client.py` 验证 `InprocClient` 在调用线程推进 Core。
- Scheduler 和 Core 日志测试验证每次调度状态、step 耗时及 batch 异常事件。

```bash
uv run ruff check x_tokens tests
uv run ruff format --check x_tokens tests
uv run pytest -q tests/engine
```

## 权衡与已知问题

### 优点

- Core 输入和输出都是稳定的 token-ID/Core-event contract，易于替换 scheduler、executor 或 transport。
- 调度测试不需要 tokenizer、torch 或 transformers。

### 限制

- `step_fn()` 是同步接口；in-process HF forward 会阻塞调用线程。
- 当前 scheduler 没有 KV cache、优先级、抢占、prefix cache 或 chunked prefill。

### 权衡


## 结论

`EngineCore` 现在具备稳定的 token-ID contract，以及明确的 scheduler/executor 边界。在保持正确性
基线简洁的同时，可以引入新的调度和执行实现，而无需将协议或 tokenizer 逻辑移动到 Core。

## 完成标准

- Core rejects invalid text prompts and validates model/sampling constraints.
- Scheduler and executor failures produce request-scoped `CoreErrorEvent` events.
- Scheduler injection and default `NaiveScheduler` behavior are tested.
- `pytest`, Ruff checks, and formatting checks pass.

通过 `Scheduler` protocol 注入调度策略，以简单的默认实现保持当前行为，同时为连续批处理等策略保留扩展点。同步 pull Core 以简单、可测试的执行语义换取独立 worker 和异步背压能力。
