# 设计文档：ModelExecutor（模型执行器）

## 架构位置

本设计属于 xTokens 三层分布式架构的 **P1/P2 边界** 区域（见 arch.md）：

- **P1 - EngineCore**：Scheduler 决定本步 batch 和 KV cache 分配
- **P1/P2 边界**：ModelExecutor 是执行层的统一入口，屏蔽单卡 vs 多卡差异
- **P2 - ModelRunner Workers**：GPUWorker 和 ModelRunner 实现 SPMD 模型执行

涉及的主要类：`ModelExecutor`（抽象）、`UniProcModelExecutor`、`MultiprocModelExecutor`、`WorkerHandle`

## 摘要

`ModelExecutor` 是 EngineCore 与模型执行层的统一接口。它根据 `ParallelConfig` 的 `distributed_executor_backend` (uni/mp) 选择具体实现，持有单个模型副本的所有 Worker 句柄，并通过 collective RPC 或直接调用将 batch forward 和采样请求分发到各 worker。EngineCore 无需关心单卡 vs 多卡，只需调用 `execute_model()` 和 `sample_tokens()` 两个同步接口。当前仅支持张量并行 (TP=1/2/...)；管道并行、专家并行等维度需要未来扩展。

## 背景

EngineCore 需要一个执行后端来完成模型前向和采样。直接调用 Hugging Face 无法支持分布式并行、自定义 layer、或灵活的 KV cache 管理。因此需要设计一个通用的执行层接口，支持：

1. **单卡执行**（UniProc）：直接调用模型，无进程间通信
2. **多卡 TP 执行**（MultiProc）：通过 NCCL ring 协调，RPC 分发

此外，执行层需要与 Scheduler 协作：
- Scheduler 决定本步的请求 batch 和 KV cache 地址
- ModelExecutor 将调度输出转译为模型输入（token IDs、采样参数、块地址等）
- 执行后返回 logits，由 EngineCore 采样得到 token IDs

执行层还需管理模型初始化、权重加载、分布式上下文建立和生命周期。

## 目标

- 提供 `execute_model()` 和 `sample_tokens()` 两个稳定的同步接口
- 支持 uni (单卡) 和 mp (多卡 TP) 两种执行 backend
- 自动推导并初始化 KV cache 容量（blocks 数）
- 支持模型热加载和正常关闭
- 将 TP/PP/EP/CP group 的建立和参数 shard 逻辑隐藏在 worker 内
- 与 Scheduler 的 KV cache metadata 无缝集成

## 非目标

- 不实现管道并行 (PP)、专家并行 (EP) 或数据并行 (DP)；仅 TP
- GPUWorker 通过模型注册表构造 Runner，再由 Runner 加载权重
- 不实现异步执行；所有操作都是同步阻塞
- 不处理模型架构检测；需要用户显式指定模型类型

## 整体设计

```
EngineCore
    ↓ SchedulerOutput (running_requests, num_scheduled_tokens)
    ↓
ModelExecutor (control plane)
    ├─ execute_model(batch)
    │   ├─ 1. Translate: SchedulerOutput → ModelRunnerInput
    │   ├─ 2. collective_rpc("execute_model", input) to all workers
    │   ├─ 3. Gather responses, validate output rank
    │   └─ return ModelForwardOutput(runner_step_id)
    │
    └─ sample_tokens(forward_output, batch)
        ├─ 1. collective_rpc("sample_tokens", step_id) to all workers
        ├─ 2. Gather token batch from output rank
        └─ return token_ids
    
Per-worker (P2):
    ├─ GPUWorker.execute_model(input)
    │   ├─ Load KV blocks from block addresses
    │   ├─ Model forward (可能包含 allreduce)
    │   └─ return logits (仅 output rank) 或 None (其他 rank)
    │
    └─ GPUWorker.sample_tokens(step_id)
        ├─ 采样逻辑（温度、top-p、top-k）
        └─ return token_ids (仅 output rank) 或 None (其他 rank)
```

## 详细设计

### 接口

```python
class ModelExecutor(Executor, ABC):
    """Engine-facing control plane for model execution."""
    
    @staticmethod
    def get_class(parallel_config: ParallelConfig) -> type[ModelExecutor]:
        """根据 backend 选择 UniProc 或 MultiProc 实现"""
        
    def execute_model(self, batch: SchedulerOutput) -> ModelForwardOutput:
        """执行本步 forward，返回 logits 和步骤标识"""
        
    def sample_tokens(
        self, output: ModelForwardOutput, batch: SchedulerOutput
    ) -> tuple[int, ...]:
        """对 forward output 采样，返回本 batch 的 token IDs"""
        
    def check_health(self) -> None:
        """检查所有 worker 的健康状态"""
        
    def shutdown(self) -> None:
        """关闭所有 worker，释放资源"""
```

### 核心流程

#### 1. 初始化

```python
executor_class = ModelExecutor.get_class(config.parallel_config)
executor = executor_class(
    config,
    pad_token_id=pad_token_id,
    eos_token_ids=eos_token_ids,
)
```

初始化步骤：
1. Core 调用 `ModelExecutor.get_class()` 根据并行配置选择 Executor 类型，创建的 Executor 校验并行参数并构建单副本拓扑。
2. Executor 在 `_init_executor()` 中创建 Worker。
   - **UniProcModelExecutor**：直接创建 1 个 GPUWorker，初始化并加载模型
   - **MultiprocModelExecutor**：创建 N 个子进程，每个持有 1 个 GPUWorker
3. GPUWorker 初始化设备和并行上下文，解析模型目录，读取一次 `config.json` 并组装 `ModelLoadConfig`。`ModelRegistry.resolve()` 根据同一份配置返回 `ModelRegistration`，登记中的 `runner_factory(load_config, context, pad_token_id=..., is_output_rank=...)` 创建对应的 Runner。权重加载使用 `load_config.model_path`。
4. Executor 调用 `_initialize_kv_cache()`。
   - 向所有 worker 查询 KV cache 容量
   - 取最小值作为全局 `num_gpu_blocks`
   - 通知 worker 初始化对应大小的块池

#### 2. Execute Forward

```python
runner_input = executor._prepare_step(batch)

responses = executor.collective_rpc(
    "execute_model",
    runner_input,
)

# 仅 output rank 返回 ModelRunnerOutput，其他 rank 返回 None
output = responses[executor.output_rank]
executor.pending_step_id = output.step_id
return ModelForwardOutput(runner_step_id=executor.pending_step_id)
```

`_prepare_step()` 从 `SchedulerRequest` 读取本步 token、起始位置、block IDs 和可选采样参数，并组装成 `ModelRunnerRequest`。各 rank 的 Runner 计算本地 KV slot，输出 rank 返回计算结果。

#### 3. Sample Tokens

```python
step_id = output.runner_step_id

responses = executor.collective_rpc(
    "sample_tokens",
    step_id,
)

token_ids = responses[executor.output_rank]
executor.pending_step_id = None
return token_ids
```

采样必须在 forward 之后、下一次 forward 之前调用。若规则违反，抛 RuntimeError。

### 两种实现

#### UniProcModelExecutor（单卡或单进程）

```python
class UniProcModelExecutor(ModelExecutor):
    def _init_executor(self):
        worker = GPUWorker(GPUWorkerConfig(..., rank=0, output_rank=self.output_rank))
        worker.init_device()
        worker.load_model()
        self.worker_handles = (LocalWorkerHandle(0, worker),)
```

优点：
- 无进程间通信开销
- 便于在 CPU 上调试和测试

缺点：
- 无法利用多卡

#### MultiprocModelExecutor（多卡 TP）

```python
class MultiprocModelExecutor(ModelExecutor):
    def _init_executor(self):
        handles = []
        for rank in range(self.topology.world_size):
            config = GPUWorkerConfig(..., rank=rank, output_rank=self.output_rank)
            handles.append(spawn_worker(rank, config))
        self.worker_handles = tuple(handles)
```

优点：
- 支持多卡 TP
- 隔离进程内存空间

缺点：
- 进程管理和 IPC 复杂
- 需要处理进程崩溃等异常

### 单副本拓扑与 WorkerHandle

`ParallelTopology` 保存 `world_size`、`rank_assignments` 和 `group_specs`。副本内的 rank 按 PP × CP × TP 排列，Executor 按全局 rank 顺序持有全部 `worker_handles`，统一向它们发 RPC。当前 TP 执行固定从 rank 0 取结果。`data_parallel_size=1` 作为单副本配置值保留。

```python
class WorkerHandle(Protocol):
    def send(method: str, args) -> None: ...     # 异步发送 RPC
    def recv(timeout) -> Any: ...                # 同步接收 RPC 响应
    def check_health() -> None: ...              # 健康检查
    
class LocalWorkerHandle(WorkerHandle):
    """单卡/单进程：直接调用"""
    def __init__(rank: int, worker: GPUWorker):
        self.worker = worker
    def send(method, args):
        self.response = getattr(worker, method)(*args)
    def recv(timeout):
        return self.response
        
class ProcessWorkerHandle(WorkerHandle):
    """多进程 TP：通过 queue 或 pipe 通信"""
    def __init__(rank: int, process: subprocess.Process):
        self.rank = rank
        self.process = process
        self.send_queue = ...
        self.recv_queue = ...
    def send(method, args):
        self.send_queue.put((method, args))
    def recv(timeout):
        return self.recv_queue.get(timeout=timeout)
```

### 与 Scheduler 的协作

```
Scheduler                          ModelExecutor

schedule()
  ├─ assign token budget to running and waiting requests
  ├─ kv_cache_manager.allocate_slots()
  └─ kv_cache_manager.get_block_ids()
    ↓ SchedulerOutput
    │  (running_requests, num_scheduled_tokens)
    │
execute_model(batch)
  ├─ Translate SchedulerOutput to ModelRunnerInput
  │  - Read scheduled token IDs
  │  - Pack sampling metadata
  │  - Pass KV block addresses
  └─ RPC to workers
    ↓ ModelForwardOutput
    │
sample_tokens(output, batch)
  └─ RPC to workers
    ↓ token_ids
    │
update_from_output(batch, token_ids, eos_ids)
  ├─ kv_cache_manager.cache_blocks()
  ├─ Check finish reason (EOS / LENGTH)
  └─ SchedulerUpdate
```

Runner 根据 `ModelRunnerRequest` 中的 token 位置和 block IDs 计算 `slot_mapping`，供每个 rank 的 `GPUKVCache` 读写。

## KV Cache 初始化

```python
def _initialize_kv_cache(self):
    # 1. Query worker capacities
    capacities = self.collective_rpc("get_kv_cache_capacity", kv_cache_config)
    
    # 2. Take minimum (单个 TP rank 的最小容量)
    self.num_gpu_blocks = min(capacities)
    
    # 3. Notify all workers
    self.collective_rpc(
        "initialize_kv_cache",
        self.num_gpu_blocks,
        kv_cache_config.block_size,
    )
    
    # 4. Scheduler 使用该值创建 KVCacheManager
    # (见 EngineCore.__init__)
```

## 错误处理与健康检查

```python
def check_health(self):
    self._raise_if_failed()
    for handle in self.worker_handles:
        handle.check_health()
```

collective RPC 失败时记录 `failed_error`。后续 RPC 和健康检查通过 `_raise_if_failed()` 报告该错误。关闭时由具体 Executor 关闭全部 Worker 并清空 `worker_handles`。

## 测试与评估

- **单卡 (uni backend)**：CPU 和 GPU 上运行小模型 forward 和采样
- **多卡 (mp backend, TP=2)**：在 2 块 GPU 上运行 Qwen3.5，验证 allreduce 和 block 分发
- **RPC 超时**：模拟 worker 卡住，验证 recv(timeout) 行为
- **KV cache 容量**：多个请求并发，验证块的分配和释放
- **Worker 崩溃**：模拟子进程 crash，验证健康检查和异常处理
- 单元测试见 `tests/executor/test_model_executor.py`

## 权衡与已知问题

### 优势

- **统一接口**：EngineCore 无需知道单卡 vs 多卡实现细节
- **灵活的并行维度**：虽然当前仅 TP，架构预留 PP/EP/DP 扩展空间
- **隔离进程**：多卡 TP 时各 rank 内存独立，无法被其他 rank 污染
- **同步接口**：forward 和 采样分离，易于调试和状态跟踪

### 局限

- **仅 TP**：PP 和 EP 需要不同的 batch 管理策略
- **无动态并行度**：运行时无法改变 TP size
- **进程管理复杂**：多进程实现需要处理各种进程通信和崩溃场景
- **采样必须两步**：forward 后必须立即调用 sample_tokens；无法异步采样

### 未来扩展

1. **管道并行 (PP)**：需要 bubble scheduling 或 megatron-style pipeline
2. **专家并行 (EP)**：需要 MoE-specific 的 all-to-all 通信
3. **异步执行**：forward 和 采样解耦，并行执行多个 step
4. **多副本支持**：支持多个独立模型副本或 MoE group 副本

## 结论

ModelExecutor 是 xTokens 中执行层的统一入口，屏蔽单卡 vs 多卡 TP 的实现差异。通过 collective RPC 和 WorkerHandle 抽象，支持灵活的并行实现。与 Scheduler 的 KV cache metadata 紧密协作，确保分块 KV 缓存的逻辑地址一致性。设计完全位于 P1/P2 边界，为 EngineCore 提供稳定的执行接口。

## 完成标准

- 实现与 `x_tokens/executor/model_executor.py`、`uniproc_model_executor.py`、`multiproc_model_executor.py` 代码一致
- 单元测试覆盖单卡执行、多卡 TP、KV cache 初始化、错误处理等
- 集成测试验证 EngineCore.step_fn() 中 execute_model + sample_tokens 流程
- 文档中的接口和流程图与代码对应
- 支持 Qwen3.5 在 TP=2 下的推理
