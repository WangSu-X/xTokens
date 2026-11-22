# 设计文档：Scheduler（请求调度器）

## 架构位置

本设计属于 xTokens 三层分布式架构的 **P1 - EngineCore** 层（见 arch.md）。

Scheduler 是 EngineCore 的核心组件，负责：
- 请求队列管理（WAITING → RUNNING → FINISHED）
- Prefill / Decode 阶段调度
- 逻辑 KV cache 分配与释放
- 请求终止条件判断（EOS / max_tokens / max_model_len）

涉及的主要类：`Scheduler`、`ScheduledRequest`、`SchedulerOutput`、`KVCacheManager`

## 摘要

Scheduler 是 EngineCore 内的单例调度器，维护请求生命周期和逻辑 KV 缓存状态。每次 `schedule()` 调用，Scheduler 执行准入、阶段选择和冲突解决，返回本步应执行的请求批次及其 KV cache 元数据。`update_from_output()` 根据采样结果更新请求状态，判断是否终止。通过内部 `KVCacheManager` 管理物理块分配，支持前缀复用、容量驱逐和自动重新准入。

## 背景

EngineCore 是单线程同步执行循环，每次 `step_fn()` 至多执行一个 batch 的模型前向。需要一个策略组件决定：

1. 哪些请求应进入本步 batch（FIFO 准入 + 容量控制）
2. 本步是 prefill 还是 decode（阶段选择）
3. 采样后请求何时终止（EOS / 长度限制）
4. 容量不足时如何处理（驱逐 vs 延迟）

同时，KV cache 的分块存储和前缀复用需要在调度层统一管理，而不是让每个 rank worker 独立决策，以确保逻辑块地址的一致性。

## 目标

- 提供稳定的 FIFO 准入和 prefill/decode 阶段调度
- 支持请求级的最大 token 数和模型上下文长度限制
- 通过 `KVCacheManager` 实现分块 KV cache、前缀复用和容量驱逐
- 支持请求中止和批处理失败时的回滚
- 为 EngineCore 提供协议无关的调度接口

## 非目标

- 不实现优先级调度、可抢占性或 SLA 保证
- 不实现跨步骤的 batch 合并或分割
- 不实现异步或后台调度工作线程
- 不处理多个 model 或多 replica 的调度协调

## 详细设计

### 请求生命周期

每个请求经历以下状态转换：

```
WAITING
  ↓ (admit: KV cache OK, running < max_num_seqs)
RUNNING
  ├─ needs_prefill=true (prefill phase)
  │  ↓ (schedule → execute → update_from_output → needs_prefill=false)
  ├─ needs_prefill=false (decode phase, loop until finish)
  │  ↓ (schedule → execute → update_from_output → finish_reason ≠ None)
FINISHED
  ↓ (release KV blocks)
[cleanup]

or

WAITING
  ↓ (abort)
ABORTED
```

### 数据结构

```python
@dataclass(slots=True)
class ScheduledRequest:
    request: GenerateRequest           # 原始请求 (含 sampling params)
    prompt_token_ids: tuple[int, ...]  # 不变的 prompt token IDs
    output_token_ids: list[int]        # 逐步累积的生成 tokens
    status: RequestStatus              # WAITING / RUNNING / FINISHED / ABORTED
    needs_prefill: bool                # 标记是否需要 prefill phase

@dataclass(frozen=True, slots=True)
class SchedulerOutput:
    requests: tuple[ScheduledRequest, ...]     # 本步执行的请求
    phase: SchedulerPhase                      # PREFILL or DECODE
    kv_cache_metadata: tuple[KVCacheMetadata]  # 每个请求的 KV cache 逻辑地址
```

### 调度算法

#### 1. 准入（Admission）

```python
def schedule():
    # Step 1: 尽量准入等待中的请求
    while waiting and len(running) < max_num_seqs:
        request = waiting[0]
        if not kv_cache_manager.admit(request_id, context_token_ids):
            break  # 容量不足，停止准入
        running[request_id] = request
        request.needs_prefill = True
        waiting.popleft()
```

准入条件：
- 等待队列非空
- running batch 未满（< max_num_seqs）
- KV cache 能分配足够 blocks（通过 `KVCacheManager.admit()` 校验）

#### 2. 阶段选择（Phase Selection）

```python
# 分离 prefill 和 decode 请求
prefill_reqs = [r for r in running if r.needs_prefill]
decode_reqs = [r for r in running if not r.needs_prefill]

# 优先级：
# 1. prefill (如果上一步不是 prefill 或没有 decode)
# 2. decode (如果存在)
# 3. prefill (最后一次)
if prefill_reqs and (last_phase != PREFILL or not decode_reqs):
    phase = PREFILL
    batch = prefill_reqs
elif decode_reqs:
    phase = DECODE
    batch = decode_reqs
else:
    return SchedulerOutput(())  # 无请求，空 batch
```

这个优先级设计避免 prefill 请求长时间阻塞 decode 流，同时允许单个 prefill 请求优先完成。

#### 3. KV Cache 准备

```python
try:
    metadata = tuple(
        kv_cache_manager.prepare_step(
            request_id,
            context_token_ids,
            is_prefill=(phase == PREFILL),
        )
        for request in batch
    )
except KVCacheCapacityError:
    # 驱逐最后加入的非当前 batch 请求
    victim = select_victim(running)
    running.pop(victim_id)
    kv_cache_manager.release(victim_id)
    victim.status = WAITING
    victim.needs_prefill = True
    waiting.appendleft(victim)
    # 重试 schedule()
```

若 KV cache 不足，选择受害者（通常是最后加入的非当前 batch 请求）进行驱逐，请求回到 WAITING 队列以便后续重新准入。

### 输出更新（Output Update）

```python
def update_from_output(batch, token_ids, eos_token_ids):
    for request, token_id in zip(batch.requests, token_ids):
        # 1. 追加 token
        request.output_token_ids.append(token_id)
        request.needs_prefill = False
        
        # 2. 提交 KV cache 步骤
        kv_cache_manager.commit_step(request_id, context_token_ids)
        
        # 3. 判断终止条件
        finish_reason = None
        if token_id in eos_token_ids and not request.sampling.ignore_eos:
            finish_reason = STOP
        elif completion_tokens >= max_tokens or context_len >= max_model_len:
            finish_reason = LENGTH
        
        # 4. 若终止，释放资源
        if finish_reason is not None:
            running.pop(request_id)
            kv_cache_manager.release(request_id)
            request.status = FINISHED
            
        yield SchedulerUpdate(request, token_id, finish_reason)
```

重要细节：
- EOS 只在 `sampling.ignore_eos=false` 时触发终止；为 `true` 时 EOS 作为普通 token
- 长度限制包括两个条件：`completion_tokens >= max_tokens` 或 `context_len >= max_model_len`
- 前者是用户指定的输出长度，后者是模型的硬限制
- 每个请求独立判断，允许 batch 内混合不同类型的终止

### KVCacheManager 接口

Scheduler 通过以下接口与 KVCacheManager 交互：

```python
class KVCacheManager:
    def admit(request_id: str, token_ids: tuple[int, ...]) -> bool:
        """尝试为请求分配初始 KV blocks，失败返回 False"""
        
    def prepare_step(request_id, token_ids, is_prefill) -> KVCacheMetadata:
        """为本步返回逻辑块地址和操作指令，不足时抛 KVCacheCapacityError"""
        
    def commit_step(request_id, token_ids) -> None:
        """提交本步的块分配，更新计数和前缀 hash"""
        
    def release(request_id) -> None:
        """请求完成或被驱逐时释放所有块"""
        
    def stats() -> KVCacheStats:
        """返回缓存统计（调试用）"""
```

详见 kvcachemanager.md。

## 接口与协议

### 外部接口

```python
class Scheduler:
    def add_request(request: GenerateRequest, prompt_token_ids: tuple[int, ...]) -> None:
        """将新请求加入 WAITING 队列"""
        
    def schedule() -> SchedulerOutput:
        """执行准入、阶段选择、KV cache 准备，返回本步 batch"""
        
    def update_from_output(
        batch: SchedulerOutput,
        token_ids: tuple[int, ...],
        eos_token_ids: frozenset[int],
    ) -> tuple[SchedulerUpdate, ...]:
        """根据采样结果更新请求状态并判断终止"""
        
    def abort(request_id: str) -> bool:
        """中止指定请求（WAITING 或 RUNNING），释放资源"""
        
    def fail_batch(batch: SchedulerOutput) -> tuple[ScheduledRequest, ...]:
        """batch 执行失败时的回滚（RUNNING → FINISHED，释放块）"""
```

### 与 EngineCore 的集成

```
EngineCore.step_fn():
    1. scheduler_output = scheduler.schedule()           # 获取本步 batch
    2. model_input = translate(scheduler_output)         # 转译为 ModelRunnerInput
    3. model_output = executor.execute_step(model_input) # 执行前向和采样
    4. updates = scheduler.update_from_output(
         scheduler_output,
         model_output.token_ids,
         eos_token_ids=executor.eos_token_ids,
       )                                                  # 更新状态
    5. for update in updates:
         outputs.add(CoreTokenEvent/CoreFinishedEvent)
```

## 测试与评估

- **FIFO 准入**：验证请求按顺序从 WAITING 进入 RUNNING，容量限制生效
- **阶段选择**：混合 prefill/decode 请求时验证阶段优先级正确
- **EOS 处理**：单独测试 `ignore_eos=false/true` 下的 EOS 行为
- **容量驱逐**：验证 KV cache 满时驱逐逻辑，被驱逐请求可正确重新准入
- **多请求生命周期**：运行多个并发请求，验证状态转换、token 累积和资源释放
- **中止和失败**：验证 `abort()` 和 `fail_batch()` 的状态恢复
- 单元测试见 `tests/engine/test_scheduler.py`

## 权衡与已知问题

### 优势

- **简单 FIFO**：无复杂优先级，易于调试和测试
- **阶段均衡**：prefill/decode 优先级平衡两类请求，避免饥饿
- **KV cache 一致**：集中管理逻辑块分配，保证所有 worker 一致
- **请求隔离**：每个请求独立状态，无全局锁争用

### 局限

- **无抢占**：驱逐是粗粒度的整请求驱逐，无细粒度 token 级别抢占
- **无优先级**：所有请求平等，无 SLA 或 QoS 控制
- **单线程**：调度器本身是单线程，无法利用多核；需要多进程副本扩展
- **静态 batch**：一旦 batch 成型，步内不能动态调整；跨步调整可能导致 prefill 延迟

### 未来扩展

1. **细粒度驱逐**：不驱逐整个请求，而是在 prefill 中间保存中间状态并恢复
2. **优先级调度**：为不同请求等级分配不同的 batch 比例或超时
3. **动态 batch**：允许 batch 内的请求数在一定范围内动态调整
4. **多 model 调度**：支持多个模型副本间的请求分配

## 结论

Scheduler 是 xTokens EngineCore 的核心调度组件，通过简洁的 FIFO 准入、阶段均衡和 KV cache 驱逐策略，支持高效的批量推理。与 KVCacheManager 配合，实现逻辑块的统一管理和前缀复用。设计完全位于 P1 层，与 P0 (API) 和 P2 (ModelRunner) 解耦。

## 完成标准

- 实现与 `x_tokens/core/scheduler.py` 代码一致
- 单元测试覆盖准入、阶段选择、EOS、驱逐、中止等关键路径
- 集成测试验证 EngineCore.step_fn() 中的端到端流程
- 文档中的接口签名和行为与代码对应
