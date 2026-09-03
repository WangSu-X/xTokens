# 硬件信息

## 整体拓扑

```text
操作系统：Ubuntu 22.04.4 LTS
内核：Linux 5.4.0-150-generic x86_64

NUMA Node 0 / CPU Socket 0
  CPU：0-25,52-77（26 个物理核心，52 个逻辑 CPU）
  内存：约 377 GiB
  PCIe GPU：GPU 0-4

NUMA Node 1 / CPU Socket 1
  CPU：26-51,78-103（26 个物理核心，52 个逻辑 CPU）
  内存：约 378 GiB
  PCIe GPU：GPU 5-9

CPU Socket 0 <--- UPI/SMP 互连 ---> CPU Socket 1
```

每张 GPU 都连接到所属 NUMA 节点的 PCIe 层级。不同分组 GPU 之间的传输路径显示为 `SYS`，会经过 PCIe 和 CPU Socket 间的互连。当前未检测到 NVLink 连接。

## CPU

| 属性 | 值 |
|---|---|
| 型号 | Intel(R) Xeon(R) Gold 5320 CPU @ 2.20GHz |
| CPU Socket 数 | 2 |
| 每个 Socket 的核心数 | 26 |
| 每个核心的线程数 | 2（SMT/超线程） |
| 物理核心总数 | 52 |
| 逻辑 CPU 总数 | 104（`0-103`） |
| 最高/最低频率 | 2201 / 800 MHz |
| NUMA 节点数 | 2 |
| L3 缓存 | 每个 Socket 78 MiB（总计 156 MiB） |
| 指令集特性 | AVX2、AVX-512、VNNI、SHA-NI、AES |

NUMA 与 CPU 的对应关系：

```text
Node 0：0-25,52-77
Node 1：26-51,78-103
```

## GPU

共有 10 张 NVIDIA RTX A6000。每张 GPU 报告显存 49,140 MiB（约 48 GiB），PCIe 链路为 PCIe Gen 4 x16；当前链路和最大链路能力均为 `16.0 GT/s PCIe`、x16。

| GPU | PCI 总线地址 | NUMA 节点 | 显存 | 驱动版本 |
|---:|---|---:|---:|---|
| 0 | `00000000:4F:00.0` | 0 | 49140 MiB | 530.30.02 |
| 1 | `00000000:52:00.0` | 0 | 49140 MiB | 530.30.02 |
| 2 | `00000000:53:00.0` | 0 | 49140 MiB | 530.30.02 |
| 3 | `00000000:56:00.0` | 0 | 49140 MiB | 530.30.02 |
| 4 | `00000000:57:00.0` | 0 | 49140 MiB | 530.30.02 |
| 5 | `00000000:CE:00.0` | 1 | 49140 MiB | 530.30.02 |
| 6 | `00000000:D1:00.0` | 1 | 49140 MiB | 530.30.02 |
| 7 | `00000000:D2:00.0` | 1 | 49140 MiB | 530.30.02 |
| 8 | `00000000:D5:00.0` | 1 | 49140 MiB | 530.30.02 |
| 9 | `00000000:D6:00.0` | 1 | 49140 MiB | 530.30.02 |

`nvidia-smi topo -m` 报告：

- GPU 0-4 属于 NUMA Node 0，CPU Affinity 为 `0-25,52-77`。
- GPU 5-9 属于 NUMA Node 1，CPU Affinity 为 `26-51,78-103`。
- 同一 NUMA 节点内的 GPU 对使用 `PIX` 或 `PXB` PCIe 路径。
- 跨 NUMA 节点的 GPU 对使用 `SYS` 路径（PCIe 加 CPU Socket 间互连）。
- 没有出现 `NV#` 路径；当前环境未暴露 NVLink。

## 内存

| 属性 | 值 |
|---|---:|
| 可见内存总量 | 755 GiB |
| 已使用 | 99 GiB |
| 可用（available） | 405 GiB |
| 空闲（free） | 38 GiB |
| Swap | 总计 9 GiB，已使用约 444 MiB |

从 sysfs 读取的 NUMA 节点内存总量（约值）：

```text
Node 0：395,670,152 kB（约 377 GiB）
Node 1：396,318,092 kB（约 378 GiB）
```

## PCIe

- PCIe 是 10 张 GPU 使用的主机互连。
- GPU 链路：PCIe Gen 4、x16，当前和最大速度均为 `16.0 GT/s`。
- GPU 0-4 位于与 NUMA Node 0 关联的 PCIe 层级下。
- GPU 5-9 位于与 NUMA Node 1 关联的 PCIe 层级下。
- 同一 PCIe Switch/NUMA 节点内的路径显示为 `PIX`/`PXB`；跨 Socket 路径显示为 `SYS`，包含 UPI/SMP 互连。
