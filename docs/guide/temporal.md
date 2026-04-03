# 时序图

本教程展示如何使用 VGL 处理时序图和进行时序事件预测。

## 构建时序图

```python
import torch
from vgl.graph import Graph

graph = Graph.temporal(
    nodes={"x": torch.randn(100, 32)},
    edges={
        "edge_index": torch.randint(0, 100, (2, 500)),
        "timestamp": torch.sort(torch.rand(500) * 100)[0],
    },
    time_attr="timestamp",
)
```

## 时序事件预测

### 1. 准备样本

每个样本是一个 `TemporalEventRecord`，携带源节点、目标节点、时间戳和标签：

```python
from vgl.dataloading import TemporalEventRecord, ListDataset, DataLoader, FullGraphSampler

samples = [
    TemporalEventRecord(graph=graph, src_index=0, dst_index=1, timestamp=3.0, label=1),
    TemporalEventRecord(graph=graph, src_index=2, dst_index=0, timestamp=5.0, label=0),
]

loader = DataLoader(
    dataset=ListDataset(samples),
    sampler=FullGraphSampler(),
    batch_size=32,
)
```

### 2. 训练

```python
from vgl.tasks import TemporalEventPredictionTask
from vgl.engine import Trainer

task = TemporalEventPredictionTask(target="label")

trainer = Trainer(
    model=model,
    task=task,
    optimizer=torch.optim.Adam,
    lr=1e-3,
    max_epochs=50,
)
trainer.fit(loader)
```

## TemporalEventBatch

`TemporalEventBatch` 将多个 `TemporalEventRecord` 聚合为一个批量输入：

- `batch.graph` — 时序图
- `batch.src_index` — 源节点索引
- `batch.dst_index` — 目标节点索引
- `batch.timestamp` — 事件时间戳
- `batch.labels` — 标签

## 时序邻居采样

对于大规模时序图，使用 `TemporalNeighborSampler` 进行严格历史采样：

```python
from vgl.dataloading import TemporalNeighborSampler

loader = DataLoader(
    dataset=ListDataset(samples),
    sampler=TemporalNeighborSampler(num_neighbors=[15, 10]),
    batch_size=32,
)
```

`TemporalNeighborSampler` 保证采样的邻居事件严格早于当前事件的时间戳。

## 异构时序图

当时序图包含多种关系或节点类型时，在每条记录上指定 `edge_type`：

```python
record = TemporalEventRecord(
    graph=graph,
    src_index=0,
    dst_index=1,
    timestamp=3.0,
    label=1,
    edge_type=("author", "writes", "paper"),
)
```

`TemporalEventBatch` 会暴露 `edge_types`、`edge_type_index`、`src_node_type` 和 `dst_node_type` 供类型感知模型使用。

## 时序图视图

`GraphView` 提供时序图的窗口和快照功能：

```python
# snapshot() 和 window() 创建轻量视图
# 视图继承基础图的运行时上下文
```

## 时序神经网络模块

VGL 提供以下时序图专用模块：

| 模块 | 说明 |
|------|------|
| `TGNMemory` | Temporal Graph Network 记忆模块 |
| `TGATLayer` / `TGATEncoder` | 时序图注意力层 |
| `TimeEncoder` | 时间编码器 |
| `IdentityTemporalMessage` | 恒等时序消息函数 |
| `LastMessageAggregator` | 最新消息聚合器 |
| `MeanMessageAggregator` | 均值消息聚合器 |

## 下一步

- [异构图](hetero.md) — 多类型节点和边的处理
- [API 参考: vgl.nn](../api/nn.md) — 时序模块 API
