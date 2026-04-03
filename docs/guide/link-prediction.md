# 链接预测

本教程展示如何使用 VGL 进行链接预测任务。

## 基本流程

链接预测使用显式的候选边进行训练，每条候选边都是一个 `LinkPredictionRecord`。

### 1. 准备数据

```python
import torch
from vgl.graph import Graph
from vgl.dataloading import LinkPredictionRecord, ListDataset, DataLoader, FullGraphSampler

graph = Graph.homo(edge_index=edge_index, x=x)

# 构建正样本和负样本
samples = [
    LinkPredictionRecord(graph=graph, src_index=0, dst_index=1, label=1),
    LinkPredictionRecord(graph=graph, src_index=2, dst_index=0, label=0),
]

loader = DataLoader(
    dataset=ListDataset(samples),
    sampler=FullGraphSampler(),
    batch_size=64,
)
```

### 2. 定义模型

```python
import torch.nn as nn
from vgl.nn import SAGEConv

class LinkPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

    def forward(self, batch):
        x = batch.graph.x
        x = self.conv1(x, batch.graph).relu()
        x = self.conv2(x, batch.graph)
        # 使用源节点和目标节点嵌入的点积作为预测
        src_emb = x[batch.src_index]
        dst_emb = x[batch.dst_index]
        return (src_emb * dst_emb).sum(dim=-1)
```

### 3. 训练

```python
from vgl.tasks import LinkPredictionTask
from vgl.engine import Trainer

task = LinkPredictionTask(target="label")

trainer = Trainer(
    model=model,
    task=task,
    optimizer=torch.optim.Adam,
    lr=1e-3,
    max_epochs=50,
)
trainer.fit(loader)
```

## 负采样

### 均匀负采样

```python
from vgl.dataloading import UniformNegativeLinkSampler

loader = DataLoader(
    dataset=ListDataset(positive_samples),
    sampler=UniformNegativeLinkSampler(num_negatives=5),
    batch_size=64,
)
```

### 硬负采样

```python
from vgl.dataloading import HardNegativeLinkSampler

loader = DataLoader(
    dataset=ListDataset(positive_samples),
    sampler=HardNegativeLinkSampler(
        num_negatives=5,
        hard_negative_dst_metadata_key="hard_pool",
    ),
    batch_size=64,
)
```

### 候选排名评估

```python
from vgl.dataloading import CandidateLinkSampler

eval_loader = DataLoader(
    dataset=ListDataset(eval_samples),
    sampler=CandidateLinkSampler(
        candidate_dst_metadata_key="candidate_pool",
    ),
    batch_size=64,
)
```

## 自动切分数据集

使用 `RandomLinkSplit` 自动将边切分为训练/验证/测试集：

```python
from vgl.transforms import RandomLinkSplit

# RandomLinkSplit 创建 LinkPredictionRecord 数据集
# 自动生成负样本并保持 query_id 分组
```

## 邻居采样链接预测

对于大图，使用 `LinkNeighborSampler`：

```python
from vgl.dataloading import LinkNeighborSampler

loader = DataLoader(
    dataset=ListDataset(samples),
    sampler=LinkNeighborSampler(num_neighbors=[15, 10]),
    batch_size=64,
)
```

使用 `output_blocks=True` 获取消息流 Block：

```python
sampler = LinkNeighborSampler(
    num_neighbors=[15, 10],
    output_blocks=True,
)
# batch.blocks 可用于 block 感知的模型
```

## 评估指标

```python
from vgl.metrics import MRR, HitsAtK, FilteredMRR, FilteredHitsAtK

# MRR (Mean Reciprocal Rank)
# HitsAtK (Hits@K)
# FilteredMRR / FilteredHitsAtK (过滤版本)
```

## 下一步

- [时序图](temporal.md) — 时序事件预测
- [采样策略](sampling.md) — 各种采样器详解
