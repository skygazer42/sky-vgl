# 图分类

本教程展示如何对多个小图进行分类。

## 多图批量训练

### 1. 准备数据

使用内置 TUDataset：

```python
from vgl.data import TUDataset
from vgl.transforms import Compose, RandomGraphSplit

dataset = TUDataset(root="data", name="MUTAG")
# dataset 包含多个图，每个图带有 graph.y 标签
```

手动构建样本列表：

```python
from vgl.dataloading import SampleRecord, ListDataset

samples = [
    SampleRecord(graph=g, metadata={}, sample_id=f"g{i}")
    for i, g in enumerate(graphs)
]
dataset = ListDataset(samples)
```

### 2. 创建 DataLoader

```python
from vgl.dataloading import DataLoader, FullGraphSampler

loader = DataLoader(
    dataset=dataset,
    sampler=FullGraphSampler(),
    batch_size=32,
    label_source="graph",
    label_key="y",
)
```

### 3. 定义模型

图分类模型需要一个图级别的池化（readout）层：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from vgl.nn import GINConv, global_mean_pool

class GINClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        mlp1 = nn.Sequential(nn.Linear(in_channels, hidden_channels), nn.ReLU())
        mlp2 = nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.ReLU())
        self.conv1 = GINConv(mlp1)
        self.conv2 = GINConv(mlp2)
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, batch):
        x = batch.graph.x
        x = self.conv1(x, batch.graph)
        x = F.relu(x)
        x = self.conv2(x, batch.graph)
        x = global_mean_pool(x, batch.graph_index)
        return self.classifier(x)
```

### 4. 训练

```python
from vgl.tasks import GraphClassificationTask
from vgl.engine import Trainer

task = GraphClassificationTask(target="y", label_source="graph")

trainer = Trainer(
    model=model,
    task=task,
    optimizer=torch.optim.Adam,
    lr=1e-3,
    max_epochs=100,
)
trainer.fit(loader)
```

## 从元数据读取标签

当标签存储在样本 metadata 中时：

```python
samples = [
    SampleRecord(graph=g, metadata={"label": label}, sample_id=f"g{i}")
    for i, (g, label) in enumerate(zip(graphs, labels))
]

loader = DataLoader(
    dataset=ListDataset(samples),
    sampler=FullGraphSampler(),
    batch_size=32,
    label_source="metadata",
    label_key="label",
)

task = GraphClassificationTask(target="label", label_source="metadata")
```

## 子图采样分类

从一个大图中采样子图并分类：

```python
from vgl.dataloading import NodeSeedSubgraphSampler

dataset = ListDataset([
    (source_graph, {"seed": 1, "label": 1, "sample_id": "s1"}),
    (source_graph, {"seed": 2, "label": 0, "sample_id": "s2"}),
])

loader = DataLoader(
    dataset=dataset,
    sampler=NodeSeedSubgraphSampler(),
    batch_size=32,
    label_source="metadata",
    label_key="label",
)

task = GraphClassificationTask(target="label", label_source="metadata")
trainer.fit(loader)
```

## 池化函数

VGL 提供三种图级别池化：

```python
from vgl.nn import global_mean_pool, global_sum_pool, global_max_pool

# 均值池化
out = global_mean_pool(x, batch.graph_index)

# 求和池化
out = global_sum_pool(x, batch.graph_index)

# 最大值池化
out = global_max_pool(x, batch.graph_index)
```

对于异构图的 batch，使用按节点类型的成员索引：

```python
paper_pool = global_mean_pool(x_paper, batch.graph_index_by_type["paper"])
```

## 下一步

- [链接预测](link-prediction.md) — 预测图中的缺失边
- [训练器与回调](training.md) — Trainer 的高级配置
