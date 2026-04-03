# 节点分类

本教程展示如何使用 VGL 完成节点分类任务。

## 全图训练

最简单的节点分类流程——在整个图上直接训练：

### 1. 准备数据

```python
from vgl.data import PlanetoidDataset
from vgl.transforms import Compose, NormalizeFeatures

dataset = PlanetoidDataset(
    root="data",
    name="Cora",
    transform=Compose([NormalizeFeatures()]),
)
graph = dataset[0]
```

### 2. 定义模型

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from vgl.nn import GCNConv

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, graph):
        x = graph.x
        x = self.conv1(x, graph)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, graph)
        return x

model = GCN(
    in_channels=graph.x.size(1),
    hidden_channels=64,
    out_channels=graph.y.max().item() + 1,
)
```

### 3. 定义任务和训练

```python
from vgl.tasks import NodeClassificationTask
from vgl.engine import Trainer, EarlyStopping, JSONLinesLogger

task = NodeClassificationTask(
    target="y",
    split=("train_mask", "val_mask", "test_mask"),
    metrics=["accuracy"],
)

trainer = Trainer(
    model=model,
    task=task,
    optimizer=torch.optim.Adam,
    lr=1e-2,
    weight_decay=5e-4,
    max_epochs=200,
    monitor="val_accuracy",
    save_best_path="artifacts/best.pt",
    callbacks=[EarlyStopping(patience=20, monitor="val_accuracy")],
    loggers=[JSONLinesLogger("artifacts/train.jsonl", flush=True)],
)

history = trainer.fit(graph, val_data=graph)
test_result = trainer.test(graph)
```

## 采样训练（大图）

对于大规模图，使用邻居采样进行 mini-batch 训练：

```python
from vgl.dataloading import DataLoader, ListDataset, NodeNeighborSampler

# 构建数据集（每个样本指定一个种子节点）
train_nodes = graph.ndata["train_mask"].nonzero(as_tuple=True)[0]
samples = [(graph, {"seed": int(n), "sample_id": f"n{n}"}) for n in train_nodes]

loader = DataLoader(
    dataset=ListDataset(samples),
    sampler=NodeNeighborSampler(num_neighbors=[15, 10]),
    batch_size=64,
)

task = NodeClassificationTask(
    target="y",
    split=("train_mask", "val_mask", "test_mask"),
    metrics=["accuracy"],
)

trainer = Trainer(
    model=model,
    task=task,
    optimizer=torch.optim.Adam,
    lr=1e-3,
    max_epochs=50,
)
trainer.fit(loader)
```

## Block 模式

使用 `output_blocks=True` 可以获取 DGL 风格的消息流 Block：

```python
loader = DataLoader(
    dataset=ListDataset(samples),
    sampler=NodeNeighborSampler(num_neighbors=[15, 10], output_blocks=True),
    batch_size=64,
)

# batch.blocks[0], batch.blocks[1] 按外层到内层排序
```

## 更换卷积层

只需替换模型中的卷积层即可切换算法：

```python
from vgl.nn import SAGEConv, GATConv, GINConv

# GraphSAGE
conv = SAGEConv(in_channels, out_channels)

# GAT
conv = GATConv(in_channels, out_channels, num_heads=8)

# GIN
conv = GINConv(nn.Sequential(
    nn.Linear(in_channels, hidden_channels),
    nn.ReLU(),
    nn.Linear(hidden_channels, out_channels),
))
```

## 下一步

- [图分类](graph-classification.md) — 对多个图进行分类
- [采样策略](sampling.md) — 深入了解各种采样器
- [卷积层速查](../examples/conv-zoo.md) — 60+ 卷积层一览
