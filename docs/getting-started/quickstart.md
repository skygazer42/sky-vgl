# 5 分钟快速入门

本教程将带你从零开始完成一个完整的节点分类任务。

## 工作流概览

VGL 的核心工作流只有四步：

1. **构建 Graph** — 用 `Graph.homo()` / `Graph.hetero()` / `Graph.temporal()` 创建图
2. **定义 Task** — 指定监督信号和评估指标
3. **构建模型** — 标准 PyTorch `nn.Module`
4. **使用 Trainer 训练** — 自动处理训练循环、验证和评估

## 节点分类示例

### 1. 导入和数据准备

```python
import torch
from vgl.graph import Graph
from vgl.nn import GCNConv
from vgl.tasks import NodeClassificationTask
from vgl.engine import Trainer
```

使用内置的 Cora 数据集：

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

也可以手动构建图：

```python
graph = Graph.homo(
    edge_index=edge_index,   # shape: [2, num_edges]
    x=features,              # shape: [num_nodes, num_features]
    y=labels,                # shape: [num_nodes]
    train_mask=train_mask,   # shape: [num_nodes], bool
    val_mask=val_mask,
    test_mask=test_mask,
)
```

### 2. 定义模型

```python
import torch.nn as nn
import torch.nn.functional as F

class GCNModel(nn.Module):
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

model = GCNModel(
    in_channels=graph.x.size(1),
    hidden_channels=64,
    out_channels=graph.y.max().item() + 1,
)
```

### 3. 定义任务

```python
task = NodeClassificationTask(
    target="y",
    split=("train_mask", "val_mask", "test_mask"),
    metrics=["accuracy"],
)
```

### 4. 训练

```python
trainer = Trainer(
    model=model,
    task=task,
    optimizer=torch.optim.Adam,
    lr=1e-2,
    max_epochs=200,
    monitor="val_accuracy",
    save_best_path="artifacts/best.pt",
)

history = trainer.fit(graph, val_data=graph)
```

### 5. 评估

```python
test_result = trainer.test(graph)
print(test_result)
```

加载最佳 checkpoint：

```python
from vgl.engine import load_checkpoint, restore_checkpoint

best_state = load_checkpoint("artifacts/best.pt")
model = restore_checkpoint(model, "artifacts/best.pt")
```

## 图分类示例

对多个小图进行分类：

```python
from vgl.dataloading import DataLoader, ListDataset, FullGraphSampler, SampleRecord
from vgl.tasks import GraphClassificationTask

# 准备样本
samples = [
    SampleRecord(graph=graph_a, metadata={}, sample_id="a"),
    SampleRecord(graph=graph_b, metadata={}, sample_id="b"),
]

# 创建 DataLoader
loader = DataLoader(
    dataset=ListDataset(samples),
    sampler=FullGraphSampler(),
    batch_size=32,
    label_source="graph",
    label_key="y",
)

# 训练
task = GraphClassificationTask(target="y", label_source="graph")
trainer = Trainer(model=model, task=task, optimizer=torch.optim.Adam, lr=1e-3, max_epochs=50)
trainer.fit(loader)
```

## 使用内置数据集

VGL 提供多个常用数据集：

```python
from vgl.data import PlanetoidDataset, TUDataset, KarateClubDataset

# 引文网络
cora = PlanetoidDataset(root="data", name="Cora")
citeseer = PlanetoidDataset(root="data", name="Citeseer")
pubmed = PlanetoidDataset(root="data", name="PubMed")

# 图分类数据集
mutag = TUDataset(root="data", name="MUTAG")
proteins = TUDataset(root="data", name="PROTEINS")

# 零下载入门数据集
karate = KarateClubDataset(root="data")
```

也可以使用字符串驱动的注册表：

```python
from vgl.data import DatasetRegistry
registry = DatasetRegistry.default()
cora = registry["cora"]
mutag = registry["tu:mutag"]
```

## 下一步

- [Graph 对象详解](../guide/graph.md) — 深入了解 Graph 的构建和操作
- [用户指南](../guide/index.md) — 完整的教程和最佳实践
- [API 参考](../api/index.md) — 所有公共 API 的详细文档
