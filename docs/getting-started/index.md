---
hide:
  - toc
---

# 快速开始

欢迎使用 VGL！按照以下步骤，几分钟内即可运行你的第一个图学习任务。

<div class="grid" markdown>

<div class="card" markdown>

### :material-download: 安装指南

安装 VGL 及其依赖。

[:octicons-arrow-right-24: 安装](installation.md)

</div>

<div class="card" markdown>

### :material-play-circle: 5 分钟快速入门

从零开始完成一个节点分类任务。

[:octicons-arrow-right-24: 快速入门](quickstart.md)

</div>

</div>

## 最小示例

```python
pip install sky-vgl
```

```python
import torch
from vgl.graph import Graph
from vgl.tasks import NodeClassificationTask
from vgl.engine import Trainer

# 1. 构建图
graph = Graph.homo(
    edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
    x=torch.randn(3, 16),
    y=torch.tensor([0, 1, 0]),
    train_mask=torch.tensor([True, True, False]),
    val_mask=torch.tensor([False, False, True]),
    test_mask=torch.tensor([False, False, True]),
)

# 2. 定义任务
task = NodeClassificationTask(
    target="y",
    split=("train_mask", "val_mask", "test_mask"),
    metrics=["accuracy"],
)

# 3. 定义模型
model = torch.nn.Sequential(
    torch.nn.Linear(16, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 2),
)

# 4. 训练
trainer = Trainer(model=model, task=task, optimizer=torch.optim.Adam, lr=1e-3, max_epochs=10)
history = trainer.fit(graph, val_data=graph)
```
