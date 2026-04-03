# FAQ

## 安装

### VGL 支持哪些 Python 版本？

Python 3.10 及以上版本。

### VGL 需要 GPU 吗？

不需要。VGL 基于 PyTorch，可以在 CPU 上运行。如果你有 NVIDIA GPU，安装对应 CUDA 版本的 PyTorch 可以加速训练。

### 如何安装所有可选依赖？

```bash
pip install "sky-vgl[full]"
```

## Graph

### VGL 的 Graph 和 DGL/PyG 的区别是什么？

VGL 使用一个统一的 `Graph` 类同时表示同构图、异构图和时序图，而不是为每种图类型使用不同的类。这简化了 API 并确保了一致的行为。

### 如何从 DGL 或 PyG 迁移？

参见 [迁移指南](migration-guide.md)。核心的迁移方式是：

- DGL: `Graph.from_dgl(dgl_graph)`
- PyG: `Graph.from_pyg(pyg_data)`

### Graph 支持哪些稀疏格式？

COO、CSR 和 CSC。使用 `graph.formats()` 查看当前状态，`graph.create_formats_()` 急切创建所有格式。

## 训练

### 如何使用早停？

```python
from vgl.engine import Trainer, EarlyStopping

trainer = Trainer(
    ...,
    callbacks=[EarlyStopping(patience=20, monitor="val_accuracy")],
)
```

### Trainer 支持哪些日志格式？

- 控制台日志（默认启用）
- JSONL (`JSONLinesLogger`)
- CSV (`CSVLogger`)
- TensorBoard (`TensorBoardLogger`)

### 如何保存和加载最佳模型？

```python
trainer = Trainer(..., monitor="val_accuracy", save_best_path="best.pt")
history = trainer.fit(graph, val_data=graph)

from vgl.engine import restore_checkpoint
model = restore_checkpoint(model, "best.pt")
```

## 采样

### 大图应该使用哪种采样器？

- **节点分类**: `NodeNeighborSampler`
- **链接预测**: `LinkNeighborSampler`
- **时序预测**: `TemporalNeighborSampler`
- **需要子图归一化**: `GraphSAINTNodeSampler` / `GraphSAINTEdgeSampler`
- **预分区加速**: `ClusterData` + `ClusterLoader`

### NodeNeighborSampler 的 num_neighbors 怎么设？

`num_neighbors` 是一个列表，每个元素对应一层的采样邻居数。例如 `[15, 10]` 表示第一层采 15 个邻居，第二层采 10 个。使用 `-1` 表示采样所有邻居。

## 模型

### VGL 有多少种卷积层？

60+ 种，包括同构卷积（GCN、GAT、SAGE 等）、异构卷积（RGCN、HGT、HAN）、边特征卷积（NNConv、GINEConv）和 Graph Transformer。完整列表见 [卷积层速查](examples/conv-zoo.md)。

### 如何做图级别的 pooling？

```python
from vgl.nn import global_mean_pool, global_sum_pool, global_max_pool

graph_embedding = global_mean_pool(node_embeddings, batch.graph_index)
```

## 数据

### 内置哪些数据集？

- **引文网络**: Cora、Citeseer、PubMed (`PlanetoidDataset`)
- **图分类**: MUTAG、PROTEINS、ENZYMES 等 (`TUDataset`)
- **入门**: Karate Club (`KarateClubDataset`)

### 如何使用自定义数据集？

直接构建 `Graph` 对象即可：

```python
graph = Graph.homo(edge_index=your_edges, x=your_features, y=your_labels)
```

或使用 `OnDiskGraphDataset` 将多个图序列化到磁盘。
