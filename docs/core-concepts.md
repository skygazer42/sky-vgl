# 核心概念

本文介绍 VGL 的核心抽象和设计理念。

## Graph

`Graph` 是 VGL 的核心数据结构。同构图是异构图和时序图的特例，三者共享同一个抽象。

### 同构图

```python
from vgl.graph import Graph
import torch

graph = Graph.homo(
    edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
    x=torch.randn(3, 16),
    y=torch.tensor([0, 1, 0]),
)
```

同构图可以通过 `Graph.homo(edge_data={...})` 携带边级别张量，通过 `graph.edata` 访问。

### 异构图

```python
graph = Graph.hetero(
    nodes={"paper": {"x": features}, "author": {"x": features}},
    edges={("author", "writes", "paper"): {"edge_index": edge_index}},
)
```

### 时序图

```python
graph = Graph.temporal(
    nodes={"x": features},
    edges={"edge_index": edge_index, "timestamp": timestamps},
    time_attr="timestamp",
)
```

### 互操作

`Graph` 支持与 PyG、DGL、NetworkX、CSV 的双向转换：

- `Graph.from_pyg()` / `graph.to_pyg()`
- `Graph.from_dgl()` / `graph.to_dgl()`
- `Graph.from_networkx()` / `graph.to_networkx()`

### Storage 后端

`Graph.from_storage(schema=..., feature_store=..., graph_store=...)` 从特征/图存储构建图视图，结构数据立即可用，特征延迟加载。

## SparseTensor 和邻接缓存

`vgl.sparse` 是底层稀疏执行引擎，提供 `SparseTensor`、COO/CSR/CSC 转换、稀疏运算（`spmm`、`sddmm`、`edge_softmax`）。

`Graph.adjacency(layout=...)` 是用户代码访问邻接结构的主接口。它通过 `vgl.sparse` 构建稀疏视图并缓存，避免重复构建。

## GraphView

`GraphView` 是已有图的轻量投影，用于 `snapshot()` 和 `window()` 等时序操作。视图继承基础图的运行时上下文。

## Block 和 HeteroBlock

`Block` 是 `to_block()` 返回的紧凑关系局部消息流容器。它包装一个二部图加上显式的源/目标节点 ID 元数据。

`HeteroBlock` 是多关系对应物，包装一个异构二部层，按原始节点类型保存 `src_n_id` 和 `dst_n_id`。

两者都支持 DGL 互操作（`to_dgl()` / `from_dgl()`）。

## GraphBatch

`GraphBatch` 将多个图分组为一个训练输入，跟踪节点到图的成员关系：

- 同构图：`graph_index` 和 `graph_ptr`
- 异构/时序图：`graph_index_by_type` 和 `graph_ptr_by_type`

## MessagePassing

`MessagePassing` 是图卷积的底层神经原语。`GCNConv`、`SAGEConv`、`GATConv` 等 60+ 种卷积层都基于它构建。

## SampleRecord

`SampleRecord` 是图分类的结构化预合并单元，携带图、元数据和样本标识。它让多小图输入和采样子图输入收敛到相同的 batch 合约。

## Task 和 Trainer

`Task` 定义监督合约，拥有 `loss()`、`targets()`、`predictions_for_metrics()` 方法。

`Trainer` 运行优化循环，不拥有图核心抽象。支持：

- `fit(train, val=None)` — 训练
- `evaluate(data)` — 验证
- `test(data)` — 测试
- `loggers=[...]` — 多种日志后端
- `callbacks=[...]` — 训练回调
- `monitor` — 指标监控和最佳模型保存

支持的任务类型：

- 节点分类 (`NodeClassificationTask`)
- 图分类 (`GraphClassificationTask`)
- 链接预测 (`LinkPredictionTask`)
- 时序事件预测 (`TemporalEventPredictionTask`)

## LinkPredictionRecord 和 TemporalEventRecord

`LinkPredictionRecord` 是链接预测的显式候选边单元，`TemporalEventRecord` 是时序预测的显式候选事件单元。它们分别合并为 `LinkPredictionBatch` 和 `TemporalEventBatch`。

## 采样计划

邻居采样通过 `SamplingPlan` 阶段执行。公共采样器在内部构建计划、执行扩展和特征获取阶段、然后将结果物化回统一的 batch 合约。这保持了用户 API 稳定，同时为大图运行时、特征存储和分片协调打开了路径。

## 数据集基础设施

`vgl.data` 提供 `DatasetManifest`、`DatasetSplit`、`DatasetRegistry` 和 `OnDiskGraphDataset`。

`vgl.distributed` 提供 `PartitionManifest`、`LocalGraphShard` 和 `LocalSamplingCoordinator`，支持同构、时序、单节点类型多关系和多节点类型异构图的本地优先分区工作流。
