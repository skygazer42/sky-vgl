# 异构图

本教程展示如何使用 VGL 处理异构图（包含多种节点类型和边类型的图）。

## 构建异构图

```python
import torch
from vgl.graph import Graph

graph = Graph.hetero(
    nodes={
        "paper": {"x": torch.randn(1000, 128)},
        "author": {"x": torch.randn(500, 64)},
        "venue": {"x": torch.randn(50, 32)},
    },
    edges={
        ("author", "writes", "paper"): {
            "edge_index": torch.randint(0, 500, (2, 3000)),
        },
        ("paper", "published_in", "venue"): {
            "edge_index": torch.randint(0, min(1000, 50), (2, 1000)),
        },
        ("paper", "cites", "paper"): {
            "edge_index": torch.randint(0, 1000, (2, 5000)),
        },
    },
)
```

## 异构卷积层

VGL 提供多种关系感知算子：

### RGCN

```python
from vgl.nn import RGCNConv

conv = RGCNConv(
    in_channels=128,
    out_channels=64,
    num_relations=3,
)
```

### HGT (Heterogeneous Graph Transformer)

```python
from vgl.nn import HGTConv

conv = HGTConv(
    in_channels={"paper": 128, "author": 64, "venue": 32},
    out_channels=64,
    num_heads=4,
)
```

### HAN (Heterogeneous Attention Network)

```python
from vgl.nn import HANConv

conv = HANConv(
    in_channels={"paper": 128, "author": 64},
    out_channels=64,
    num_heads=4,
)
```

## 异构图节点分类

```python
from vgl.tasks import NodeClassificationTask
from vgl.engine import Trainer

# 假设我们要分类 paper 节点
task = NodeClassificationTask(
    target="y",
    split=("train_mask", "val_mask", "test_mask"),
    metrics=["accuracy"],
)

trainer = Trainer(
    model=hetero_model,
    task=task,
    optimizer=torch.optim.Adam,
    lr=1e-3,
    max_epochs=100,
)
trainer.fit(graph, val_data=graph)
```

## HeteroBlock

对于异构图的采样，当一个被监督的节点类型有零个或多个入边关系时，采样器会生成 `HeteroBlock`（而非 `Block`）：

```python
from vgl.dataloading import NodeNeighborSampler

sampler = NodeNeighborSampler(
    num_neighbors=[15, 10],
    output_blocks=True,
)
# batch.blocks 可能包含 Block 或 HeteroBlock
```

`HeteroBlock` 支持 DGL 互操作：

```python
dgl_block = hetero_block.to_dgl()
hetero_block = HeteroBlock.from_dgl(dgl_block)
```

## 异构图的采样和分区

异构图同样支持分区和分片：

```python
from vgl.distributed import write_partitioned_graph, LocalGraphShard, LocalSamplingCoordinator

manifest = write_partitioned_graph(graph, "partitions", num_partitions=2)
shard = LocalGraphShard.from_partition_dir("partitions", partition_id=0)
coordinator = LocalSamplingCoordinator({0: shard})
```

采样器可以通过 coordinator 跨分区拼接异构前沿结构。

## 下一步

- [采样策略](sampling.md) — 各种采样器的详细对比
- [API 参考: vgl.graph](../api/graph.md) — HeteroBlock API
