# 跨分区采样端到端示例

本示例演示一条完整的分布式采样路径:把一张大图离线切到分区目录,然后在训练时通过 `StoreBackedSamplingCoordinator` 懒加载、按需跨分区取图结构和特征,最后挂到 `DataLoader` 供采样器消费。

与 `LocalSamplingCoordinator`(把所有 `LocalGraphShard` 装进内存)相比,`StoreBackedSamplingCoordinator` 适合:

- 分区数较多、单机内存装不下全部切片。
- 训练只访问一小部分前沿、不愿意为冷分区付全量加载成本。
- 想让本地开发和分布式运行共用同一套 `DataLoader(feature_store=...)` 代码。

## 1. 离线一次:把整图切到分区目录

```python
import torch
from vgl.graph import Graph
from vgl.distributed import write_partitioned_graph

edge_index = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]])
x = torch.randn(6, 16)
graph = Graph.homo(edge_index=edge_index, x=x, num_nodes=6)

manifest = write_partitioned_graph(
    graph,
    "artifacts/partitions",
    num_partitions=2,
)
print(manifest.num_partitions)  # → 2
```

写盘后目录结构形如:

```
artifacts/partitions/
├── manifest.json
├── part-0.pt
└── part-1.pt
```

每个 `part-<id>.pt` 分区 payload 会打包该分区的子图序列化结果、全局节点 ID 映射和跨分区边界边信息;`manifest.json` 则保存分区数量、节点范围和关系元数据。

## 2. 在线:懒打开 Store 与 Coordinator

```python
from vgl.distributed import (
    StoreBackedSamplingCoordinator,
    load_partitioned_stores,
)

manifest, feature_store, graph_store = load_partitioned_stores("artifacts/partitions")
coordinator = StoreBackedSamplingCoordinator(
    manifest=manifest,
    feature_store=feature_store,
    graph_store=graph_store,
)

# 更短的等价入口
coordinator = StoreBackedSamplingCoordinator.from_partition_dir("artifacts/partitions")
```

`feature_store` / `graph_store` 此时还没有读任何分区 payload;每次 `coordinator.fetch_*(...)` 或图结构查询被调用时才按需读取对应的 `part-<id>.pt`。`manifest` 则提供分区数量、节点范围和边类型等元数据。

## 3. 挂到 DataLoader,与单机接口一致

```python
from vgl.dataloading import DataLoader, ListDataset, NodeNeighborSampler

seed = torch.tensor([0, 3])
samples = [(graph, {"seed": seed, "sample_id": "b0"})]

loader = DataLoader(
    dataset=ListDataset(samples),
    sampler=NodeNeighborSampler(num_neighbors=[10, 5]),
    batch_size=1,
    feature_store=coordinator,
)

for batch in loader:
    blocks = batch.blocks
    # 跨分区的远端前沿已经被 coordinator 自动拼接到 block 里
    ...
```

整条 `partition → store → coordinator → DataLoader` 路径与 `LocalSamplingCoordinator` 完全同接口,可以把 `LocalSamplingCoordinator({...})` 直接换成 `StoreBackedSamplingCoordinator.from_partition_dir(...)` 而不动模型代码。

## 相关 API

- `vgl.distributed.write_partitioned_graph`、`load_partitioned_stores`、`StoreBackedSamplingCoordinator` — 见 [API 参考:vgl.distributed](../api/distributed.md)
- [用户指南:采样 / 跨分区 StoreBacked](../guide/sampling.md)
