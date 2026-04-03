# vgl.distributed

分布式模块，包含分区元数据、本地分片和采样协调器。

## 分区写入

```python
from vgl.distributed import write_partitioned_graph

manifest = write_partitioned_graph(
    graph,
    "artifacts/partitions",
    num_partitions=4,
)
```

支持同构图、时序图、单节点类型多关系图和多节点类型异构图。

## LocalGraphShard

::: vgl.distributed.LocalGraphShard
    options:
      show_root_heading: true
      show_source: false

```python
from vgl.distributed import LocalGraphShard

shard = LocalGraphShard.from_partition_dir("artifacts/partitions", partition_id=0)

local_graph = shard.graph
global_eids = shard.global_edge_index(edge_type=("node", "follows", "node"))
boundary_eids = shard.boundary_edge_index(edge_type=("node", "follows", "node"))
```

`LocalGraphShard` 的核心功能：

- 本地 ID 到全局 ID 的映射（按节点类型）
- 关系范围的全局边 ID
- 跨分区边界边的全局 ID 空间保留
- 自有本地与完整入射边前沿的恢复

## LocalSamplingCoordinator

::: vgl.distributed.LocalSamplingCoordinator
    options:
      show_root_heading: true
      show_source: false

```python
from vgl.distributed import LocalSamplingCoordinator

coordinator = LocalSamplingCoordinator({0: shard_0, 1: shard_1})

# 节点路由
node_ids = coordinator.partition_node_ids(0, node_type="paper")

# 边路由
edge_ids = coordinator.partition_edge_ids(0, edge_type=("author", "writes", "paper"))

# 特征获取
features = coordinator.fetch_edge_features(
    ("edge", ("author", "writes", "paper"), "weight"),
    edge_ids,
)

# 邻接查询
adj = coordinator.fetch_partition_adjacency(
    0, edge_type=("node", "follows", "node"), layout="csr",
)
```

## 与采样器集成

```python
from vgl.dataloading import DataLoader, ListDataset, NodeNeighborSampler

loader = DataLoader(
    dataset=ListDataset([(shard.graph, {"seed": 1, "sample_id": "s1"})]),
    sampler=NodeNeighborSampler(num_neighbors=[-1, -1]),
    batch_size=1,
    feature_store=coordinator,  # 通过 coordinator 跨分区拼接
)
```

采样器可通过 coordinator 的入射边查询自动拼接远程分区的前沿节点和边到采样子图中。支持同构、非时序异构和时序异构工作负载。
