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

## 分区元数据

`write_partitioned_graph` 会在分区目录下写入 `manifest.json`,描述分区数量、节点/关系切分和每分区目录。`PartitionManifest` 是该元数据的内存表示,`PartitionShard` 描述单个分区切片的地址/大小等固定信息(和 `LocalGraphShard` 不同,`PartitionShard` 不持有张量,是 manifest 的一部分)。

::: vgl.distributed.PartitionManifest
    options:
      show_root_heading: true
      show_source: false

::: vgl.distributed.PartitionShard
    options:
      show_root_heading: true
      show_source: false

::: vgl.distributed.load_partition_manifest
    options:
      show_root_heading: true
      show_source: false

::: vgl.distributed.save_partition_manifest
    options:
      show_root_heading: true
      show_source: false

## Store 抽象

分区目录可以直接被懒加载为 Store 对象,供 coordinator 按需取图结构与特征,而不需要把所有分区都实例化成 `LocalGraphShard`。

::: vgl.distributed.DistributedGraphStore
    options:
      show_root_heading: true
      show_source: false

::: vgl.distributed.DistributedFeatureStore
    options:
      show_root_heading: true
      show_source: false

::: vgl.distributed.PartitionedGraphStore
    options:
      show_root_heading: true
      show_source: false

::: vgl.distributed.PartitionedFeatureStore
    options:
      show_root_heading: true
      show_source: false

### 本地适配器

把单个 `LocalGraphShard` 适配成 Store 协议,便于与 `StoreBackedSamplingCoordinator` 同构使用。

::: vgl.distributed.LocalGraphStoreAdapter
    options:
      show_root_heading: true
      show_source: false

::: vgl.distributed.LocalFeatureStoreAdapter
    options:
      show_root_heading: true
      show_source: false

### 加载入口

::: vgl.distributed.load_partitioned_stores
    options:
      show_root_heading: true
      show_source: false

```python
from vgl.distributed import load_partitioned_stores

manifest, feature_store, graph_store = load_partitioned_stores("artifacts/partitions")
```

## SamplingCoordinator 协议

`SamplingCoordinator` 是跨分区采样的统一协议。`LocalSamplingCoordinator` 是内存版本(持有 `LocalGraphShard`),`StoreBackedSamplingCoordinator` 则通过 Store 按需拉数据,适合分区数较多或无法一次装入内存的场景。`ShardRoute` 描述节点/边在分片中的路由结果。

::: vgl.distributed.SamplingCoordinator
    options:
      show_root_heading: true
      show_source: false

::: vgl.distributed.ShardRoute
    options:
      show_root_heading: true
      show_source: false

### StoreBackedSamplingCoordinator

::: vgl.distributed.StoreBackedSamplingCoordinator
    options:
      show_root_heading: true
      show_source: false

```python
from vgl.distributed import StoreBackedSamplingCoordinator, load_partitioned_stores

manifest, feature_store, graph_store = load_partitioned_stores("artifacts/partitions")
coordinator = StoreBackedSamplingCoordinator(
    manifest=manifest,
    feature_store=feature_store,
    graph_store=graph_store,
)

# 更短的等价入口
coordinator = StoreBackedSamplingCoordinator.from_partition_dir("artifacts/partitions")
```

与 `LocalSamplingCoordinator` 具有相同的 `partition_node_ids` / `partition_edge_ids` / `fetch_*` 接口,可无感替换到 `DataLoader(feature_store=coordinator)` 路径。
