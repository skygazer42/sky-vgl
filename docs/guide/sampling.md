# 采样策略

VGL 提供多种图采样策略，适用于不同场景的大规模图训练。

## 采样器对比

| 采样器 | 类型 | 适用场景 | 特点 |
|--------|------|----------|------|
| `FullGraphSampler` | 全图 | 小图、图分类 | 不采样，返回完整图 |
| `NodeNeighborSampler` | 邻居采样 | 节点分类（大图） | 逐跳采样固定数量邻居 |
| `LinkNeighborSampler` | 邻居采样 | 链接预测（大图） | 链接预测专用邻居采样 |
| `TemporalNeighborSampler` | 时序采样 | 时序预测 | 保证严格时间顺序 |
| `NodeSeedSubgraphSampler` | 种子子图 | 子图级任务 | 以种子节点诱导子图 |
| `GraphSAINTNodeSampler` | GraphSAINT | 大图训练 | 基于节点的子图采样 |
| `GraphSAINTEdgeSampler` | GraphSAINT | 大图训练 | 基于边的子图采样 |
| `GraphSAINTRandomWalkSampler` | GraphSAINT | 大图训练 | 基于随机游走的子图采样 |
| `RandomWalkSampler` | 随机游走 | 无监督嵌入 | 单次或多次随机游走 |
| `Node2VecWalkSampler` | Node2Vec | 无监督嵌入 | 带参数 p/q 的偏置游走 |
| `ShaDowKHopSampler` | K 跳子图 | 节点级任务 | K 跳诱导子图采样 |
| `ClusterData` / `ClusterLoader` | 图分区 | 大图训练 | 预分区后按 cluster 加载 |

## NodeNeighborSampler

最常用的节点采样器，按层采样固定数量的邻居：

```python
from vgl.dataloading import NodeNeighborSampler

sampler = NodeNeighborSampler(
    num_neighbors=[15, 10],  # 第一层采 15 个，第二层采 10 个
)
```

使用 `-1` 表示采样所有邻居：

```python
sampler = NodeNeighborSampler(num_neighbors=[-1, -1])
```

### Block 输出

```python
sampler = NodeNeighborSampler(
    num_neighbors=[15, 10],
    output_blocks=True,
)
# batch.blocks 为每层提供 Block 或 HeteroBlock
```

### 带特征获取

```python
sampler = NodeNeighborSampler(
    num_neighbors=[15, 10],
    node_feature_names=["x"],
    edge_feature_names=["weight"],
)
```

## GraphSAINT 系列

GraphSAINT 采样器从全图中采样一个小子图进行训练：

```python
from vgl.dataloading import (
    GraphSAINTNodeSampler,
    GraphSAINTEdgeSampler,
    GraphSAINTRandomWalkSampler,
)

# 节点采样
node_sampler = GraphSAINTNodeSampler(budget=1000)

# 边采样
edge_sampler = GraphSAINTEdgeSampler(budget=2000)

# 随机游走采样
rw_sampler = GraphSAINTRandomWalkSampler(budget=1000, walk_length=4)
```

这些采样器暴露 `seed_positions` 用于将种子节点映射回子图中的位置，以及 `sampled_num_nodes`、`sampled_num_edges` 用于记录子图大小。

## ClusterGCN

预先将图分区，然后按 cluster 加载：

```python
from vgl.dataloading import ClusterData, ClusterLoader

cluster_data = ClusterData(graph, num_parts=100)
loader = ClusterLoader(cluster_data, batch_size=10)
# 每个 batch 由 10 个 cluster 组成
```

## 随机游走采样

```python
from vgl.dataloading import RandomWalkSampler, Node2VecWalkSampler

# 基础随机游走
rw_sampler = RandomWalkSampler(walk_length=10, num_walks=5)

# Node2Vec 偏置游走
n2v_sampler = Node2VecWalkSampler(walk_length=10, num_walks=5, p=1.0, q=1.0)
```

游走采样器暴露丰富的元数据：

- `walks` — 游走序列
- `walk_lengths` — 每条游走的有效长度
- `walk_nodes` — 所有被访问的唯一节点
- `walk_starts` — 游走起始节点
- `walk_ended_early` — 是否提前终止
- `walk_edge_pairs` — 游走路径上的边

## ShaDowKHopSampler

以种子节点为中心，提取 K 跳诱导子图：

```python
from vgl.dataloading import ShaDowKHopSampler

sampler = ShaDowKHopSampler(num_hops=2, num_neighbors=10)
```

## 与 DataLoader 配合

所有采样器都通过统一的 `DataLoader` 接口使用：

```python
from vgl.dataloading import DataLoader, ListDataset

loader = DataLoader(
    dataset=ListDataset(samples),
    sampler=sampler,
    batch_size=64,
)

for batch in loader:
    # 根据采样器类型，batch 可能是 NodeBatch、GraphBatch 等
    pass
```

## 分布式采样

配合 `LocalSamplingCoordinator`，采样器可以跨分区拼接远程前沿节点：

```python
from vgl.distributed import LocalSamplingCoordinator

coordinator = LocalSamplingCoordinator({0: shard_0, 1: shard_1})

loader = DataLoader(
    dataset=ListDataset(samples),
    sampler=NodeNeighborSampler(num_neighbors=[-1, -1]),
    batch_size=64,
    feature_store=coordinator,
)
```

## 下一步

- [训练器与回调](training.md) — 配置训练参数
- [API 参考: vgl.dataloading](../api/dataloading.md) — 采样器完整 API
