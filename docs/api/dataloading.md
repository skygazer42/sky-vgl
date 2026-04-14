# vgl.dataloading

数据加载模块，包含 DataLoader、采样器和数据集类型。

## Loader 参数边界

`Loader` 的 buffering / backpressure 参数有明确边界：

- `prefetch` 只用于 `num_workers == 0` 的单线程路径
- `prefetch_factor` 只用于 `num_workers > 0`
- `persistent_workers=True` 也只适用于 `num_workers > 0`
- 不要在 `num_workers>0` 时设置 `prefetch`

推荐做法：

- 单线程调试或轻量采样：`num_workers=0`，按需设置 `prefetch`
- 多 worker 吞吐路径：`num_workers>0`，配合 `prefetch_factor`
- 只有当 worker 需要跨 epoch 复用时，再开启 `persistent_workers=True`

## DataLoader

::: vgl.dataloading.DataLoader
    options:
      show_root_heading: true
      show_source: false

## Loader

::: vgl.dataloading.Loader
    options:
      show_root_heading: true
      show_source: false

### Loader 预取参数约束

`Loader` / `DataLoader` 的预取配置现在有明确边界：

- `prefetch` 只用于 `num_workers == 0` 的单线程路径
- `prefetch_factor` 只用于 `num_workers > 0` 的 worker 预取
- `persistent_workers=True` 也只适用于 `num_workers > 0`
- 当 `num_workers > 0` 时不要再设置 `prefetch`

推荐记忆方式：

| 场景 | 参数 |
|------|------|
| 单线程顺序调试 | `num_workers=0`, 可选 `prefetch>0` |
| worker 并行吞吐 | `num_workers>0`, 可选 `prefetch_factor` |
| 复用 worker 进程 | `num_workers>0`, `persistent_workers=True` |

更完整的示例和回压说明，见[采样指南](../guide/sampling.md)。

## 数据集

### ListDataset

::: vgl.dataloading.ListDataset
    options:
      show_root_heading: true
      show_source: false

## 样本记录

### SampleRecord

::: vgl.dataloading.SampleRecord
    options:
      show_root_heading: true
      show_source: false

### LinkPredictionRecord

::: vgl.dataloading.LinkPredictionRecord
    options:
      show_root_heading: true
      show_source: false

### TemporalEventRecord

::: vgl.dataloading.TemporalEventRecord
    options:
      show_root_heading: true
      show_source: false

## 采样器

### FullGraphSampler

::: vgl.dataloading.FullGraphSampler
    options:
      show_root_heading: true
      show_source: false

### NodeNeighborSampler

::: vgl.dataloading.NodeNeighborSampler
    options:
      show_root_heading: true
      show_source: false

### LinkNeighborSampler

::: vgl.dataloading.LinkNeighborSampler
    options:
      show_root_heading: true
      show_source: false

### TemporalNeighborSampler

::: vgl.dataloading.TemporalNeighborSampler
    options:
      show_root_heading: true
      show_source: false

### NodeSeedSubgraphSampler

::: vgl.dataloading.NodeSeedSubgraphSampler
    options:
      show_root_heading: true
      show_source: false

### GraphSAINT 采样器

::: vgl.dataloading.GraphSAINTNodeSampler
    options:
      show_root_heading: true
      show_source: false

::: vgl.dataloading.GraphSAINTEdgeSampler
    options:
      show_root_heading: true
      show_source: false

::: vgl.dataloading.GraphSAINTRandomWalkSampler
    options:
      show_root_heading: true
      show_source: false

### 随机游走采样器

::: vgl.dataloading.RandomWalkSampler
    options:
      show_root_heading: true
      show_source: false

::: vgl.dataloading.Node2VecWalkSampler
    options:
      show_root_heading: true
      show_source: false

### ShaDowKHopSampler

::: vgl.dataloading.ShaDowKHopSampler
    options:
      show_root_heading: true
      show_source: false

### 链接预测采样器

::: vgl.dataloading.UniformNegativeLinkSampler
    options:
      show_root_heading: true
      show_source: false

::: vgl.dataloading.HardNegativeLinkSampler
    options:
      show_root_heading: true
      show_source: false

::: vgl.dataloading.CandidateLinkSampler
    options:
      show_root_heading: true
      show_source: false

### ClusterGCN

::: vgl.dataloading.ClusterData
    options:
      show_root_heading: true
      show_source: false

::: vgl.dataloading.ClusterLoader
    options:
      show_root_heading: true
      show_source: false
