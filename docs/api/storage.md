# vgl.storage

特征存储和图存储模块，支持大规模图的延迟加载和 mmap 后端。

## FeatureStore

::: vgl.storage.FeatureStore
    options:
      show_root_heading: true
      show_source: false

## InMemoryGraphStore

::: vgl.storage.InMemoryGraphStore
    options:
      show_root_heading: true
      show_source: false

## MmapTensorStore

基于内存映射的张量存储，适用于大规模特征表：

::: vgl.storage.mmap.MmapTensorStore
    options:
      show_root_heading: true
      show_source: false

## 使用方式

```python
import torch
from vgl.storage import FeatureStore, InMemoryGraphStore, MmapTensorStore

# 保存大规模特征到 mmap 文件
MmapTensorStore.save("features/x.bin", torch.randn(1000000, 128))

# 创建特征存储
feature_store = FeatureStore({
    ("node", "node", "x"): MmapTensorStore("features/x.bin"),
})

# 创建图结构存储
graph_store = InMemoryGraphStore(
    edges={("node", "to", "node"): edge_index},
    num_nodes={"node": 1000000},
)
```

与 `Graph.from_storage()` 配合使用：

```python
from vgl.graph import Graph, GraphSchema

graph = Graph.from_storage(
    schema=schema,
    feature_store=feature_store,
    graph_store=graph_store,
)
# graph.feature_store 保留原始来源，供后续 plan 执行使用
```
