# Graph 对象详解

`Graph` 是 VGL 中唯一的图数据抽象，统一覆盖同构图、异构图和时序图。

## 构建同构图

```python
import torch
from vgl.graph import Graph

graph = Graph.homo(
    edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
    x=torch.randn(3, 16),         # 节点特征
    y=torch.tensor([0, 1, 0]),     # 节点标签
    train_mask=torch.tensor([True, True, False]),
    val_mask=torch.tensor([False, False, True]),
    test_mask=torch.tensor([False, False, True]),
)
```

同构图还可以携带边级别张量：

```python
graph = Graph.homo(
    edge_index=edge_index,
    x=x,
    edge_data={"weight": torch.randn(num_edges)},
)
# 通过 graph.edata["weight"] 访问
```

## 构建异构图

```python
graph = Graph.hetero(
    nodes={
        "paper": {"x": torch.randn(100, 128)},
        "author": {"x": torch.randn(50, 64)},
    },
    edges={
        ("author", "writes", "paper"): {
            "edge_index": torch.randint(0, 50, (2, 200)),
        },
        ("paper", "cites", "paper"): {
            "edge_index": torch.randint(0, 100, (2, 500)),
        },
    },
)
```

## 构建时序图

```python
graph = Graph.temporal(
    nodes={"x": torch.randn(100, 32)},
    edges={
        "edge_index": torch.randint(0, 100, (2, 500)),
        "timestamp": torch.rand(500) * 100,
    },
    time_attr="timestamp",
)
```

## 节点和边数据访问

```python
# 节点数据
graph.x                    # 节点特征张量
graph.ndata["x"]           # 等价于 graph.x
graph.y                    # 节点标签

# 边数据
graph.edata["weight"]      # 边特征

# 图的基本信息
graph.num_nodes()          # 节点数
graph.num_edges()          # 边数
graph.number_of_nodes()    # 别名
graph.number_of_edges()    # 别名
```

## Schema

每个 Graph 都有一个 `GraphSchema`，描述节点类型、边类型和特征映射：

```python
from vgl.graph import GraphSchema

schema = GraphSchema(
    node_types=("paper", "author"),
    edge_types=(("author", "writes", "paper"),),
    node_features={"paper": ("x",), "author": ("x",)},
)
```

## 邻接和稀疏格式

```python
# 稀疏格式状态
graph.formats()             # 报告 coo/csr/csc 创建状态
graph.create_formats_()     # 急切创建所有允许的格式

# 邻接视图
adj = graph.adjacency(layout="coo")
adj = graph.adj(eweight_name="weight", layout="csr")  # DGL 风格加权邻接

# 拉普拉斯矩阵
lap = graph.laplacian(normalization="sym", layout="csr")

# 外部格式导出
torch_sparse = graph.adj_external(torch_fmt="csr")
scipy_sparse = graph.adj_external(scipy_fmt="coo")

# 原始张量导出
src, dst = graph.adj_tensors("coo")
crow, col, eids = graph.adj_tensors("csr")

# 关联矩阵
inc = graph.inc(typestr="in")
```

## 边查询

```python
# 查找边
graph.find_edges(edge_ids)
graph.edge_ids(src, dst)
graph.has_edges_between(src, dst)

# 邻域查询
graph.in_edges(nodes)
graph.out_edges(nodes)
graph.predecessors(node)
graph.successors(node)

# 度数
graph.in_degrees(nodes)
graph.out_degrees(nodes)
```

## 图变换

```python
# 去重平行边
simple = graph.to_simple(count_attr="count")

# 反转边方向
rev = graph.reverse()

# 子图提取
sub = graph.in_subgraph(nodes)   # 入边子图
sub = graph.out_subgraph(nodes)  # 出边子图

# 全边枚举
src, dst = graph.all_edges(order="eid")
```

## 互操作

VGL 支持与多种框架的双向转换：

```python
# PyG
from vgl.graph import Graph
graph = Graph.from_pyg(pyg_data)
pyg_data = graph.to_pyg()

# DGL
graph = Graph.from_dgl(dgl_graph)
dgl_graph = graph.to_dgl()

# NetworkX
graph = Graph.from_networkx(nx_graph)
nx_graph = graph.to_networkx()
```

## Storage 后端

对于大规模图，可以使用 storage 后端实现延迟加载：

```python
from vgl.graph import Graph, GraphSchema
from vgl.storage import FeatureStore, InMemoryGraphStore, MmapTensorStore

# mmap 后端的特征存储
MmapTensorStore.save("features/x.bin", torch.randn(1000000, 128))
feature_store = FeatureStore({
    ("node", "node", "x"): MmapTensorStore("features/x.bin"),
})

graph_store = InMemoryGraphStore(
    edges={("node", "to", "node"): edge_index},
    num_nodes={"node": 1000000},
)

graph = Graph.from_storage(
    schema=schema,
    feature_store=feature_store,
    graph_store=graph_store,
)
# 节点特征在首次访问时从 store 解析
```

## 下一步

- [节点分类](node-classification.md) — 使用 Graph 进行节点分类
- [API 参考: vgl.graph](../api/graph.md) — 完整 API 文档
