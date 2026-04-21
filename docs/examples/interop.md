# 框架互操作示例

本示例汇总 VGL 与 DGL、PyG、NetworkX、`torch.sparse`、CSV 之间的最小可运行往返代码,便于在迁移或联合工作流中快速对位。

所有示例都假设已经构建出一张同构 `Graph`:

```python
import torch
from vgl.graph import Graph

edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
graph = Graph.homo(
    edge_index=edge_index,
    x=torch.randn(3, 8),
    y=torch.tensor([0, 1, 0]),
)
```

## DGL

```python
dgl_graph = graph.to_dgl()           # dgl.graph(...) 或 dgl.heterograph(...)
recovered = Graph.from_dgl(dgl_graph)
```

- 同构图导出为 `dgl.graph`;异构或时序图导出为 `dgl.heterograph`。
- 异构图规范边类型、`dgl.NID` / `dgl.EID`、时序 `vgl_time_attr` 都在往返中保留。
- Block 级别:`Block.from_dgl` / `HeteroBlock.from_dgl` 或 `vgl.compat.block_{to,from}_dgl` / `hetero_block_{to,from}_dgl`。

## PyG

```python
pyg_data = graph.to_pyg()            # torch_geometric.data.Data / HeteroData
recovered = Graph.from_pyg(pyg_data)
```

- 同构图走 `Data`,异构图走 `HeteroData`。
- 常见字段(`x`、`y`、`edge_attr`、`train_mask` 等)按名称直通。

## NetworkX

```python
import torch
from vgl.compat import from_networkx, to_networkx
from vgl.sparse import from_torch_sparse, to_coo, to_torch_sparse

nx_graph = to_networkx(graph)
recovered = from_networkx(nx_graph)
```

- 当前路径默认产出 `networkx.MultiDiGraph`,节点与边属性按名称写入节点/边字典。
- 当前只覆盖同构图 round-trip；读回来时不需要额外的 `node_type` / `edge_type` 参数,会直接还原为默认 `("node", "to", "node")` 关系。

## torch.sparse

```python
vst = graph.adj(layout="csr")                  # vgl.sparse.SparseTensor
torch_csr = to_torch_sparse(vst)               # torch.Tensor (sparse CSR)

vst2 = from_torch_sparse(torch_csr)
coo = to_coo(vst2)
edge_index = torch.stack((coo.row, coo.col))
```

`SparseLayout` / `to_coo` / `to_csr` / `to_csc` 的细节见 [API 参考:vgl.sparse](../api/sparse.md)。

## CSV / 边列表

```python
graph = Graph.from_edge_list_csv("edges.csv")
graph = Graph.from_csv_tables("nodes.csv", "edges.csv")
graph = Graph.from_edge_list([(0, 1), (1, 2), (2, 0)])
```

- `from_csv_tables` 支持独立的节点与边 CSV,带类型列可直接识别为异构图。
- `from_edge_list` 适合 REPL 快速构造;默认把节点类型设为 `"node"`,边类型为 `("node", "to", "node")`。

## 相关 API

- [`vgl.compat`](../api/compat.md) — DGL / PyG / NetworkX 互操作
- [`vgl.sparse`](../api/sparse.md) — torch.sparse 互通
- [用户指南:Graph 对象](../guide/graph.md)
