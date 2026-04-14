# vgl.compat

框架互操作模块，提供 VGL 与 DGL、PyG、NetworkX 等框架的双向转换。

## 验证路径

- PyG 互操作验证：`python scripts/interop_smoke.py --backend pyg`
- DGL 互操作验证：`python scripts/interop_smoke.py --backend dgl`
- 同时验证两者：`python scripts/interop_smoke.py --backend all`
- `--backend all` 只适用于同时安装了 `sky-vgl[pyg]` 和 `sky-vgl[dgl]` 的环境
- 如果后端缺失，`scripts/interop_smoke.py` 会在报错里回显对应的 extras 安装命令

## DGL 互操作

### 图级别

```python
from vgl.graph import Graph

# 导入
graph = Graph.from_dgl(dgl_graph)

# 导出
dgl_graph = graph.to_dgl()
```

- 同构图使用 `dgl.graph(...)` 导出
- 异构/时序图使用 `dgl.heterograph(...)` 导出
- 规范边类型在异构图往返中保留
- 外部 DGL 的 `dgl.NID` / `dgl.EID` 导入为 VGL 的 `n_id` / `e_id`
- 时序图通过 `vgl_time_attr` DGL 图属性保留 `graph.schema.time_attr`

### Block 级别

```python
from vgl.graph import Block, HeteroBlock

# 单关系 Block
block = Block.from_dgl(dgl_block)
dgl_block = block.to_dgl()

# 多关系 HeteroBlock
hetero_block = HeteroBlock.from_dgl(dgl_block)
dgl_block = hetero_block.to_dgl()
```

模块级别的辅助函数：

```python
from vgl.compat import (
    block_to_dgl,
    block_from_dgl,
    hetero_block_to_dgl,
    hetero_block_from_dgl,
)
```

## PyG 互操作

```python
from vgl.graph import Graph

# 导入 PyG Data
graph = Graph.from_pyg(pyg_data)

# 导出为 PyG Data
pyg_data = graph.to_pyg()
```

## NetworkX 互操作

```python
from vgl.graph import Graph

# 导入
graph = Graph.from_networkx(nx_graph)

# 导出
nx_graph = graph.to_networkx()
```

## CSV 互操作

```python
from vgl.graph import Graph

# 从边列表 CSV 导入
graph = Graph.from_csv_edge_list("edges.csv")

# 从节点/边 CSV 对导入
graph = Graph.from_csv("nodes.csv", "edges.csv")
```

## 内存边列表

```python
from vgl.graph import Graph

# 从内存边列表导入
edges = [(0, 1), (1, 2), (2, 0)]
graph = Graph.from_edge_list(edges)
```
