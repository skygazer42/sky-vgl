# vgl.ops

图结构操作模块，提供子图提取、度数查询、邻接矩阵、随机游走等底层操作。

## 图查询

| 函数 | 说明 |
|------|------|
| `num_nodes(graph)` | 节点数 |
| `num_edges(graph)` | 边数 |
| `number_of_nodes(graph)` | 别名 |
| `number_of_edges(graph)` | 别名 |
| `all_edges(graph, order="eid")` | 全边枚举 |
| `formats(graph)` | 稀疏格式状态 |
| `create_formats_(graph)` | 急切创建稀疏格式 |

## 边查询

| 函数 | 说明 |
|------|------|
| `find_edges(graph, eids)` | 按边 ID 查找端点 |
| `edge_ids(graph, src, dst)` | 按端点查找边 ID |
| `has_edges_between(graph, src, dst)` | 检查边是否存在 |
| `in_edges(graph, nodes)` | 入边查询 |
| `out_edges(graph, nodes)` | 出边查询 |
| `predecessors(graph, node)` | 前驱节点 |
| `successors(graph, node)` | 后继节点 |
| `in_degrees(graph, nodes)` | 入度 |
| `out_degrees(graph, nodes)` | 出度 |

## 邻接和稀疏视图

| 函数 | 说明 |
|------|------|
| `adj(graph, ...)` | 加权邻接稀疏视图 |
| `laplacian(graph, ...)` | 拉普拉斯稀疏视图 |
| `adj_external(graph, ...)` | 外部格式导出 (torch / SciPy) |
| `adj_tensors(graph, "coo")` | 原始 COO/CSR/CSC 张量 |
| `inc(graph, typestr=...)` | 关联矩阵 |

## 结构变换

| 函数 | 说明 |
|------|------|
| `to_simple(graph, ...)` | 去重平行边 |
| `reverse(graph)` | 反转边方向 |
| `in_subgraph(graph, nodes)` | 入边前沿子图 |
| `out_subgraph(graph, nodes)` | 出边前沿子图 |
| `to_block(graph, ...)` | 关系局部消息流 Block |
| `to_hetero_block(graph, ...)` | 多关系 HeteroBlock |
| `line_graph(graph)` | 线图 |

## 子图和 K 跳

| 函数 | 说明 |
|------|------|
| `node_subgraph(graph, nodes)` | 节点诱导子图 |
| `edge_subgraph(graph, eids)` | 边诱导子图 |
| `khop_nodes(graph, nodes, k)` | K 跳可达节点 |
| `khop_subgraph(graph, nodes, k)` | K 跳诱导子图 |
| `compact_nodes(graph, nodes)` | 紧凑化节点 ID |

## 随机游走

| 函数 | 说明 |
|------|------|
| `random_walk(graph, ...)` | 关系局部随机游走 |
| `metapath_random_walk(graph, ...)` | 元路径随机游走 |
| `metapath_reachable_graph(graph, ...)` | 元路径可达图 |

所有函数同时支持同构图和异构图（通过 `edge_type` 参数选择关系）。

## to_simple 语义

`to_simple` 对每个关系去重平行边并按字典序归并(`(src, dst)` → 聚合到唯一边),返回的新图与原图共享节点视图,但边 payload/特征会按以下规则合并:

- 整型/浮点边特征默认走 `sum` 归约;可传 `reduce="mean"` / `"max"` / `"min"` 调整。
- 对 `eid`/`n_id` 等保留字段以"出现的第一条"为准,不归约。
- 可传 `return_edge_mapping=True` 获得 `(new_graph, edge_mapping)`,其中 `edge_mapping` 是长度为原边数的 `LongTensor`,将旧 eid 映射到去重后的 eid。

::: vgl.ops.to_simple
    options:
      show_root_heading: true
      show_source: false
