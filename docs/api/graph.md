# vgl.graph

图核心模块，包含 Graph 对象及其相关类型。

## Graph

::: vgl.graph.Graph
    options:
      show_root_heading: true
      show_source: false
      members_order: source

## GraphBatch

::: vgl.graph.GraphBatch
    options:
      show_root_heading: true
      show_source: false

## GraphView

::: vgl.graph.GraphView
    options:
      show_root_heading: true
      show_source: false

## GraphSchema

::: vgl.graph.GraphSchema
    options:
      show_root_heading: true
      show_source: false

## Block

::: vgl.graph.Block
    options:
      show_root_heading: true
      show_source: false

## HeteroBlock

`HeteroBlock` 是多关系版本的消息流块(Block),为每条规范 `(src_type, rel, dst_type)` 维护一份独立的源/目标 ID 映射和稀疏邻接,常用于异构邻居采样器把多关系前沿打包成单次模型调用。与 `Block` 的关键区别:

- 对外以字典形式暴露 `src_nodes[ntype]` / `dst_nodes[ntype]` / `edges[rel]`,保持节点类型不混淆。
- 与 `vgl.ops.to_hetero_block` 以及 `vgl.compat.hetero_block_{to,from}_dgl` 互通。

::: vgl.graph.HeteroBlock
    options:
      show_root_heading: true
      show_source: false

## NodeBatch

::: vgl.graph.NodeBatch
    options:
      show_root_heading: true
      show_source: false

## LinkPredictionBatch

::: vgl.graph.LinkPredictionBatch
    options:
      show_root_heading: true
      show_source: false

## TemporalEventBatch

::: vgl.graph.TemporalEventBatch
    options:
      show_root_heading: true
      show_source: false
