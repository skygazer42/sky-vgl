# To-Block And Block Container Design

## Context

VGL already covers full-graph training, sampled subgraph training, relation-local graph transforms, and a widening sparse/data/distributed substrate. One important DGL-class foundation gap still remains in the mini-batch graph surface: there is no explicit message-flow block abstraction and no `to_block(...)` transform that rewrites one relation into a source/destination bipartite frontier.

That gap matters for two reasons:

- it is a familiar migration surface for DGL users
- it gives VGL a compact foundation for future block-based samplers or loader outputs without forcing those larger changes into this batch

## Scope Choice

Three plausible slices were considered:

1. Add only a `to_block(...)` graph transform that returns a raw `Graph`.
2. Add a `to_block(...)` transform plus a lightweight `Block` container that carries the remapped graph and source/destination id metadata.
3. Jump straight to block-native sampler outputs and block mini-batch training.

Option 2 is the right slice. Returning only a raw `Graph` would leak awkward internal node-type naming into the public surface and make source/destination metadata easy to misuse. Jumping directly to block-native samplers is a much larger runtime change. A small `Block` container keeps the first batch focused and useful.

## Recommended API

Add:

- `vgl.ops.to_block(graph, dst_nodes, *, edge_type=None, include_dst_in_src=True)`
- `Graph.to_block(dst_nodes, *, edge_type=None, include_dst_in_src=True)`
- `vgl.graph.Block`

`to_block(...)` operates on one selected relation. It gathers every edge whose destination is in `dst_nodes`, compacts the participating source and destination node ids, and returns a `Block`.

The `Block` container should expose:

- `graph`: the compacted relation-local bipartite graph
- `edge_type`: the original selected relation
- `src_type` / `dst_type`: the original endpoint node types
- `src_n_id` / `dst_n_id`: global node ids from the source graph
- `srcdata` / `dstdata` / `edata` convenience views
- `edge_index`, `to(...)`, and `pin_memory()`

## Internal Representation

The wrapped `graph` should stay a regular `Graph` so all existing store and transfer machinery keeps working.

For relations whose endpoint node types differ, the wrapped graph can keep the original node type names. For relations whose source and destination node types are the same, the wrapped graph must avoid the collision by using deterministic internal names such as `<type>__src` and `<type>__dst`. The `Block` wrapper hides that internal detail from callers.

The block graph should always carry:

- `n_id` on both source and destination node stores
- `e_id` on the selected edge store

Node- and edge-aligned tensors from the source graph should be sliced into the block in stable source-graph order, matching current `node_subgraph(...)` and `edge_subgraph(...)` semantics.

## Semantics

`dst_nodes` are ordered unique destination ids.

Selected edges are all edges from the chosen relation whose destination appears in `dst_nodes`.

For same-type relations:

- when `include_dst_in_src=True`, `src_n_id` should begin with `dst_n_id` and then append additional predecessor nodes in first-seen order
- when `include_dst_in_src=False`, `src_n_id` should contain only predecessor nodes in first-seen order

For bipartite relations with different source and destination node types, `include_dst_in_src` has no effect because the two endpoint spaces are already distinct.

Empty inputs should stay valid. A block with no incoming edges should still preserve destination nodes and produce empty `e_id` / `edge_index`.

## Validation

`to_block(...)` should fail early when:

- `edge_type` is omitted on a graph whose selected relation is ambiguous
- `dst_nodes` is not rank-1 after tensor coercion
- any destination id falls outside the destination node range

This batch should stay relation-local. It should not build multi-relation blocks, stacked block lists, or block-native sampler outputs.

## Testing Strategy

Add focused tests for:

- homogeneous `to_block(...)` feature slicing and `n_id` / `e_id` metadata
- `include_dst_in_src=False`
- hetero relation-local block construction
- empty frontier behavior
- `Block.to(...)` and `Block.pin_memory()`
- `Graph.to_block(...)` and package exports

Then run the full suite before merge.
