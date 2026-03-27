# Multi-Relation Heterogeneous Block Operation Design

## Context

VGL already supports:

- relation-local `Block` construction through `to_block(...)`
- homogeneous node/link sampler block output
- relation-local heterogeneous block output for one supervised relation
- single-relation `Block` round-trips to and from DGL blocks

The remaining message-flow gap is multi-relation heterogeneous block construction itself.

Today, if a sampled heterogeneous frontier needs more than one relation in the same layer, VGL has no dedicated public object for that layer. The current `Block` abstraction is intentionally relation-local, so callers cannot express a DGL-style heterogeneous block layer without manually stitching several relation-local blocks together and carrying their own store-type bookkeeping.

## Scope Choice

Three options were considered:

1. Jump straight to full sampler integration for arbitrary heterogeneous block output.
2. First add a reusable multi-relation heterogeneous block abstraction plus `to_hetero_block(...)`.
3. Keep the current relation-local `Block` only and document the limitation.

Option 2 is the right next slice.

Sampler integration is the long-term user-facing payoff, but it depends on a stable public block-layer object first. Leaving the gap undocumented would preserve the limitation that currently blocks full hetero message-flow parity.

## Recommended Abstraction

Add a new `HeteroBlock` container for one heterogeneous message-flow layer.

It should carry:

- one bipartite `graph`
- `edge_types`: the original relation types included in the layer
- `src_n_id`: a dict of public source-node ids keyed by original node type
- `dst_n_id`: a dict of public destination-node ids keyed by original node type
- `src_store_types` / `dst_store_types`: mappings from original node type to the actual node store names used inside the wrapped graph

This mirrors the current `Block` contract but generalizes it to multiple relations in one layer.

## Construction Semantics

Add `to_hetero_block(graph, dst_nodes_by_type, *, edge_types=None, include_dst_in_src=True)`.

Behavior:

- `dst_nodes_by_type` is keyed by destination node type and may omit unused node types
- `edge_types=None` means include every relation in `graph.edges`
- each selected relation contributes only edges whose destination endpoint falls inside the requested destination frontier for that relation's `dst_type`
- source-node frontiers are the ordered unique predecessor union per source node type
- for same-type relations, `include_dst_in_src=True` keeps destination nodes in the corresponding source frontier just like `to_block(...)`
- public `n_id` / `e_id` metadata stays aligned to the sliced frontier and edge order

Because the result is bipartite, node types that appear on both sides of the layer need distinct internal store names. The container should hide that detail through `src_store_types` and `dst_store_types`.

## Public Surface

This batch should add:

- `vgl.graph.HeteroBlock`
- `vgl.ops.to_hetero_block(...)`
- `Graph.to_hetero_block(...)`

It should not yet:

- replace `Block`
- change sampler outputs
- add DGL import/export for multi-relation blocks
- redesign `NodeBatch.blocks` or `LinkPredictionBatch.blocks`

## Testing Strategy

Add focused regressions for:

- building a multi-relation heterogeneous block that includes both bipartite and same-type relations in one layer
- restricting `to_hetero_block(...)` to a selected subset of relations
- preserving public `n_id` / `e_id` metadata and feature slices
- `HeteroBlock.to(...)` and `pin_memory()` behavior
- `Graph.to_hetero_block(...)` bridging to the ops layer

Then rerun the focused block and graph-API suites plus the full repository suite before merge.
