# DGL Graph Adapter Boundary And Count Preservation Design

## Context

VGL now has dedicated `Block` <-> DGL block helpers and keeps `Graph.from_dgl(...)` / `Graph.to_dgl(...)` as graph-only entry points. There are still two practical compatibility edges left in the graph adapter itself.

First, `Graph.from_dgl(...)` currently accepts a DGL block object and silently imports it as a graph because the adapter only distinguishes homogeneous vs heterogeneous DGL objects. That blurs the public API boundary established by the new block helpers.

Second, external DGL graphs can carry node cardinality even when node features are absent or when some nodes are isolated. VGL's internal `Graph` abstraction infers node counts from node tensors or structure, so importing a DGL graph with no node features can drop isolated-node counts unless the adapter materializes count-preserving metadata.

## Scope Choice

This batch should harden the graph adapter contract without widening any public graph or block abstractions.

1. Reject DGL blocks at `Graph.from_dgl(...)` / `vgl.compat.from_dgl(...)` with a clear error that points users to `Block.from_dgl(...)` or `block_from_dgl(...)`.
2. Preserve node counts from external DGL graphs by materializing `n_id` metadata only when the imported node data would otherwise be unable to preserve cardinality.
3. Normalize external DGL `dgl.NID` / `dgl.EID` metadata onto VGL's `n_id` / `e_id` keys during graph import.

This keeps the graph adapter graph-only, preserves isolated nodes, and makes imported DGL subgraph metadata line up with the rest of VGL.

## Recommended Behavior

### Graph/block boundary

- `Graph.from_dgl(dgl_block)` should fail with a `ValueError`
- the error should mention that DGL blocks are graph-incompatible for this entry point and direct users to `Block.from_dgl(...)`
- `Block.from_dgl(...)` remains the dedicated import path for message-flow blocks

### External DGL node count preservation

For each imported node type:

- keep existing node features unchanged when they already preserve count
- if `n_id` exists, keep it
- else if `dgl.NID` exists, rename it to `n_id`
- else if there is no rank-1-or-higher node tensor to preserve the declared node count, inject `n_id = arange(num_nodes)`

This is deliberately additive metadata. It only appears when needed to preserve node identity or count.

### External DGL edge metadata normalization

For each imported edge type:

- keep existing `e_id` if present
- else if `dgl.EID` exists, rename it to `e_id`
- do not inject `e_id` when DGL did not provide one, because edge cardinality is already preserved by `edge_index`

## Testing Strategy

Add focused DGL adapter regressions for:

- homogeneous external DGL graphs with isolated nodes and no node features keeping `num_nodes()` on round-trip
- heterogeneous external DGL graphs with declared node counts and no node features keeping per-type `num_nodes(...)` on round-trip
- external DGL `dgl.NID` / `dgl.EID` metadata importing as `n_id` / `e_id`
- `Graph.from_dgl(...)` rejecting DGL blocks with a clear message

Then rerun the DGL adapter suite and the full repository suite before merge.
