# Heterogeneous Node Block Output Design

## Context

VGL now supports:

- local and stitched homogeneous node block output
- relation-local heterogeneous `Block` construction through `to_block(..., edge_type=...)`
- relation-local heterogeneous link block output for both local and stitched sampling

The remaining gap is on node sampling. `NodeNeighborSampler(..., output_blocks=True)` still rejects every heterogeneous graph, even when the sampled supervision target has one unambiguous inbound relation that already maps cleanly onto the existing `Block` abstraction.

That leaves an avoidable large-graph training gap. A heterogeneous node workload with one message-passing relation into the supervised node type can already sample the right subgraph, but it still cannot ask the sampler for layer-wise blocks.

## Scope Choice

Three slices were considered:

1. Add full heterogeneous node block output for arbitrary multi-relation frontiers.
2. Add relation-local heterogeneous node block output only when the supervised `node_type` has exactly one inbound relation.
3. Skip node sampling and move straight to a richer multi-relation block container.

Option 2 is the right slice.

Option 1 immediately collides with the current `NodeBatch.blocks: list[Block]` contract because one node frontier layer may need multiple relation-local blocks. Option 3 is probably the long-term answer, but it is a larger public-surface change. Option 2 unlocks a real training workflow now while keeping the contract honest: one block schema per layer, one relation per block list.

## Recommended Semantics

Keep the public API unchanged:

- `NodeNeighborSampler(..., output_blocks=False)`
- `NodeNeighborSampler(..., output_blocks=True)`
- `SampleRecord.blocks`
- `NodeBatch.blocks`

New behavior:

- homogeneous behavior stays unchanged
- heterogeneous local node sampling may emit blocks when the supervised `node_type` has exactly one inbound `edge_type`
- stitched heterogeneous node sampling through `LocalSamplingCoordinator` may emit the same relation-local blocks under the same rule
- blocks stay ordered outer-to-inner

Still unsupported:

- heterogeneous node sampling where the supervised `node_type` has zero inbound relations
- heterogeneous node sampling where the supervised `node_type` has multiple inbound relations
- any richer multi-relation block list abstraction

## Construction Strategy

The block list should stay relation-local and should be built from the final sampled subgraph after feature overlays, just like the homogeneous path.

For a supervised `node_type`, choose one `node_block_edge_type` such that:

- `edge_type[2] == node_type`
- it is the only inbound relation matching that destination node type

Then:

1. retain cumulative hop snapshots by node type during heterogeneous neighbor expansion
2. take the destination-type snapshots for the supervised `node_type`
3. map those public/global ids back into sampled-graph local ids through the sampled graph's `n_id`
4. call `to_block(sampled_graph, dst_nodes=..., edge_type=node_block_edge_type)` for each layer

This works for both local and stitched heterogeneous paths because both already produce sampled graphs with stable per-type `n_id`.

## Validation

This batch should fail clearly when `output_blocks=True` is requested for:

- heterogeneous node sampling with no inbound relation into the supervised `node_type`
- heterogeneous node sampling with more than one inbound relation into the supervised `node_type`

Those are real modeling ambiguities with the current `list[Block]` contract and should stay explicit rather than guessing.

## Testing Strategy

Add focused regressions for:

- local heterogeneous node sampling with one inbound relation producing relation-local blocks
- stitched heterogeneous node sampling with one inbound relation producing the same blocks
- fixed hop count when later expansions add no new supervised-type nodes
- feature overlays remaining visible on stitched heterogeneous block graphs
- `NodeBatch.from_samples(...)` batching heterogeneous block layers across samples
- clear failure on ambiguous multi-inbound-relation hetero node block requests

Then rerun the focused node/block/coordinator suites and the full repository suite before merge.
