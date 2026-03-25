# Random Walk Ops Design

## Context

`vgl.ops` now covers several structure transforms such as self-loop rewrites, bidirection conversion, line-graph construction, metapath reachability, induced subgraphs, k-hop extraction, and compaction. One clear DGL-class gap still remains in the graph-operations layer: walk primitives.

Two especially useful missing operations are:

- homogeneous or relation-local `random_walk(...)`
- heterogeneous `metapath_random_walk(...)`

These belong in the graph-ops substrate instead of loader internals because they are reusable path-building primitives for preprocessing, augmentation, neighborhood context extraction, and future walk-based samplers.

## Scope Choice

Three plausible slices were considered:

1. Add only homogeneous `random_walk(...)`.
2. Add `random_walk(...)` plus `metapath_random_walk(...)` as one compact walk batch.
3. Jump directly to node2vec-style biased walks or restart-probability variants.

Option 2 is the right batch. It adds both the single-relation and typed-metapath path builders users expect from DGL-class graph systems while staying narrow enough to test directly in `vgl.ops`, `Graph`, and package export coverage.

## Recommended Design

Extend `vgl.ops.path` and export:

- `random_walk(graph, seeds, *, length, edge_type=None)`
- `metapath_random_walk(graph, seeds, metapath)`

Both operations should return a dense tensor of node ids with shape `(num_seeds, num_steps + 1)`. Column 0 is always the seed id. Each later column contains the visited node id for that step, or `-1` when the walk has already terminated.

`random_walk(...)` should repeatedly sample outgoing neighbors from one selected relation. When `edge_type` is omitted, it should use the graph default edge type. Repeated walks over one relation are only valid when the relation composes with itself for more than one step, so multi-step walks over bipartite relations should fail early.

`metapath_random_walk(...)` should validate the metapath with the same edge-type chaining rules already used by `metapath_reachable_graph(...)`. Each step samples from the outgoing edges of the corresponding relation in the metapath. Because the metapath defines the node-type progression, the returned trace tensor does not need an additional type payload in this batch.

## Sampling Semantics

This batch should use uniform random choice among available outgoing neighbors. If a node has no outgoing edge for the active relation, that walk step and all later steps should be `-1`.

The implementation can stay simple and deterministic under `torch.manual_seed(...)` by using PyTorch random sampling directly. Performance-oriented alias tables, restart probabilities, edge probability weights, and node2vec biasing are intentionally out of scope for now.

## API And Integration

`Graph` should grow two convenience bridges:

- `graph.random_walk(seeds, *, length, edge_type=None)`
- `graph.metapath_random_walk(seeds, metapath)`

`vgl.ops.__all__` and package-layout assertions should expose both functions as stable namespace exports.

## Testing Strategy

Add focused coverage for:

- deterministic homogeneous random walks on one-outgoing-edge paths
- termination behavior with `-1` padding after dead ends
- relation-local random walks on single-node-type multi-relation graphs
- heterogeneous metapath random walks across multiple node types
- metapath dead-end padding
- invalid repeated bipartite walks and invalid metapath chains

Then extend `tests/core/test_graph_ops_api.py` to verify the `Graph` bridges and update `tests/test_package_layout.py` so the new walk functions are part of the stable exported surface.

## Non-Goals

This batch should not include:

- node2vec or biased random walks
- restart probabilities
- probability-weighted edge sampling
- returning edge-id traces or node-type traces
- walk-based dataloaders or training loops
