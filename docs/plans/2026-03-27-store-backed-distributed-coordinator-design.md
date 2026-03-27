# Store-Backed Distributed Coordinator Design

## Context

VGL already has the core pieces of a local distributed substrate: partition writing, `PartitionManifest`, shard-local graph reload through `LocalGraphShard`, routed node and edge feature fetch, and stitched homogeneous, heterogeneous, and temporal sampling paths. The remaining bottom-layer gap versus DGL or GraphBolt-style large-graph systems is that the public coordinator still depends on fully reconstructed local shard graphs.

`LocalSamplingCoordinator` builds its routing state by scanning `LocalGraphShard` objects in-process. That makes the current runtime usable for local partition tests, but it prevents a cleaner manifest/store-backed execution model where routing and partition queries can be served from a partition book plus distributed stores. The executor still leaks this assumption too: several stitched hetero and temporal paths explicitly check `source.shards` or walk shard boundary payloads directly.

## Goals

- Add a new coordinator that works from `PartitionManifest`, `DistributedFeatureStore`, and `DistributedGraphStore` without requiring `LocalGraphShard`
- Keep the existing public sampling-coordinator surface stable for routed node ids, routed edge ids, partition queries, and feature fetch
- Make stitched heterogeneous and temporal execution paths rely on coordinator protocol methods instead of private shard access
- Preserve `LocalSamplingCoordinator` and current shard-backed flows for existing callers

## Scope Choice

The minimal useful slice is:

1. Persist edge ownership metadata per partition in the manifest.
2. Add partition-aware distributed store wrappers so one store object can serve many partitions.
3. Introduce `StoreBackedSamplingCoordinator`.
4. Refactor stitched executor paths that still hard-code `source.shards`.

The edge-ownership metadata is the key enabler. Node ownership already comes from typed node ranges, so node routing can be derived from the manifest alone. Edge ownership cannot. Partition-local edge ids are an arbitrary filtered subsequence of the original relation-local edge id space, so the coordinator needs explicit per-partition `edge_ids_by_type` metadata to map global relation edge ids to local row positions in partition-local edge tensors. Boundary edge ids are also persisted so the coordinator can expose the existing owned-vs-incident query split without reconstructing a shard graph.

## Recommended Design

### Manifest metadata

Teach `write_partitioned_graph(...)` to record two additive metadata maps on each `PartitionShard`:

- `edge_ids_by_type`
- `boundary_edge_ids_by_type`

These stay relation-scoped and preserve the original per-edge-type global id space already used by the runtime. `PartitionShard` should normalize these maps internally to tuple edge-type keys while keeping JSON serialization stable.

### Partition-aware distributed stores

Keep the store protocols partition-aware through `partition_id=...`, but add thin wrappers that can dispatch a single logical store call to a partition-specific backend:

- `PartitionedFeatureStore`
- `PartitionedGraphStore`

`DistributedGraphStore` also needs one additive query the stitched runtime already depends on conceptually:

- `boundary_edge_index(edge_type=None, partition_id=None)`

Boundary edge indexes remain in global node-id space, matching the current shard payloads and existing stitched logic.

### Store-backed coordinator

Add `StoreBackedSamplingCoordinator(manifest, feature_store, graph_store)`.

The coordinator should:

- derive node ownership and local node ids from manifest typed node ranges
- derive edge ownership and local edge ids from manifest edge metadata
- answer partition-local graph queries through `DistributedGraphStore`
- translate partition-local edge indexes into global node-id space using typed node-range offsets
- fetch routed node and edge features through `DistributedFeatureStore`

The new coordinator should also expose `partition_ids()` so stitched temporal paths can enumerate partitions through the public coordinator surface instead of touching private shard state.

### Executor de-sharding

Refactor stitched hetero and temporal helpers to use coordinator protocol methods only:

- remove `source.shards` recognition checks from `_match_*partition_shard(...)`
- rewrite temporal history builders to iterate `partition_ids()`, query incident edge ids/indexes, and fetch timestamps through `fetch_edge_features(...)`

That keeps the execution path compatible with both coordinators while finally decoupling stitched sampling from `LocalGraphShard`.

## Non-Goals

- No RPC, process-group orchestration, or remote executors in this batch
- No attempt to compact manifest edge metadata beyond straightforward lists
- No removal of `LocalSamplingCoordinator`
- No new on-disk partition-directory store implementation in this batch
- No repartitioning policy changes

## Verification

- Partition metadata regression proving edge ownership and boundary edge ids round-trip through the manifest
- Store protocol regression proving partition-aware store wrappers and boundary-edge queries dispatch correctly
- Coordinator regression proving routed node and edge fetch works without `LocalGraphShard`
- Sampling regressions proving stitched heterogeneous and temporal coordinator paths work with the new store-backed coordinator
- Fresh targeted regression plus full `python -m pytest -q` before merge
