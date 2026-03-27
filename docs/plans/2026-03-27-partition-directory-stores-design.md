# Partition-Directory Store Assembly Design

## Context

VGL now has a real `StoreBackedSamplingCoordinator` plus partition-aware distributed store wrappers. That closed the routing and stitched-executor gap, but one important bootstrap gap remains: every current store-backed test still assembles those stores from `LocalGraphShard`.

That means the coordinator implementation no longer depends on `LocalGraphShard`, but the public way to instantiate it still does. This is the remaining mismatch between the intended architecture and the actual entrypoint. The partition files already contain everything needed to build the stores directly:

- manifest metadata with typed node ranges and edge ownership
- serialized partition-local node and edge feature tensors
- partition-local edge indexes
- boundary-edge indexes and features in global-id space

No shard graph reconstruction is required to surface those pieces.

## Goals

- Add a direct on-disk entrypoint for partition-aware distributed stores
- Allow `StoreBackedSamplingCoordinator` to be created from a partition directory without `LocalGraphShard`
- Reuse the current `PartitionedFeatureStore`, `PartitionedGraphStore`, and `Local*Adapter` abstractions instead of inventing a second runtime stack
- Keep `LocalGraphShard` unchanged for callers that still need full partition-local `Graph` objects

## Recommended Design

Add lightweight payload readers in `vgl.distributed.store` that build partition-aware stores directly from `manifest.json` plus `part-*.pt` payloads.

### Store assembly

For each partition payload:

1. Read `payload["graph"]` directly as a serialized graph dictionary.
2. Build a `FeatureStore` from node tensors and owned edge tensors without deserializing a `Graph`.
3. Build an `InMemoryGraphStore` from the partition-local edge indexes and per-type local node counts.
4. Pass `payload["boundary_edges"]` into the existing boundary-aware `LocalFeatureStoreAdapter` and `LocalGraphStoreAdapter`.

Then aggregate these partition-local adapters into:

- `PartitionedFeatureStore`
- `PartitionedGraphStore`

The natural public helper is a single loader:

- `load_partitioned_stores(root) -> (manifest, feature_store, graph_store)`

### Coordinator entrypoint

Add:

- `StoreBackedSamplingCoordinator.from_partition_dir(root)`

This classmethod should call `load_partitioned_stores(root)` and then construct the coordinator. That gives callers one obvious, correct bootstrap path and keeps the constructor usable for advanced custom stores.

## Scope Boundaries

This batch should not:

- remove `LocalGraphShard`
- replace partition-local graph loading for samplers or trainers
- introduce mmap or lazy tensor paging for partition payloads
- redesign partition payload schema

The win here is architectural honesty and a real on-disk entrypoint, not a new storage backend.

## Verification

- Store protocol regressions proving direct partition-directory assembly exposes the same local and boundary queries as the current adapter stack
- Coordinator regressions proving `StoreBackedSamplingCoordinator.from_partition_dir(...)` routes and fetches features without `LocalGraphShard`
- Sampling regressions updated to use the direct entrypoint for the coordinator while preserving current stitched behavior
- Fresh full-suite regression before merge
