# Distributed Edge-Feature Routing Design

## Context

VGL's local distributed runtime can now partition homogeneous, temporal, multi-relation, and true multi-node-type heterogeneous graphs. It can also route typed node ids and expose partition graph structure. The remaining visible data-plane gap is edge features: shards preserve `e_id`, `weight`, `score`, and other edge attributes, but the coordinator still cannot route global edge ids back to the correct shard and fetch those edge features through one stable surface.

## Recommended Scope

This batch should stay relation-local and additive. Global edge ids are already preserved per relation through `e_id`, so `LocalGraphShard` can build a per-edge-type global-to-local map without changing the manifest. `LocalSamplingCoordinator` can then expose `route_edge_ids(...)`, `partition_edge_ids(...)`, and `fetch_edge_features(...)`, mirroring the existing node-feature path while requiring an `edge_type` either directly or through the feature key.

## Design

`LocalGraphShard` should expose relation-scoped edge id helpers: the global edge ids for a relation, a global-to-local edge map, and a local-to-global edge conversion path. These helpers should be built from the reconstructed shard graph's `e_id` tensors, so no new on-disk payload fields are required.

`LocalSamplingCoordinator` should infer edge type from an edge feature key like `("edge", edge_type, "weight")`, route the requested global edge ids to the owning shards, fetch the local feature slices, and reassemble them in caller order. Because edge ids are relation-local, the public API must stay explicit about edge type; there is no safe untyped edge namespace once a graph carries multiple relations.

## Testing Strategy

Add red-green regression coverage in `tests/distributed/test_local_shard.py` and `tests/distributed/test_sampling_coordinator.py` for relation-scoped edge id routing and feature fetch. Add a small end-to-end regression in `tests/integration/test_foundation_partition_local.py` that writes a hetero partitioned graph, reloads shards, and fetches relation-local edge weights and scores across partitions through the coordinator.
