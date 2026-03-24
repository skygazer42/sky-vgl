# Multi-Relation Partition Shards Design

## Context

VGL's local distributed runtime can now partition homogeneous and temporal homogeneous graphs, but it still collapses at the first multi-relation graph because `write_partitioned_graph(...)`, `LocalGraphShard`, and coordinator graph queries assume exactly one edge type. That leaves a clear DGL-class substrate gap: relation-aware graph storage exists in `Graph`, `FeatureStore`, and `InMemoryGraphStore`, but local partition artifacts cannot preserve more than one relation.

## Scope Choice

There are three plausible next steps:

1. Full heterogeneous partition manifests with per-node-type ownership.
2. Single-node-type, multi-relation partition shards.
3. Remote/distributed execution semantics on top of the current single-relation shards.

Option 2 is the right next slice. It materially improves graph storage and partition realism, keeps the current manifest shape, and avoids a larger redesign of partition ownership APIs. With one node type, the existing global node-id range model still works; only relation payloads and query surfaces need to expand.

## Recommended Design

Keep `PartitionManifest` unchanged. `write_partitioned_graph(...)` should accept graphs whose node set is still only `{"node"}` but whose edge set may contain multiple relations. For each partition, filter every relation to in-partition edges, relabel local node ids once, and serialize the resulting multi-relation graph through the existing on-disk graph payload format.

`LocalGraphShard.from_partition_dir(...)` should rebuild all serialized edge types, place per-relation edge features in the `FeatureStore`, construct an `InMemoryGraphStore` with every local relation, and preserve `time_attr` when present. `LocalGraphShard.global_edge_index(...)` and coordinator partition graph-query helpers should accept an optional `edge_type` so callers can query relation-local structure without ambiguity when a shard contains multiple relations.

## Testing Strategy

Add red-green coverage in `tests/distributed/test_partition_writer.py`, `tests/distributed/test_local_shard.py`, and `tests/distributed/test_sampling_coordinator.py` for a graph with one node type and two relations. Assert that partition payloads preserve both relations, that shards reconstruct all edge types plus relation-local edge features, and that coordinator partition queries return the requested relation in both local and global forms. Add one integration regression proving the existing loader/trainer path still works when the shard graph contains multiple relations and the dataset metadata supplies `node_type="node"`.
