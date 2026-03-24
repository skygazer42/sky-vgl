# Temporal Partition Shards Design

## Context

VGL now has local partition metadata, partition file writing, shard-local graph loading, and coordinator-based partition queries, but the partition runtime still rejects any graph with `schema.time_attr`. That leaves a visible substrate gap for large temporal-graph workflows: temporal graphs can be serialized on disk and sampled in-memory, yet they cannot move through the local partition path that the distributed foundation exposes.

## Options Considered

1. Extend the current homogeneous partition path to support temporal homogeneous graphs.
2. Redesign partition metadata and payloads to support full heterogeneous graphs.
3. Introduce a separate temporal-only shard format.

Option 1 is the right next slice. It keeps the current manifest shape, preserves existing homogeneous behavior, and only requires the partition writer and shard loader to retain `time_attr` plus the edge feature named by that attribute. Option 2 is valuable, but it needs per-type node ownership and a broader manifest redesign. Option 3 would duplicate the current runtime surface without solving the underlying limitation.

## Recommended Design

Keep partition manifests unchanged: each shard still owns one contiguous global node range and one serialized graph payload. Extend `_partition_subgraph(...)` so it returns `Graph.temporal(...)` when the source graph is temporal, preserving node features, edge features, and the temporal edge attribute on the filtered in-partition edges. `write_partitioned_graph(...)` should stop rejecting temporal homogeneous graphs while still rejecting heterogeneous graphs.

On the read path, `LocalGraphShard.from_partition_dir(...)` should accept temporal homogeneous payloads, rebuild the `FeatureStore` and `InMemoryGraphStore` exactly as today, and construct a `GraphSchema` with `time_attr` set. The shard object and its global/local id mapping APIs remain unchanged. Existing coordinator and loader integration can then operate on temporal shards because the public graph contract stays the same.

## Testing Strategy

Add red-green regression coverage in `tests/distributed/test_partition_writer.py` and `tests/distributed/test_local_shard.py` for temporal graphs. The writer test should assert that each partition payload round-trips `schema.time_attr`, preserves filtered timestamps, and keeps global `node_ids`. The shard test should assert that a temporal shard reloads through `Graph.from_storage(...)`, exposes the time attribute on the reconstructed graph, and maps local/global node ids exactly as before.

Add one integration regression in `tests/integration/test_foundation_partition_local.py` that routes temporal partition shards through the existing loader/trainer path. This confirms the distributed substrate remains usable by the current public training API after the temporal extension.
