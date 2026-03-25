# Distributed Boundary Edge Queries Design

## Problem

VGL's local distributed runtime can already partition graphs, reload shard-local graphs, route global node and edge ids through `LocalSamplingCoordinator`, and expose partition-local structure for each shard. The current partition format still drops one critical class of structure: cross-partition boundary edges. `write_partitioned_graph(...)` keeps only edges whose endpoints are both owned by the current partition, so every shard loses visibility into outbound and inbound frontiers that leave its local node ranges.

That keeps the current runtime simple, but it leaves a real substrate gap versus DGL-class large-graph systems. Even after the sampled feature-alignment fix, the coordinator still cannot answer a basic question needed for stitched sampling: which global edges touch this partition but cross into another shard?

## Goals

- Keep the current `Graph`, shard-local graph loading, and coordinator feature-routing APIs unchanged for existing callers
- Preserve partition-local `graph_store` and `graph` behavior exactly as-is for local-only sampling paths
- Persist cross-partition boundary edges in partition payloads so each shard can expose global frontier structure
- Add additive shard/coordinator query helpers for boundary and full incident edge ids/indexes
- Support homogeneous, heterogeneous, and temporal partition payloads through one format

## Recommended Design

Keep partition-local graphs unchanged, but add one more serialized payload section: `boundary_edges`.

For each partition and each edge type:

1. Continue writing partition-local edges exactly as today when both endpoints are owned by the partition.
2. Also collect boundary edges whose source or destination is owned by the partition, but not both.
3. Store those boundary edges in global-id space, keeping `edge_index`, `e_id`, and any aligned edge features from the source graph. Keeping the full aligned edge payload avoids another format change when temporal or edge-aware stitched sampling needs timestamp or relation features later.

`LocalGraphShard` should load this payload additively beside the existing local `graph_store`. The new public helpers should stay explicit and global:

- `boundary_edge_ids(edge_type=None)`
- `boundary_edge_index(edge_type=None)`
- `incident_edge_ids(edge_type=None)` returning local owned edge ids plus boundary edge ids
- `incident_edge_index(edge_type=None)` returning global edge indexes for the same combined frontier

`LocalSamplingCoordinator` should delegate those queries partition-by-partition with matching additive methods:

- `partition_boundary_edge_ids(partition_id, edge_type=None)`
- `fetch_partition_boundary_edge_index(partition_id, edge_type=None)`
- `partition_incident_edge_ids(partition_id, edge_type=None)`
- `fetch_partition_incident_edge_index(partition_id, edge_type=None)`

This keeps the current `fetch_partition_edge_index(...)` contract stable for owned local structure while exposing the missing cross-partition frontier explicitly. Future stitched samplers can then consume `incident_*` queries without needing another partition-format change.

## Non-Goals

- No stitched multi-partition sampling in this batch
- No RPC, multiprocessing, or remote executors
- No change to routed feature fetch ownership or edge-feature routing semantics
- No mutation of shard-local `graph` objects to include halo nodes
- No attempt to deduplicate boundary edge storage across partitions

## Verification

- Partition-writer regression proving payloads now preserve boundary edges in global-id space
- Shard regression proving homogeneous and typed shards can expose boundary and incident edge queries without disturbing local edge APIs
- Coordinator regression proving partition-scoped boundary/incident queries round-trip through the public runtime API
- Fresh full repository regression before merge
