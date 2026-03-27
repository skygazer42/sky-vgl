# Lazy Partition Store Loading Design

## Context

VGL can now build `PartitionedFeatureStore`, `PartitionedGraphStore`, and `StoreBackedSamplingCoordinator` directly from a partition directory. That fixed the architectural bootstrap issue, but the current implementation still eagerly calls `torch.load(...)` on every partition payload during construction.

That means the public “store-backed” path is functionally correct but still not suitable as a large-graph substrate. Even before the first routed fetch or partition query, the runtime pays the cost of deserializing every `part-*.pt` payload into memory. For a large partitioned graph, that defeats the point of a manifest/store-backed coordinator.

## Goals

- Make `load_partitioned_stores(...)` avoid eager payload deserialization at construction time
- Make `StoreBackedSamplingCoordinator.from_partition_dir(...)` inherit that lazy behavior automatically
- Reuse the current adapter stack and payload parsing helpers instead of redesigning the partition format
- Ensure repeated feature and graph queries against the same partition reuse one loaded payload bundle

## Options Considered

### 1. Keep eager loading

This preserves current simplicity but leaves the large-graph bottleneck untouched. It is not a meaningful next step.

### 2. Lazy wrappers around the existing local adapters

Use one shared per-partition cache object that loads a payload only on first access, builds the current local feature and graph adapters from that payload, then reuses them for future calls. This keeps behavior stable and minimizes new code.

### 3. Brand-new on-disk distributed stores

This could skip the local adapters entirely, but it is wider than necessary for this batch and duplicates already-correct logic.

## Recommended Design

Pick option 2.

### Shared partition bundle cache

Introduce an internal cache keyed by `partition_id`. Each cache entry stores a small bundle:

- `LocalFeatureStoreAdapter`
- `LocalGraphStoreAdapter`

On first access for a partition:

1. Load `part-<id>.pt`
2. Parse node tensors, owned edge tensors, and boundary-edge payloads
3. Build both local adapters from the same payload
4. Cache the bundle

That guarantees one `torch.load(...)` per partition for the lifetime of the distributed stores, even if both feature and graph queries hit the same partition.

### Lazy adapter layer

Add thin internal adapters that satisfy the existing distributed store protocols while deferring to the shared bundle cache:

- lazy feature adapter
- lazy graph adapter

These should expose metadata that can be resolved without payload loading where possible:

- `edge_types` from manifest edge metadata
- `num_nodes(node_type)` from `PartitionShard.node_ranges`

This matters because `StoreBackedSamplingCoordinator` inspects edge types during construction. If `edge_types` itself loaded payloads, construction would still be eager.

### Public behavior

`load_partitioned_stores(root)` should become lazy by default.

`StoreBackedSamplingCoordinator.from_partition_dir(root)` should use the same lazy path with no extra flags required.

## Non-Goals

- No payload schema changes
- No mmap conversion of partition payloads in this batch
- No eviction policy or bounded cache; one loaded bundle per touched partition is enough here
- No changes to `LocalGraphShard`

## Verification

- Regression proving `load_partitioned_stores(...)` performs zero `torch.load(...)` calls at construction time
- Regression proving the first query for a partition loads exactly one payload and reuses it across feature and graph queries
- Regression proving `StoreBackedSamplingCoordinator.from_partition_dir(...)` remains manifest-only until a real store access occurs
- Fresh full-suite regression before merge
