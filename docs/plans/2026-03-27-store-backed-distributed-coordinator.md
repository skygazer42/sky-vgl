# Store-Backed Distributed Coordinator Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a manifest/store-backed sampling coordinator that can drive routed feature fetch and stitched distributed sampling without requiring `LocalGraphShard` objects.

**Architecture:** Persist additive edge-routing metadata in `PartitionManifest`, add partition-aware distributed store wrappers, implement `StoreBackedSamplingCoordinator`, and refactor stitched executor paths to depend only on coordinator protocol methods such as routed feature fetch, partition edge queries, and `partition_ids()`.

**Tech Stack:** Python 3.10+, PyTorch, pytest

---

### Task 1: Add failing regressions for manifest metadata and store-backed coordinator behavior

**Files:**
- Modify: `tests/distributed/test_partition_metadata.py`
- Modify: `tests/distributed/test_store_protocol.py`
- Modify: `tests/distributed/test_sampling_coordinator.py`
- Modify: `tests/data/test_temporal_neighbor_sampler.py`

**Step 1: Write the failing test**

Add regressions proving:

- `PartitionShard` preserves `edge_ids_by_type` and `boundary_edge_ids_by_type` through manifest round-trip
- partition-aware distributed stores dispatch partition-specific feature, local edge, and boundary-edge queries
- `StoreBackedSamplingCoordinator` can route node ids, route edge ids, fetch routed edge features, and answer partition boundary/incident queries without `LocalGraphShard`
- stitched heterogeneous temporal sampling works when `feature_store=` is the new store-backed coordinator rather than `LocalSamplingCoordinator`

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/distributed/test_partition_metadata.py tests/distributed/test_store_protocol.py tests/distributed/test_sampling_coordinator.py tests/data/test_temporal_neighbor_sampler.py -k "store_backed or edge_ids_by_type or boundary_edge_index"`
Expected: FAIL because manifest edge metadata helpers, partition-aware store wrappers, the new coordinator, and shard-free temporal stitched execution do not exist yet.

**Step 3: Write minimal implementation**

No implementation yet in this task.

**Step 4: Run test to verify it still fails for the intended reason**

Run: `python -m pytest -q tests/distributed/test_partition_metadata.py tests/distributed/test_store_protocol.py tests/distributed/test_sampling_coordinator.py tests/data/test_temporal_neighbor_sampler.py -k "store_backed or edge_ids_by_type or boundary_edge_index"`
Expected: FAIL on missing metadata normalization, missing store classes, missing coordinator, or shard-only executor assumptions.

**Step 5: Commit**

Do not commit yet.

### Task 2: Implement manifest edge metadata and partition-aware distributed store wrappers

**Files:**
- Modify: `vgl/distributed/partition.py`
- Modify: `vgl/distributed/writer.py`
- Modify: `vgl/distributed/store.py`
- Modify: `vgl/distributed/__init__.py`
- Modify: `tests/distributed/test_partition_metadata.py`
- Modify: `tests/distributed/test_store_protocol.py`

**Step 1: Write the failing test**

Use the regressions from Task 1.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/distributed/test_partition_metadata.py tests/distributed/test_store_protocol.py -k "store_backed or edge_ids_by_type or boundary_edge_index"`
Expected: FAIL

**Step 3: Write minimal implementation**

Implement:

- manifest metadata normalization and JSON round-trip for `edge_ids_by_type` and `boundary_edge_ids_by_type`
- partition writer persistence of those metadata maps
- additive `boundary_edge_index(...)` support in `DistributedGraphStore`
- `LocalGraphStoreAdapter` support for optional boundary-edge indexes
- `PartitionedFeatureStore` and `PartitionedGraphStore`
- public exports for the new store types

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/distributed/test_partition_metadata.py tests/distributed/test_store_protocol.py -k "store_backed or edge_ids_by_type or boundary_edge_index"`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 3: Implement the store-backed sampling coordinator

**Files:**
- Modify: `vgl/distributed/coordinator.py`
- Modify: `vgl/distributed/__init__.py`
- Modify: `tests/distributed/test_sampling_coordinator.py`

**Step 1: Write the failing test**

Use the regressions from Task 1.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/distributed/test_sampling_coordinator.py -k "store_backed"`
Expected: FAIL

**Step 3: Write minimal implementation**

Implement `StoreBackedSamplingCoordinator` with:

- `partition_ids()`
- manifest-derived node routing and local node ids
- manifest-derived edge routing and local edge ids
- partition edge/boundary/incident id queries from manifest metadata
- partition edge/boundary/incident index queries from `DistributedGraphStore`
- routed node and edge feature fetch through `DistributedFeatureStore`

Keep `LocalSamplingCoordinator` intact while adding `partition_ids()` for compatibility with executor refactors.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/distributed/test_sampling_coordinator.py -k "store_backed"`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 4: Remove executor shard-only assumptions and verify stitched sampling

**Files:**
- Modify: `vgl/dataloading/executor.py`
- Modify: `tests/data/test_temporal_neighbor_sampler.py`
- Modify: `tests/data/test_node_neighbor_sampler.py`
- Modify: `tests/data/test_link_neighbor_sampler.py`

**Step 1: Write the failing test**

Use the regressions from Task 1 and add any extra stitched coordinator regressions needed for heterogeneous node or link sampling if temporal coverage alone does not touch all shard-only checks.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/data/test_node_neighbor_sampler.py tests/data/test_link_neighbor_sampler.py tests/data/test_temporal_neighbor_sampler.py -k "store_backed or stitched"`
Expected: FAIL because stitched recognition and temporal history building still depend on `source.shards`.

**Step 3: Write minimal implementation**

Refactor executor helpers to use coordinator protocol methods:

- remove `source.shards` gating from stitched candidate detection
- use `partition_ids()` plus public partition incident queries in temporal history builders
- preserve current behavior for the local shard-backed coordinator

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/data/test_node_neighbor_sampler.py tests/data/test_link_neighbor_sampler.py tests/data/test_temporal_neighbor_sampler.py -k "store_backed or stitched"`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 5: Run focused and full verification

**Files:**
- Modify: any touched files from prior tasks as needed

**Step 1: Write the failing test**

No new test. Re-read the plan and touched code paths to confirm the intended scope is covered.

**Step 2: Run focused verification**

Run: `python -m pytest -q tests/distributed/test_partition_metadata.py tests/distributed/test_store_protocol.py tests/distributed/test_sampling_coordinator.py tests/data/test_node_neighbor_sampler.py tests/data/test_link_neighbor_sampler.py tests/data/test_temporal_neighbor_sampler.py tests/integration/test_foundation_partition_local.py`
Expected: PASS

**Step 3: Run full verification**

Run: `python -m pytest -q`
Expected: PASS across the repository.

**Step 4: Prepare integration**

Merge the worktree branch back to `main`, push `origin/main`, and remove the temporary worktree and feature branch so only `main` remains locally and remotely.

**Step 5: Commit**

Run: `git add vgl tests docs && git commit -m "feat: add store-backed distributed coordinator"`
