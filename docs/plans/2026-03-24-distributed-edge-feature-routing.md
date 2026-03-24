# Distributed Edge-Feature Routing Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend the local distributed runtime so shards and coordinators can route relation-scoped global edge ids and fetch edge features across partitioned graphs.

**Architecture:** Reuse the existing per-relation `e_id` edge feature that partition writing already preserves. Build edge-id lookup maps inside `LocalGraphShard`, then mirror the current node-feature routing flow inside `LocalSamplingCoordinator` for `("edge", edge_type, feature_name)` keys.

**Tech Stack:** Python 3.11, PyTorch, pytest

---

### Task 1: Add shard edge-id regression coverage

**Files:**
- Modify: `tests/distributed/test_local_shard.py`
- Modify: `vgl/distributed/shard.py`

**Step 1: Write the failing test**

Add tests proving a shard can expose partition-global edge ids for a relation and map global edge ids back to local relation-local positions.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/distributed/test_local_shard.py -k edge_id`
Expected: FAIL because shard edge-id helpers do not exist.

**Step 3: Write minimal implementation**

Add relation-scoped `edge_ids(...)`, `global_to_local_edge(...)`, and `local_to_global_edge(...)` helpers derived from the shard graph's `e_id` tensors.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/distributed/test_local_shard.py`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 2: Add coordinator edge routing and edge-feature fetch

**Files:**
- Modify: `tests/distributed/test_sampling_coordinator.py`
- Modify: `vgl/distributed/coordinator.py`

**Step 1: Write the failing test**

Add tests proving the coordinator can route global edge ids for a selected relation, expose partition edge ids, and fetch relation-scoped edge features through an `("edge", edge_type, feature_name)` key.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/distributed/test_sampling_coordinator.py -k edge`
Expected: FAIL because coordinator edge routing and fetch APIs do not exist.

**Step 3: Write minimal implementation**

Implement `route_edge_ids(...)`, `partition_edge_ids(...)`, and `fetch_edge_features(...)`, inferring the relation from the feature key and reassembling outputs in caller order.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/distributed/test_sampling_coordinator.py`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 3: Add end-to-end edge-feature routing coverage

**Files:**
- Modify: `tests/integration/test_foundation_partition_local.py`

**Step 1: Write the failing test**

Add an integration regression that partitions a hetero graph, reloads shards, and fetches relation-scoped edge weights or scores across partitions via the coordinator.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/integration/test_foundation_partition_local.py -k edge_feature`
Expected: FAIL because the coordinator does not expose edge-feature routing yet.

**Step 3: Write minimal implementation**

Use the production changes from Tasks 1-2; only patch integration helpers if a real gap appears.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/integration/test_foundation_partition_local.py`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 4: Refresh docs and run full regression

**Files:**
- Modify: `README.md`
- Modify: `docs/core-concepts.md`
- Modify: `docs/quickstart.md`
- Modify: any touched modules as needed

**Step 1: Write the failing test**

No code test. Review docs for places that still describe the coordinator as node-feature-only on the data plane.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/distributed/test_local_shard.py tests/distributed/test_sampling_coordinator.py tests/integration/test_foundation_partition_local.py`
Expected: PASS; documentation gap is manual.

**Step 3: Write minimal implementation**

Document the new relation-scoped edge-id and edge-feature routing helpers alongside the existing node-feature and partition-graph query surfaces.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q`
Expected: PASS across the repository.

**Step 5: Commit**

Run: `git add vgl tests docs README.md && git commit -m "feat: add distributed edge feature routing"`
