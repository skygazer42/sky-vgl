# Distributed Boundary Edge Queries Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Preserve cross-partition boundary edges in partition payloads and expose additive shard/coordinator query helpers so the local distributed runtime can see partition frontiers needed for future stitched sampling.

**Architecture:** Keep shard-local graphs and local edge queries unchanged. Extend `write_partitioned_graph(...)` with a serialized `boundary_edges` payload in global-id space, load it additively in `LocalGraphShard`, and surface explicit boundary/incident edge query helpers through `LocalSamplingCoordinator`.

**Tech Stack:** Python 3.11, PyTorch, pytest

---

### Task 1: Add failing distributed regressions for boundary-edge payloads and queries

**Files:**
- Modify: `tests/distributed/test_partition_writer.py`
- Modify: `tests/distributed/test_local_shard.py`
- Modify: `tests/distributed/test_sampling_coordinator.py`

**Step 1: Write the failing test**

Add regressions proving:

- `write_partitioned_graph(...)` preserves cross-partition boundary edges in the partition payload
- `LocalGraphShard` exposes `boundary_edge_ids(...)`, `boundary_edge_index(...)`, `incident_edge_ids(...)`, and `incident_edge_index(...)`
- `LocalSamplingCoordinator` exposes matching partition-scoped boundary and incident edge queries for homogeneous and typed relations

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/distributed/test_partition_writer.py tests/distributed/test_local_shard.py tests/distributed/test_sampling_coordinator.py -k boundary`
Expected: FAIL because boundary edges are still dropped at write time and the shard/coordinator query helpers do not exist.

**Step 3: Write minimal implementation**

No implementation yet in this task.

**Step 4: Run test to verify it still fails for the intended reason**

Run: `python -m pytest -q tests/distributed/test_partition_writer.py tests/distributed/test_local_shard.py tests/distributed/test_sampling_coordinator.py -k boundary`
Expected: FAIL on missing payload fields or missing query methods.

**Step 5: Commit**

Do not commit yet.

### Task 2: Implement boundary-edge persistence and query helpers

**Files:**
- Modify: `vgl/distributed/writer.py`
- Modify: `vgl/distributed/shard.py`
- Modify: `vgl/distributed/coordinator.py`
- Modify: `tests/distributed/test_partition_writer.py`
- Modify: `tests/distributed/test_local_shard.py`
- Modify: `tests/distributed/test_sampling_coordinator.py`

**Step 1: Write the failing test**

Use the regressions from Task 1.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/distributed/test_partition_writer.py tests/distributed/test_local_shard.py tests/distributed/test_sampling_coordinator.py -k boundary`
Expected: FAIL

**Step 3: Write minimal implementation**

Teach the partition writer to serialize per-type `boundary_edges` in global-id space, keeping `edge_index`, `e_id`, and aligned edge features. Teach `LocalGraphShard` to load that payload and expose explicit boundary and incident query helpers. Teach `LocalSamplingCoordinator` to delegate the same partition-scoped queries additively without changing current local-only methods.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/distributed/test_partition_writer.py tests/distributed/test_local_shard.py tests/distributed/test_sampling_coordinator.py -k boundary`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 3: Refresh docs and run branch verification

**Files:**
- Modify: `README.md`
- Modify: `docs/core-concepts.md`
- Modify: `docs/quickstart.md`
- Modify: any touched modules as needed

**Step 1: Write the failing test**

No code test. Review distributed docs for places that still imply partition queries only expose owned local edges.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/distributed/test_partition_writer.py tests/distributed/test_local_shard.py tests/distributed/test_sampling_coordinator.py`
Expected: PASS; documentation gap is manual.

**Step 3: Write minimal implementation**

Document that partition payloads now preserve cross-partition boundary edges and that the coordinator can expose owned-local versus full-incident frontier structure through explicit query helpers.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q`
Expected: PASS across the repository.

**Step 5: Commit**

Run: `git add vgl tests docs README.md && git commit -m "feat: expose distributed boundary edge queries"`
