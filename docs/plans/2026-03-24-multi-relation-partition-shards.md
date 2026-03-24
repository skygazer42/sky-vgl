# Multi-Relation Partition Shards Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend the local partition writer, shard loader, and coordinator graph queries to preserve single-node-type multi-relation graphs.

**Architecture:** Keep the current partition manifest shape and contiguous node-range ownership model. Partition payloads will continue to store one serialized graph plus `node_ids`, but the graph payload and shard reconstruction path will now preserve every relation in graphs whose only node type is `node`, with optional `edge_type` selection added to shard/coordinator structure queries where ambiguity exists.

**Tech Stack:** Python 3.11, PyTorch, pytest

---

### Task 1: Add multi-relation partition writer regression coverage

**Files:**
- Modify: `tests/distributed/test_partition_writer.py`

**Step 1: Write the failing test**

Add a test proving `write_partitioned_graph(...)` accepts a graph with one node type and multiple edge types, and that each partition payload preserves every in-partition relation with filtered edge features.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/distributed/test_partition_writer.py -k relation`
Expected: FAIL because the writer still assumes a single edge type.

**Step 3: Write minimal implementation**

No implementation yet in this task.

**Step 4: Run test to verify it still fails for the intended reason**

Run: `python -m pytest -q tests/distributed/test_partition_writer.py -k relation`
Expected: FAIL on the single-edge-type assumption in the writer.

**Step 5: Commit**

Do not commit yet.

### Task 2: Implement multi-relation partition writing

**Files:**
- Modify: `vgl/distributed/writer.py`
- Modify: `tests/distributed/test_partition_writer.py`

**Step 1: Write the failing test**

Use the test from Task 1.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/distributed/test_partition_writer.py -k relation`
Expected: FAIL

**Step 3: Write minimal implementation**

Generalize the partition writer to filter and serialize every relation in single-node-type graphs, preserving temporal metadata when present.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/distributed/test_partition_writer.py -k relation`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 3: Add multi-relation shard and coordinator regression coverage

**Files:**
- Modify: `tests/distributed/test_local_shard.py`
- Modify: `tests/distributed/test_sampling_coordinator.py`

**Step 1: Write the failing test**

Add tests proving `LocalGraphShard` reconstructs all relations in a partition, exposes relation-local global edge indices, and that `LocalSamplingCoordinator` can return partition-local/global edge indices and adjacency for a selected `edge_type`.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/distributed/test_local_shard.py -k relation tests/distributed/test_sampling_coordinator.py -k relation`
Expected: FAIL because shard reconstruction and coordinator partition queries still assume one edge type.

**Step 3: Write minimal implementation**

No implementation yet in this task.

**Step 4: Run test to verify it still fails for the intended reason**

Run: `python -m pytest -q tests/distributed/test_local_shard.py -k relation tests/distributed/test_sampling_coordinator.py -k relation`
Expected: FAIL on single-edge-type assumptions and missing `edge_type` plumbing.

**Step 5: Commit**

Do not commit yet.

### Task 4: Implement multi-relation shard reconstruction and partition queries

**Files:**
- Modify: `vgl/distributed/shard.py`
- Modify: `vgl/distributed/coordinator.py`
- Modify: `tests/distributed/test_local_shard.py`
- Modify: `tests/distributed/test_sampling_coordinator.py`
- Modify: `tests/integration/test_foundation_partition_local.py`

**Step 1: Write the failing test**

Use the tests from Task 3 and add one integration assertion that shard graphs with multiple relations still flow through the public loader/trainer path when dataset metadata specifies `node_type="node"`.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/distributed/test_local_shard.py tests/distributed/test_sampling_coordinator.py tests/integration/test_foundation_partition_local.py -k relation`
Expected: FAIL

**Step 3: Write minimal implementation**

Rebuild all serialized relations in `LocalGraphShard`, preserve per-relation feature-store keys, add optional `edge_type` parameters to shard/coordinator structure queries, and update the partition-local integration path to exercise a multi-relation shard graph.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/distributed/test_partition_writer.py tests/distributed/test_local_shard.py tests/distributed/test_sampling_coordinator.py tests/integration/test_foundation_partition_local.py`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 5: Refresh docs and run full regression

**Files:**
- Modify: `README.md`
- Modify: `docs/core-concepts.md`
- Modify: `docs/quickstart.md`
- Modify: any touched modules as needed

**Step 1: Write the failing test**

No code test. Review docs for places that still describe local partition/shard support as single-relation only.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/distributed/test_partition_writer.py tests/distributed/test_local_shard.py tests/distributed/test_sampling_coordinator.py tests/integration/test_foundation_partition_local.py`
Expected: PASS; documentation gap is manual.

**Step 3: Write minimal implementation**

Document that the local partition writer and shard runtime now preserve single-node-type multi-relation graphs, and that partition graph queries can be relation-scoped.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q`
Expected: PASS across the repository.

**Step 5: Commit**

Run: `git add vgl tests docs README.md && git commit -m "feat: add multi-relation partition shard support"`
