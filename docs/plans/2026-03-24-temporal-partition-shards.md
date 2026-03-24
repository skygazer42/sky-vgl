# Temporal Partition Shards Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend the local partition writer and shard loader so temporal homogeneous graphs can flow through the distributed foundation runtime.

**Architecture:** Keep the current partition manifest and shard payload structure intact. Partition files will still store one serialized graph payload plus global `node_ids`, but the payload path will now preserve `time_attr` and temporal edge features so `LocalGraphShard` can rebuild a temporal graph through the existing storage-backed graph path.

**Tech Stack:** Python 3.11, PyTorch, pytest

---

### Task 1: Add temporal partition writer regression coverage

**Files:**
- Modify: `tests/distributed/test_partition_writer.py`

**Step 1: Write the failing test**

Add a test proving `write_partitioned_graph(...)` accepts a temporal homogeneous graph, emits partition payloads whose deserialized graphs still declare `schema.time_attr`, and preserves filtered timestamp edge features inside each partition.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/distributed/test_partition_writer.py -k temporal`
Expected: FAIL because temporal graphs are currently rejected.

**Step 3: Write minimal implementation**

No implementation yet in this task.

**Step 4: Run test to verify it still fails for the intended reason**

Run: `python -m pytest -q tests/distributed/test_partition_writer.py -k temporal`
Expected: FAIL on the temporal-graph guard in `write_partitioned_graph(...)`.

**Step 5: Commit**

Do not commit yet.

### Task 2: Implement temporal partition writing

**Files:**
- Modify: `vgl/distributed/writer.py`
- Modify: `tests/distributed/test_partition_writer.py`

**Step 1: Write the failing test**

Use the test from Task 1.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/distributed/test_partition_writer.py -k temporal`
Expected: FAIL

**Step 3: Write minimal implementation**

Teach `_partition_subgraph(...)` and `write_partitioned_graph(...)` to preserve temporal homogeneous graphs by keeping `time_attr` and returning `Graph.temporal(...)` when the input graph is temporal.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/distributed/test_partition_writer.py -k temporal`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 3: Add temporal shard-loading regression coverage

**Files:**
- Modify: `tests/distributed/test_local_shard.py`

**Step 1: Write the failing test**

Add a test proving `LocalGraphShard.from_partition_dir(...)` can reload a temporal partition, expose `graph.schema.time_attr`, preserve timestamp edge data through the storage-backed graph, and keep local/global id mapping behavior unchanged.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/distributed/test_local_shard.py -k temporal`
Expected: FAIL because temporal partitions are currently rejected.

**Step 3: Write minimal implementation**

No implementation yet in this task.

**Step 4: Run test to verify it still fails for the intended reason**

Run: `python -m pytest -q tests/distributed/test_local_shard.py -k temporal`
Expected: FAIL on the temporal-only guard in `LocalGraphShard.from_partition_dir(...)`.

**Step 5: Commit**

Do not commit yet.

### Task 4: Implement temporal shard loading and integration coverage

**Files:**
- Modify: `vgl/distributed/shard.py`
- Modify: `tests/distributed/test_local_shard.py`
- Modify: `tests/integration/test_foundation_partition_local.py`

**Step 1: Write the failing test**

Use the test from Task 3 and add one integration assertion that a temporal partition shard can still flow into the loader/trainer path.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/distributed/test_local_shard.py -k temporal tests/integration/test_foundation_partition_local.py`
Expected: FAIL

**Step 3: Write minimal implementation**

Allow `LocalGraphShard` to rebuild temporal homogeneous graphs by preserving `time_attr` in the schema and storage-backed graph reconstruction. Update the partition-local integration path to exercise temporal shards through the existing public training API.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/distributed/test_partition_writer.py tests/distributed/test_local_shard.py tests/integration/test_foundation_partition_local.py`
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

No code test. Review docs for places that still describe partition writing and shard loading as homogeneous non-temporal only.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/distributed/test_partition_writer.py tests/distributed/test_local_shard.py tests/integration/test_foundation_partition_local.py`
Expected: PASS; documentation gap is manual.

**Step 3: Write minimal implementation**

Document that the local partition writer and shard loader now support temporal homogeneous graphs while preserving the current manifest/payload format.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q`
Expected: PASS across the repository.

**Step 5: Commit**

Run: `git add vgl tests docs README.md && git commit -m "feat: add temporal partition shard support"`
