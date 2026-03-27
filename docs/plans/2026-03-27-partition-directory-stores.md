# Partition-Directory Store Assembly Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Let VGL build partition-aware distributed stores and the store-backed sampling coordinator directly from a partition directory without going through `LocalGraphShard`.

**Architecture:** Parse `manifest.json` and partition payloads directly in `vgl.distributed.store`, build one local feature adapter and one local graph adapter per partition, aggregate them into `PartitionedFeatureStore` and `PartitionedGraphStore`, and expose `StoreBackedSamplingCoordinator.from_partition_dir(...)` as the public entrypoint.

**Tech Stack:** Python 3.10+, PyTorch, pytest

---

### Task 1: Add failing regressions for direct partition-directory assembly

**Files:**
- Modify: `tests/distributed/test_store_protocol.py`
- Modify: `tests/distributed/test_sampling_coordinator.py`
- Modify: `tests/data/test_node_neighbor_sampler.py`
- Modify: `tests/data/test_temporal_neighbor_sampler.py`

**Step 1: Write the failing test**

Add regressions proving:

- `load_partitioned_stores(root)` returns a manifest plus partition-aware distributed stores built directly from partition files
- the direct stores expose local edge indexes, boundary edge indexes, node features, and boundary edge features without `LocalGraphShard`
- `StoreBackedSamplingCoordinator.from_partition_dir(root)` routes node and edge ids and fetches features without helper assembly
- existing stitched store-backed sampling tests can use the direct coordinator entrypoint

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/distributed/test_store_protocol.py tests/distributed/test_sampling_coordinator.py tests/data/test_node_neighbor_sampler.py tests/data/test_temporal_neighbor_sampler.py -k "from_partition_dir or load_partitioned_stores"`
Expected: FAIL because the direct loading helpers and coordinator classmethod do not exist yet.

**Step 3: Write minimal implementation**

No implementation yet in this task.

**Step 4: Run test to verify it still fails for the intended reason**

Run: `python -m pytest -q tests/distributed/test_store_protocol.py tests/distributed/test_sampling_coordinator.py tests/data/test_node_neighbor_sampler.py tests/data/test_temporal_neighbor_sampler.py -k "from_partition_dir or load_partitioned_stores"`
Expected: FAIL on missing functions or missing classmethod.

**Step 5: Commit**

Do not commit yet.

### Task 2: Implement direct partition-directory store loading

**Files:**
- Modify: `vgl/distributed/store.py`
- Modify: `vgl/distributed/__init__.py`
- Modify: `tests/distributed/test_store_protocol.py`

**Step 1: Write the failing test**

Use the regressions from Task 1.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/distributed/test_store_protocol.py -k "load_partitioned_stores"`
Expected: FAIL

**Step 3: Write minimal implementation**

Implement:

- partition payload parsing helpers
- `load_partitioned_stores(root)`
- direct assembly of `PartitionedFeatureStore` and `PartitionedGraphStore` from on-disk payloads using the existing local adapters
- public exports for the new helper

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/distributed/test_store_protocol.py -k "load_partitioned_stores"`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 3: Implement coordinator direct entrypoint

**Files:**
- Modify: `vgl/distributed/coordinator.py`
- Modify: `vgl/distributed/__init__.py`
- Modify: `tests/distributed/test_sampling_coordinator.py`

**Step 1: Write the failing test**

Use the regressions from Task 1.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/distributed/test_sampling_coordinator.py -k "from_partition_dir"`
Expected: FAIL

**Step 3: Write minimal implementation**

Implement `StoreBackedSamplingCoordinator.from_partition_dir(root)` by calling `load_partitioned_stores(root)` and forwarding the result into the existing constructor.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/distributed/test_sampling_coordinator.py -k "from_partition_dir"`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 4: Switch store-backed sampling tests to the direct entrypoint

**Files:**
- Modify: `tests/data/test_node_neighbor_sampler.py`
- Modify: `tests/data/test_temporal_neighbor_sampler.py`

**Step 1: Write the failing test**

Update the existing store-backed coordinator helpers in these tests so they call `StoreBackedSamplingCoordinator.from_partition_dir(tmp_path)` instead of manually building partitioned stores from `LocalGraphShard`.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/data/test_node_neighbor_sampler.py tests/data/test_temporal_neighbor_sampler.py -k "store_backed"`
Expected: FAIL until the direct entrypoint is wired correctly.

**Step 3: Write minimal implementation**

Adjust test helpers to use the direct coordinator entrypoint and keep local shard graphs only where the sampled graph fixture itself still needs them.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/data/test_node_neighbor_sampler.py tests/data/test_temporal_neighbor_sampler.py -k "store_backed"`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 5: Run focused and full verification

**Files:**
- Modify: any touched files from prior tasks as needed

**Step 1: Write the failing test**

No new test. Re-read the scope and confirm the direct entrypoint no longer relies on `LocalGraphShard`.

**Step 2: Run focused verification**

Run: `python -m pytest -q tests/distributed/test_store_protocol.py tests/distributed/test_sampling_coordinator.py tests/data/test_node_neighbor_sampler.py tests/data/test_temporal_neighbor_sampler.py tests/integration/test_foundation_partition_local.py`
Expected: PASS

**Step 3: Run full verification**

Run: `python -m pytest -q`
Expected: PASS across the repository.

**Step 4: Prepare integration**

Merge the worktree branch back to `main`, push `origin/main`, and remove the temporary worktree and branch so only `main` remains.

**Step 5: Commit**

Run: `git add vgl tests docs && git commit -m "feat: load distributed stores from partition directories"`
