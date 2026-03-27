# Lazy Partition Store Loading Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make partition-directory distributed stores and the store-backed coordinator load partition payloads lazily, one partition at a time, instead of eagerly deserializing every partition during construction.

**Architecture:** Keep the current payload parsing and local adapter logic, but move it behind a shared per-partition bundle cache and thin lazy adapters that satisfy the existing distributed store protocols. Derive metadata such as node counts and edge types from the manifest so construction stays manifest-only.

**Tech Stack:** Python 3.10+, PyTorch, pytest

---

### Task 1: Add failing lazy-loading regressions

**Files:**
- Modify: `tests/distributed/test_store_protocol.py`
- Modify: `tests/distributed/test_sampling_coordinator.py`

**Step 1: Write the failing test**

Add regressions proving:

- `load_partitioned_stores(...)` does not call `torch.load(...)` during construction
- the first query against a partition loads exactly one payload and reuses it across feature and graph queries for that partition
- `StoreBackedSamplingCoordinator.from_partition_dir(...)` can route from manifest metadata without forcing payload loads, and only triggers loading when it fetches real partition data

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/distributed/test_store_protocol.py tests/distributed/test_sampling_coordinator.py -k "lazy"`
Expected: FAIL because partition payloads are still eagerly loaded during construction.

**Step 3: Write minimal implementation**

No implementation yet in this task.

**Step 4: Run test to verify it still fails for the intended reason**

Run: `python -m pytest -q tests/distributed/test_store_protocol.py tests/distributed/test_sampling_coordinator.py -k "lazy"`
Expected: FAIL on eager `torch.load(...)` counts or missing lazy/cache behavior.

**Step 5: Commit**

Do not commit yet.

### Task 2: Implement shared lazy partition bundle caching

**Files:**
- Modify: `vgl/distributed/store.py`
- Modify: `tests/distributed/test_store_protocol.py`

**Step 1: Write the failing test**

Use the regressions from Task 1.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/distributed/test_store_protocol.py -k "lazy"`
Expected: FAIL

**Step 3: Write minimal implementation**

Implement:

- internal per-partition store bundle cache
- lazy feature and graph adapters backed by that cache
- manifest-derived `edge_types` and `num_nodes(...)` paths that avoid payload loads
- lazy default behavior in `load_partitioned_stores(...)`

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/distributed/test_store_protocol.py -k "lazy"`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 3: Verify coordinator lazy behavior

**Files:**
- Modify: `vgl/distributed/coordinator.py`
- Modify: `tests/distributed/test_sampling_coordinator.py`

**Step 1: Write the failing test**

Use the coordinator lazy regression from Task 1.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/distributed/test_sampling_coordinator.py -k "lazy"`
Expected: FAIL

**Step 3: Write minimal implementation**

Ensure `StoreBackedSamplingCoordinator.from_partition_dir(...)` uses the lazy store path and that its manifest-only methods do not force partition payload loads during construction or pure routing.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/distributed/test_sampling_coordinator.py -k "lazy"`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 4: Run focused and full verification

**Files:**
- Modify: any touched files from prior tasks as needed

**Step 1: Write the failing test**

No new test. Re-read the touched code and confirm the public entrypoints remain behavior-compatible.

**Step 2: Run focused verification**

Run: `python -m pytest -q tests/distributed/test_store_protocol.py tests/distributed/test_sampling_coordinator.py tests/data/test_node_neighbor_sampler.py tests/data/test_temporal_neighbor_sampler.py tests/integration/test_foundation_partition_local.py`
Expected: PASS

**Step 3: Run full verification**

Run: `python -m pytest -q`
Expected: PASS across the repository.

**Step 4: Prepare integration**

Merge the worktree branch back to `main`, push `origin/main`, and remove the temporary worktree and branch so only `main` remains.

**Step 5: Commit**

Run: `git add vgl tests docs && git commit -m "feat: lazy-load partition stores"`
