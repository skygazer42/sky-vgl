# Lazy Storage-Backed Edge Index Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `Graph.from_storage(...)` resolve structural `edge_index` lazily so storage-backed graphs stop touching graph-store structure during construction.

**Architecture:** Keep `EdgeStore` and `LazyFeatureMap` as the public-facing store layer, but register `edge_index` and edge-feature loaders lazily instead of eagerly materializing graph structure up front. This keeps graph construction metadata-only and caches structure on first access.

**Tech Stack:** Python 3.10+, PyTorch, pytest

---

### Task 1: Add failing lazy-structure regressions

**Files:**
- Modify: `tests/core/test_feature_backed_graph.py`

**Step 1: Write the failing test**

Add regressions proving:

- `Graph.from_storage(...)` does not call `graph_store.edge_index(...)` during construction
- first `graph.edge_index` access calls `graph_store.edge_index(...)` exactly once
- repeated edge-index and adjacency access reuses the cached structure

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/core/test_feature_backed_graph.py -k lazy`
Expected: FAIL because `EdgeStore.from_storage(...)` still eagerly fetches `edge_index`.

**Step 3: Write minimal implementation**

No implementation yet in this task.

**Step 4: Run test to verify it still fails for the intended reason**

Run: `python -m pytest -q tests/core/test_feature_backed_graph.py -k lazy`
Expected: FAIL on eager graph-store structure calls.

**Step 5: Commit**

Do not commit yet.

### Task 2: Implement lazy `edge_index` resolution in storage-backed edge stores

**Files:**
- Modify: `vgl/graph/stores.py`
- Modify: `tests/core/test_feature_backed_graph.py`

**Step 1: Write the failing test**

Use the regressions from Task 1.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/core/test_feature_backed_graph.py -k lazy`
Expected: FAIL

**Step 3: Write minimal implementation**

Teach `EdgeStore.from_storage(...)` to:

- register `edge_index` as a lazy cached loader
- defer `graph_store.edge_count(...)` lookups into edge-feature loaders
- preserve the existing `LazyFeatureMap` cache semantics

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/core/test_feature_backed_graph.py -k lazy`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 3: Run focused and full verification

**Files:**
- Modify: any touched files from prior tasks as needed

**Step 1: Re-read touched code paths**

Confirm lazy structure loading does not change graph semantics once `edge_index` is accessed.

**Step 2: Run focused verification**

Run: `python -m pytest -q tests/core/test_feature_backed_graph.py tests/integration/test_foundation_large_graph_flow.py tests/distributed/test_store_protocol.py tests/distributed/test_sampling_coordinator.py`
Expected: PASS

**Step 3: Run full verification**

Run: `python -m pytest -q`
Expected: PASS across the repository.

**Step 4: Prepare integration**

Merge the worktree branch back to `main`, push `origin/main`, and remove the temporary worktree and branch so only `main` remains.

**Step 5: Commit**

Run: `git add vgl tests docs && git commit -m "feat: lazy-load storage-backed edge indices"`
