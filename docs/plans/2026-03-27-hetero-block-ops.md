# Multi-Relation Heterogeneous Block Operation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a reusable multi-relation heterogeneous block abstraction and `to_hetero_block(...)` operation that can express one DGL-style hetero message-flow layer.

**Architecture:** Keep the existing relation-local `Block` unchanged and add an additive `HeteroBlock` container plus `to_hetero_block(...)`. Build one bipartite heterograph layer from a per-type destination frontier, preserve public `n_id` / `e_id` metadata, and expose graph and ops bridge methods without changing sampler contracts in this batch.

**Tech Stack:** Python 3.10+, PyTorch, pytest

---

### Task 1: Add failing hetero block operation regressions

**Files:**
- Modify: `tests/ops/test_block_ops.py`
- Modify: `tests/core/test_graph_ops_api.py`

**Step 1: Write the failing test**

Add regressions proving:

- `to_hetero_block(...)` builds one heterogeneous block layer from multiple relations
- relation restriction via `edge_types=` works
- public `n_id` / `e_id` metadata and feature slices stay aligned
- `HeteroBlock.to(...)` and `pin_memory()` move only tensor state
- `Graph.to_hetero_block(...)` bridges to the ops layer

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/ops/test_block_ops.py -k hetero_block tests/core/test_graph_ops_api.py -k hetero_block`
Expected: FAIL because `HeteroBlock` and `to_hetero_block(...)` do not exist yet.

**Step 3: Write minimal implementation**

No implementation yet in this task.

**Step 4: Run test to verify it still fails for the intended reason**

Run: `python -m pytest -q tests/ops/test_block_ops.py -k hetero_block tests/core/test_graph_ops_api.py -k hetero_block`
Expected: FAIL on missing symbols or missing graph bridge.

**Step 5: Commit**

Do not commit yet.

### Task 2: Implement HeteroBlock and to_hetero_block(...)

**Files:**
- Modify: `vgl/graph/block.py`
- Modify: `vgl/graph/graph.py`
- Modify: `vgl/graph/__init__.py`
- Modify: `vgl/ops/block.py`
- Modify: `vgl/ops/__init__.py`

**Step 1: Write the failing test**

Use the regressions from Task 1.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/ops/test_block_ops.py -k hetero_block tests/core/test_graph_ops_api.py -k hetero_block`
Expected: FAIL

**Step 3: Write minimal implementation**

Implement:

- `HeteroBlock` with `graph`, `edge_types`, `src_n_id`, `dst_n_id`, `src_store_types`, and `dst_store_types`
- helper methods for resolving block edge-store names per original `edge_type`
- non-mutating `.to()` and `pin_memory()`
- `to_hetero_block(...)` that:
  - validates `dst_nodes_by_type`
  - resolves selected `edge_types`
  - slices edges per relation by destination frontier
  - builds per-type source and destination frontiers
  - preserves public `n_id` / `e_id`
  - returns one heterogeneous bipartite layer
- `Graph.to_hetero_block(...)`
- public exports from `vgl.graph` and `vgl.ops`

Implementation rules:

- keep `Block` and `to_block(...)` behavior unchanged
- preserve edge order within each selected relation
- only move tensor fields during transfer helpers
- keep the wrapped graph hetero/temporal according to the source graph schema

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/ops/test_block_ops.py -k hetero_block tests/core/test_graph_ops_api.py -k hetero_block`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 3: Refresh docs and run full regression

**Files:**
- Modify: `README.md`
- Modify: `docs/core-concepts.md`
- Modify: `docs/quickstart.md`

**Step 1: Write the failing test**

No code test. Review docs for places that still describe `to_block(...)` as the only block-building graph op.

**Step 2: Run test to verify the code path state**

Run: `python -m pytest -q tests/ops/test_block_ops.py tests/core/test_graph_ops_api.py`
Expected: PASS if the new block op is complete.

**Step 3: Write minimal implementation**

Document that:

- `to_block(...)` remains relation-local
- `to_hetero_block(...)` constructs one multi-relation heterogeneous message-flow layer
- sampler outputs still use `Block` / relation-local block lists in this batch

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q`
Expected: PASS across the repository.

**Step 5: Commit**

Run: `git add vgl tests docs README.md && git commit -m "feat: add hetero block graph op"`
