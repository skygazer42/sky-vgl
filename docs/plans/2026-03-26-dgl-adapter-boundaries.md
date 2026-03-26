# DGL Graph Adapter Boundary And Count Preservation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Harden the graph-only DGL adapter boundary and preserve external DGL graph node counts plus ID metadata on import.

**Architecture:** Keep `Graph.from_dgl(...)` graph-only by rejecting DGL blocks, then extend the graph import path to normalize `dgl.NID` / `dgl.EID` metadata and inject `n_id` only when needed to preserve declared node counts for featureless or isolate-heavy graphs.

**Tech Stack:** Python 3.10+, PyTorch, pytest

---

### Task 1: Add failing DGL graph adapter regressions

**Files:**
- Modify: `tests/compat/test_dgl_adapter.py`

**Step 1: Write the failing test**

Add regressions proving:

- `Graph.from_dgl(...)` rejects DGL block objects and points callers to `Block.from_dgl(...)`
- homogeneous external DGL graphs keep declared `num_nodes()` even without node features
- heterogeneous external DGL graphs keep declared per-type `num_nodes(...)` even without node features
- external DGL `dgl.NID` / `dgl.EID` metadata imports as `n_id` / `e_id`

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/compat/test_dgl_adapter.py -k "rejects_blocks or preserves_homo_num_nodes or preserves_hetero_num_nodes or normalizes_external_graph_ids"`
Expected: FAIL because the current graph adapter accepts blocks and does not yet preserve counts or normalize external DGL IDs.

**Step 3: Write minimal implementation**

No implementation yet in this task.

**Step 4: Run test to verify it still fails for the intended reason**

Run: `python -m pytest -q tests/compat/test_dgl_adapter.py -k "rejects_blocks or preserves_homo_num_nodes or preserves_hetero_num_nodes or normalizes_external_graph_ids"`
Expected: FAIL on graph adapter boundary or missing count/ID normalization.

**Step 5: Commit**

Do not commit yet.

### Task 2: Implement graph adapter boundary and count preservation

**Files:**
- Modify: `vgl/compat/dgl.py`
- Modify: `tests/compat/test_dgl_adapter.py`

**Step 1: Write the failing test**

Use the regressions from Task 1.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/compat/test_dgl_adapter.py -k "rejects_blocks or preserves_homo_num_nodes or preserves_hetero_num_nodes or normalizes_external_graph_ids"`
Expected: FAIL

**Step 3: Write minimal implementation**

Implement:

- early DGL block rejection in `from_dgl(...)`
- node-count inspection helpers for graph import
- `dgl.NID` -> `n_id` normalization for graph node data
- `dgl.EID` -> `e_id` normalization for graph edge data
- `n_id` fallback injection only when imported node data would otherwise lose declared node cardinality

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/compat/test_dgl_adapter.py -k "rejects_blocks or preserves_homo_num_nodes or preserves_hetero_num_nodes or normalizes_external_graph_ids"`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 3: Refresh migration docs and run full regression

**Files:**
- Modify: `docs/migration-guide.md`

**Step 1: Write the failing test**

No code test. Review migration docs for DGL guidance that still treats graph and block imports as one path or omits the isolated-node preservation behavior.

**Step 2: Run test to verify the code path state**

Run: `python -m pytest -q tests/compat/test_dgl_adapter.py`
Expected: PASS if the adapter changes are complete.

**Step 3: Write minimal implementation**

Document that:

- `Graph.from_dgl(...)` is graph-only
- DGL blocks should use `Block.from_dgl(...)` / `block_from_dgl(...)`
- external DGL `NID` / `EID` metadata imports as `n_id` / `e_id`
- featureless external DGL graphs preserve node counts via imported node-id metadata when needed

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q`
Expected: PASS across the repository.

**Step 5: Commit**

Run: `git add vgl tests docs && git commit -m "feat: harden dgl graph adapter imports"`
