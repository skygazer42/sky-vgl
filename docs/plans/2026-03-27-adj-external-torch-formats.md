# External Adjacency Torch Format Export Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend `adj_external(...)` so graph-level external adjacency export can return native PyTorch COO / CSR / CSC sparse tensors without breaking the existing default behavior.

**Architecture:** Keep `adj_external(...)` as the graph-level export boundary, add one additive `torch_fmt` keyword, and reuse the new `vgl.sparse.to_torch_sparse(...)` helper for compressed torch layouts. Preserve current SciPy behavior, keep default export as torch COO, and reject mixed `scipy_fmt` + `torch_fmt` calls clearly.

**Tech Stack:** Python 3.11+, PyTorch, pytest, SciPy

---

### Task 1: Add failing adj_external torch-format regressions

**Files:**
- Modify: `tests/ops/test_query_ops.py`
- Modify: `tests/core/test_graph_ops_api.py`
- Modify: `tests/core/test_feature_backed_graph.py`

**Step 1: Write the failing test**

Add regressions proving:

- `adj_external(graph, torch_fmt="csr")` returns `torch.sparse_csr`
- `adj_external(graph, torch_fmt="csc", transpose=True)` returns `torch.sparse_csc` with swapped orientation
- default torch COO behavior still works
- `scipy_fmt` and `torch_fmt` cannot be combined
- heterogeneous relation selection works with torch compressed export
- featureless storage-backed graphs preserve declared shape under torch compressed export
- `Graph.adj_external(...)` forwards the new keyword argument

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/ops/test_query_ops.py tests/core/test_graph_ops_api.py tests/core/test_feature_backed_graph.py -k "adj_external and (torch_fmt or compressed or bridge)"`
Expected: FAIL because `adj_external(...)` does not yet accept or handle `torch_fmt`.

**Step 3: Write minimal implementation**

No implementation yet in this task.

**Step 4: Run test to verify it still fails for the intended reason**

Run: `python -m pytest -q tests/ops/test_query_ops.py tests/core/test_graph_ops_api.py tests/core/test_feature_backed_graph.py -k "adj_external and (torch_fmt or compressed or bridge)"`
Expected: FAIL on unexpected keyword arguments or missing compressed export behavior.

**Step 5: Commit**

Do not commit yet.

### Task 2: Implement torch-format selection in adj_external

**Files:**
- Modify: `vgl/ops/query.py`
- Modify: `vgl/graph/graph.py`

**Step 1: Write the failing test**

Use the regressions from Task 1.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/ops/test_query_ops.py tests/core/test_graph_ops_api.py tests/core/test_feature_backed_graph.py -k "adj_external and (torch_fmt or compressed or bridge)"`
Expected: FAIL

**Step 3: Write minimal implementation**

Implement:

- additive `torch_fmt` validation in `adj_external(...)`
- native torch `"coo"`, `"csr"`, and `"csc"` export selection
- rejection when `scipy_fmt` and `torch_fmt` are both supplied
- `Graph.adj_external(...)` signature forwarding the new keyword

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/ops/test_query_ops.py tests/core/test_graph_ops_api.py tests/core/test_feature_backed_graph.py -k "adj_external and (torch_fmt or compressed or bridge)"`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 3: Run focused and full verification

**Files:**
- Modify: any touched modules as needed

**Step 1: Focused verification**

Run: `python -m pytest -q tests/ops/test_query_ops.py tests/core/test_graph_ops_api.py tests/core/test_feature_backed_graph.py tests/sparse/test_sparse_convert.py tests/test_package_layout.py`
Expected: PASS

**Step 2: Full verification**

Run: `python -m pytest -q`
Expected: PASS across the repository.

**Step 3: Prepare integration**

Merge the worktree branch back to `main`, push `origin/main`, and remove the temporary worktree and branch so only `main` remains locally and remotely.

**Step 4: Commit**

Run: `git add vgl tests docs README.md && git commit -m "feat: add compressed torch adjacency export"`
