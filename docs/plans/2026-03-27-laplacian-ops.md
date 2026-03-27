# Laplacian Sparse View Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a public `laplacian(...)` sparse graph operator that returns unnormalized or normalized Laplacian matrices on square graph relations.

**Architecture:** Extend `vgl.ops.query` with one additive `laplacian(...)` function that resolves a selected relation, computes row-degree-based Laplacian entries with duplicate-coordinate aggregation, returns a `SparseTensor` in COO/CSR/CSC layout, and bridges through `Graph` plus `vgl.ops`.

**Tech Stack:** Python 3.11+, PyTorch, pytest

---

### Task 1: Add failing Laplacian regressions

**Files:**
- Modify: `tests/ops/test_query_ops.py`
- Modify: `tests/core/test_graph_ops_api.py`
- Modify: `tests/core/test_feature_backed_graph.py`
- Modify: `tests/test_package_layout.py`

**Step 1: Write the failing test**

Add regressions proving:

- `laplacian(...)` returns `D - A` for weighted homogeneous graphs
- `normalization="rw"` and `normalization="sym"` produce the expected normalized sparse values
- square heterogeneous relations are supported
- bipartite relations and invalid normalization names are rejected
- `Graph.laplacian(...)` forwards to the new query op
- `vgl.ops` exports `laplacian`
- featureless storage-backed graphs preserve declared shape

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/ops/test_query_ops.py tests/core/test_graph_ops_api.py tests/core/test_feature_backed_graph.py tests/test_package_layout.py -k "laplacian"`
Expected: FAIL because the operator, Graph bridge, and namespace export do not exist yet.

**Step 3: Write minimal implementation**

No implementation yet in this task.

**Step 4: Run test to verify it still fails for the intended reason**

Run: `python -m pytest -q tests/ops/test_query_ops.py tests/core/test_graph_ops_api.py tests/core/test_feature_backed_graph.py tests/test_package_layout.py -k "laplacian"`
Expected: FAIL on missing imports, missing Graph methods, or missing operator behavior.

**Step 5: Commit**

Do not commit yet.

### Task 2: Implement `laplacian(...)`

**Files:**
- Modify: `vgl/ops/query.py`
- Modify: `vgl/ops/__init__.py`
- Modify: `vgl/graph/graph.py`

**Step 1: Write the failing test**

Use the regressions from Task 1.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/ops/test_query_ops.py tests/core/test_graph_ops_api.py tests/core/test_feature_backed_graph.py tests/test_package_layout.py -k "laplacian"`
Expected: FAIL

**Step 3: Write minimal implementation**

Implement:

- `laplacian(graph, *, edge_type=None, normalization=None, eweight_name=None, layout="coo")` in `vgl.ops.query`
- `Graph.laplacian(...)`
- `vgl.ops.laplacian`

Implementation rules:

- require matching source and destination node types
- support `normalization=None`, `"rw"`, and `"sym"`
- use unit weights when `eweight_name` is omitted
- require scalar edge weights
- preserve declared node-space shape for storage-backed graphs
- aggregate duplicate coordinates into one visible sparse entry

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/ops/test_query_ops.py tests/core/test_graph_ops_api.py tests/core/test_feature_backed_graph.py tests/test_package_layout.py -k "laplacian"`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 3: Refresh docs and verify the branch

**Files:**
- Modify: `README.md`
- Modify: `docs/core-concepts.md`
- Modify: `docs/quickstart.md`

**Step 1: Write the failing test**

No code test. Review graph-query and sparse-view docs for places that should mention Laplacian support.

**Step 2: Run focused verification**

Run: `python -m pytest -q tests/ops/test_query_ops.py tests/core/test_graph_ops_api.py tests/core/test_feature_backed_graph.py tests/test_package_layout.py`
Expected: PASS if the operator is implemented cleanly.

**Step 3: Write minimal implementation**

Document:

- `laplacian(...)` as a non-mutating sparse graph operator
- the supported normalization modes
- square-relation requirement
- weighted behavior through `eweight_name=`

**Step 4: Run full verification**

Run: `python -m pytest -q`
Expected: PASS across the repository.

**Step 5: Commit**

Run: `git add vgl tests docs README.md && git commit -m "feat: add laplacian sparse view"`
