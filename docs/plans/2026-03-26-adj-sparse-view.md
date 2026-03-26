# Weighted Adjacency Sparse View Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a DGL-style `adj(...)` API that returns weighted or unweighted sparse adjacency views through VGL's `SparseTensor`.

**Architecture:** Extend `vgl.ops.query` with one adjacency sparse-view builder that resolves relations like the other query ops, orders visible entries by public `e_id`, and constructs COO / CSR / CSC `SparseTensor` outputs directly so weighted values and compressed ordering stay aligned. Bridge the new op through `Graph` and `vgl.ops`, then refresh docs around adjacency-oriented structure exports.

**Tech Stack:** Python 3.10+, PyTorch, pytest

---

### Task 1: Add failing adjacency-view regressions

**Files:**
- Modify: `tests/ops/test_query_ops.py`
- Modify: `tests/core/test_graph_ops_api.py`
- Modify: `tests/core/test_feature_backed_graph.py`
- Modify: `tests/test_package_layout.py`

**Step 1: Write the failing test**

Add regressions proving:

- `adj(...)` returns a `SparseTensor`
- `eweight_name` controls sparse values
- COO output follows public `e_id` ordering
- CSR / CSC preserve public edge order within compressed buckets
- heterogeneous graphs work with explicit `edge_type`
- missing edge-weight features fail clearly
- featureless storage-backed graphs preserve declared node-space shape
- `Graph` bridges and `vgl.ops` exports expose the new API

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/ops/test_query_ops.py tests/core/test_graph_ops_api.py tests/core/test_feature_backed_graph.py tests/test_package_layout.py -k "adj and not adj_tensors and not adjacency"`
Expected: FAIL because the new op, Graph bridge, and exports do not exist yet.

**Step 3: Write minimal implementation**

No implementation yet in this task.

**Step 4: Run test to verify it still fails for the intended reason**

Run: `python -m pytest -q tests/ops/test_query_ops.py tests/core/test_graph_ops_api.py tests/core/test_feature_backed_graph.py tests/test_package_layout.py -k "adj and not adj_tensors and not adjacency"`
Expected: FAIL on missing imports, missing Graph methods, or missing query behavior.

**Step 5: Commit**

Do not commit yet.

### Task 2: Implement weighted adjacency sparse views

**Files:**
- Modify: `vgl/ops/query.py`
- Modify: `tests/ops/test_query_ops.py`

**Step 1: Write the failing test**

Use the regressions from Task 1.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/ops/test_query_ops.py -k "adj and not adj_tensors and not adjacency"`
Expected: FAIL

**Step 3: Write minimal implementation**

Implement in `vgl/ops/query.py`:

- `adj(graph, *, edge_type=None, eweight_name=None, layout="coo")`

Implementation rules:

- normalize layout from string or `SparseLayout`
- resolve the target relation with the same helper as other query ops
- use public-`e_id` ordering for visible structure
- build `SparseTensor` directly for COO / CSR / CSC
- use edge feature values when `eweight_name` is provided
- preserve declared node counts in shape and pointer lengths
- reject missing edge-weight features clearly

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/ops/test_query_ops.py -k "adj and not adj_tensors and not adjacency"`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 3: Bridge Graph methods and stable exports

**Files:**
- Modify: `vgl/graph/graph.py`
- Modify: `vgl/ops/__init__.py`
- Modify: `tests/core/test_graph_ops_api.py`
- Modify: `tests/core/test_feature_backed_graph.py`
- Modify: `tests/test_package_layout.py`

**Step 1: Write the failing test**

Use the regressions from Task 1.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/ops/test_query_ops.py tests/core/test_graph_ops_api.py tests/core/test_feature_backed_graph.py tests/test_package_layout.py -k "adj and not adj_tensors and not adjacency"`
Expected: FAIL

**Step 3: Write minimal implementation**

Add:

- `Graph.adj(...)`

Update `vgl.ops` exports and namespace expectations so `adj` becomes part of the stable public surface.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/ops/test_query_ops.py tests/core/test_graph_ops_api.py tests/core/test_feature_backed_graph.py tests/test_package_layout.py -k "adj and not adj_tensors and not adjacency"`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 4: Refresh docs and run the full regression suite

**Files:**
- Modify: `README.md`
- Modify: `docs/core-concepts.md`
- Modify: `docs/quickstart.md`

**Step 1: Write the failing test**

No code test. Review adjacency-oriented docs for places that mention `Graph.adjacency(...)` but not the DGL-style `adj(...)` entry point.

**Step 2: Run test to verify the code path state**

Run: `python -m pytest -q tests/ops/test_query_ops.py tests/core/test_graph_ops_api.py tests/core/test_feature_backed_graph.py tests/test_package_layout.py -k "adj and not adj_tensors and not adjacency"`
Expected: PASS if the adjacency-view path is implemented cleanly.

**Step 3: Write minimal implementation**

Document:

- the `adj(...)` return type
- `eweight_name` semantics
- public-`e_id` ordering and compressed-bucket stability

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q`
Expected: PASS across the repository.

**Step 5: Commit**

Run: `git add vgl tests docs README.md && git commit -m "feat: add weighted adjacency sparse views"`
