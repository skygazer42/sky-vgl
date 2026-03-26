# Degree Query Ops Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add DGL-style `in_degrees(...)` and `out_degrees(...)` query operators so VGL can report scalar, batched, or full-node-space degree counts through the public graph API.

**Architecture:** Extend `vgl.ops.query` with minimal degree-count helpers layered on top of current edge stores and `graph._node_count(...)`. Reuse the existing relation-resolution and node-validation model, return Python `int` for scalar queries, return tensors for vector or all-node queries, and bridge the new operators through `Graph` and `vgl.ops` exports.

**Tech Stack:** Python 3.10+, PyTorch, pytest

---

### Task 1: Add failing degree-query regressions

**Files:**
- Modify: `tests/ops/test_query_ops.py`
- Modify: `tests/core/test_graph_ops_api.py`
- Modify: `tests/core/test_feature_backed_graph.py`
- Modify: `tests/test_package_layout.py`

**Step 1: Write the failing test**

Add regressions proving:

- `in_degrees(...)` and `out_degrees(...)` support scalar, vector, and omitted-node queries
- scalar queries return Python `int`
- heterogeneous relation degree queries count against the correct endpoint type
- invalid node ids raise `ValueError`
- featureless storage-backed graphs preserve zero-degree isolated declared nodes
- `Graph` bridges and `vgl.ops` exports expose the new APIs

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/ops/test_query_ops.py tests/core/test_graph_ops_api.py tests/core/test_feature_backed_graph.py tests/test_package_layout.py -k "in_degrees or out_degrees"`
Expected: FAIL because the new degree-query functions, Graph methods, and exports do not exist yet.

**Step 3: Write minimal implementation**

No implementation yet in this task.

**Step 4: Run test to verify it still fails for the intended reason**

Run: `python -m pytest -q tests/ops/test_query_ops.py tests/core/test_graph_ops_api.py tests/core/test_feature_backed_graph.py tests/test_package_layout.py -k "in_degrees or out_degrees"`
Expected: FAIL on missing imports, missing Graph methods, or missing degree-query behavior.

**Step 5: Commit**

Do not commit yet.

### Task 2: Implement the degree query ops

**Files:**
- Modify: `vgl/ops/query.py`
- Modify: `vgl/ops/__init__.py`
- Modify: `tests/ops/test_query_ops.py`

**Step 1: Write the failing test**

Use the regressions from Task 1.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/ops/test_query_ops.py -k "in_degrees or out_degrees"`
Expected: FAIL

**Step 3: Write minimal implementation**

Implement in `vgl/ops/query.py`:

- `in_degrees(graph, v=None, *, edge_type=None)`
- `out_degrees(graph, u=None, *, edge_type=None)`

Implementation rules:

- resolve the selected relation through the existing tuple-or-default convention
- when no nodes are provided, return all degrees for the declared node space of the relevant endpoint type
- when a scalar node id is provided, return a Python `int`
- when many node ids are provided, return a one-dimensional tensor in request order
- preserve duplicates in the request order
- validate node ids against `graph._node_count(...)`
- return empty tensors for empty vector inputs on the edge-store device

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/ops/test_query_ops.py -k "in_degrees or out_degrees"`
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

Run: `python -m pytest -q tests/ops/test_query_ops.py tests/core/test_graph_ops_api.py tests/core/test_feature_backed_graph.py tests/test_package_layout.py -k "in_degrees or out_degrees"`
Expected: FAIL

**Step 3: Write minimal implementation**

Add:

- `Graph.in_degrees(...)`
- `Graph.out_degrees(...)`

Update `vgl.ops` exports and namespace expectations so the new functions become part of the stable public surface.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/ops/test_query_ops.py tests/core/test_graph_ops_api.py tests/core/test_feature_backed_graph.py tests/test_package_layout.py -k "in_degrees or out_degrees"`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 4: Refresh docs and run the full regression suite

**Files:**
- Modify: `README.md`
- Modify: `docs/core-concepts.md`
- Modify: `docs/quickstart.md`

**Step 1: Write the failing test**

No code test. Review public docs for graph-query descriptions that mention adjacency inspection but not degree queries.

**Step 2: Run test to verify the code path state**

Run: `python -m pytest -q tests/ops/test_query_ops.py tests/core/test_graph_ops_api.py tests/core/test_feature_backed_graph.py tests/test_package_layout.py -k "in_degrees or out_degrees"`
Expected: PASS if the degree-query path is implemented cleanly.

**Step 3: Write minimal implementation**

Document:

- scalar versus tensor return behavior
- omitted-node “all nodes” behavior
- storage-backed isolated-node semantics

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q`
Expected: PASS across the repository.

**Step 5: Commit**

Run: `git add vgl tests docs README.md && git commit -m "feat: add degree query ops"`
