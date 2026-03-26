# Graph Sparse Format State Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add DGL-style `formats(...)` and `create_formats_()` APIs for graph sparse-format status and eager creation.

**Architecture:** Extend `Graph` with lightweight graph-level sparse-format state, update adjacency materialization to record newly created layouts, and add graph-clone helpers so `formats(...)` can return isolated format-state clones while still sharing underlying node/edge tensors. Bridge the new APIs through `vgl.ops`, then document the format-state surface.

**Tech Stack:** Python 3.10+, PyTorch, pytest

---

### Task 1: Add failing graph-format regressions

**Files:**
- Modify: `tests/core/test_graph_sparse_cache.py`
- Modify: `tests/core/test_graph_ops_api.py`
- Modify: `tests/core/test_feature_backed_graph.py`
- Modify: `tests/test_package_layout.py`

**Step 1: Write the failing test**

Add regressions proving:

- fresh graphs report `coo` created and `csr` / `csc` not created
- `formats("csr")` and `formats(["coo", "csr"])` return clones with the right status
- clone format state is isolated from the base graph
- invalid or empty selections fail clearly
- `create_formats_()` eagerly creates all allowed formats and returns `None`
- storage-backed graphs preserve the same status model
- `Graph` bridges and `vgl.ops` exports expose the new API

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/core/test_graph_sparse_cache.py tests/core/test_graph_ops_api.py tests/core/test_feature_backed_graph.py tests/test_package_layout.py -k "formats or create_formats_"`
Expected: FAIL because the new APIs and status state do not exist yet.

**Step 3: Write minimal implementation**

No implementation yet in this task.

**Step 4: Run test to verify it still fails for the intended reason**

Run: `python -m pytest -q tests/core/test_graph_sparse_cache.py tests/core/test_graph_ops_api.py tests/core/test_feature_backed_graph.py tests/test_package_layout.py -k "formats or create_formats_"`
Expected: FAIL on missing imports, missing Graph methods, or missing state behavior.

**Step 5: Commit**

Do not commit yet.

### Task 2: Implement sparse-format state and graph ops

**Files:**
- Modify: `vgl/graph/graph.py`
- Modify: `vgl/graph/stores.py`
- Modify: `vgl/ops/query.py`
- Modify: `tests/core/test_graph_sparse_cache.py`

**Step 1: Write the failing test**

Use the regressions from Task 1.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/core/test_graph_sparse_cache.py -k "formats or create_formats_"`
Expected: FAIL

**Step 3: Write minimal implementation**

Implement:

- graph-level allowed / created sparse-format state
- `formats(graph, formats=None)`
- `create_formats_(graph)`
- adjacency materialization updating created-format state
- clone helpers that share underlying data mappings but isolate caches and format state

Implementation rules:

- support only `coo` / `csr` / `csc`
- fresh graphs start with created `coo`
- `formats(selected)` retains intersection with current created formats
- if the intersection is empty, mark the highest-priority requested format created using canonical DGL order `coo -> csr -> csc`
- `create_formats_()` returns `None`

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/core/test_graph_sparse_cache.py -k "formats or create_formats_"`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 3: Bridge Graph methods, exports, and storage-backed coverage

**Files:**
- Modify: `vgl/ops/__init__.py`
- Modify: `tests/core/test_graph_ops_api.py`
- Modify: `tests/core/test_feature_backed_graph.py`
- Modify: `tests/test_package_layout.py`

**Step 1: Write the failing test**

Use the regressions from Task 1.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/core/test_graph_sparse_cache.py tests/core/test_graph_ops_api.py tests/core/test_feature_backed_graph.py tests/test_package_layout.py -k "formats or create_formats_"`
Expected: FAIL

**Step 3: Write minimal implementation**

Add:

- `Graph.formats(...)`
- `Graph.create_formats_()`
- `vgl.ops.formats`
- `vgl.ops.create_formats_`

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/core/test_graph_sparse_cache.py tests/core/test_graph_ops_api.py tests/core/test_feature_backed_graph.py tests/test_package_layout.py -k "formats or create_formats_"`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 4: Refresh docs and run the full regression suite

**Files:**
- Modify: `README.md`
- Modify: `docs/core-concepts.md`
- Modify: `docs/quickstart.md`

**Step 1: Write the failing test**

No code test. Review sparse and adjacency docs for places that mention export APIs but not graph sparse-format state.

**Step 2: Run test to verify the code path state**

Run: `python -m pytest -q tests/core/test_graph_sparse_cache.py tests/core/test_graph_ops_api.py tests/core/test_feature_backed_graph.py tests/test_package_layout.py -k "formats or create_formats_"`
Expected: PASS if the graph-format path is implemented cleanly.

**Step 3: Write minimal implementation**

Document:

- `formats()` status reporting
- clone semantics
- `create_formats_()` eager creation behavior

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q`
Expected: PASS across the repository.

**Step 5: Commit**

Run: `git add vgl tests docs README.md && git commit -m "feat: add graph sparse format state"`
