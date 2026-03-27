# NetworkX Interoperability Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a first-class homogeneous NetworkX interoperability bridge so VGL graphs can import from directed NetworkX graphs and export back through a stable public API.

**Architecture:** Follow the existing compat-module pattern used for PyG and DGL. Add a dedicated `vgl.compat.networkx` adapter, wire it into `Graph` convenience methods, keep this batch scoped to homogeneous directed graphs, and export through `networkx.MultiDiGraph` to preserve parallel edges.

**Tech Stack:** Python 3.10+, PyTorch, NetworkX, pytest

---

### Task 1: Add failing NetworkX compatibility regressions

**Files:**
- Create: `tests/compat/test_networkx_adapter.py`
- Modify: `tests/test_package_layout.py`

**Step 1: Write the failing test**

Add regressions proving:

- homogeneous VGL graphs export to `networkx.MultiDiGraph` with node and edge attributes preserved
- directed external NetworkX graphs import into homogeneous VGL graphs
- parallel directed edges survive a round-trip
- heterogeneous VGL graphs fail clearly on export
- undirected NetworkX graphs fail clearly on import
- package exports expose the new helper functions

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/compat/test_networkx_adapter.py tests/test_package_layout.py -k networkx`
Expected: FAIL because no NetworkX compatibility surface exists yet.

**Step 3: Write minimal implementation**

No implementation yet in this task.

**Step 4: Run test to verify it still fails for the intended reason**

Run: `python -m pytest -q tests/compat/test_networkx_adapter.py tests/test_package_layout.py -k networkx`
Expected: FAIL on missing imports or missing `Graph` methods.

**Step 5: Commit**

Do not commit yet.

### Task 2: Implement compat adapter and graph methods

**Files:**
- Create: `vgl/compat/networkx.py`
- Modify: `vgl/compat/__init__.py`
- Modify: `vgl/graph/graph.py`
- Modify: `vgl/__init__.py`

**Step 1: Write the failing test**

Use the regressions from Task 1.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/compat/test_networkx_adapter.py tests/test_package_layout.py -k networkx`
Expected: FAIL

**Step 3: Write minimal implementation**

Implement:

- homogeneous `to_networkx(graph)` exporting to `networkx.MultiDiGraph`
- directed `from_networkx(nx_graph)` importing `DiGraph` / `MultiDiGraph`
- `Graph.from_networkx(...)` and `Graph.to_networkx()`
- compat and top-level export wiring consistent with the existing package layout

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/compat/test_networkx_adapter.py tests/test_package_layout.py -k networkx`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 3: Run focused and full verification

**Files:**
- Modify: any touched modules as needed

**Step 1: Focused verification**

Run: `python -m pytest -q tests/compat/test_networkx_adapter.py tests/compat/test_dgl_adapter.py tests/core/test_graph_ops_api.py tests/test_package_layout.py`
Expected: PASS

**Step 2: Full verification**

Run: `python -m pytest -q`
Expected: PASS across the repository.

**Step 3: Prepare integration**

Merge the worktree branch back to `main`, push `origin/main`, and remove the temporary worktree and branch so only `main` remains locally and remotely.

**Step 4: Commit**

Run: `git add vgl tests docs && git commit -m "feat: add networkx graph interoperability"`
