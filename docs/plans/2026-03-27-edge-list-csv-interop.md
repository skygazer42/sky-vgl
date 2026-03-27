# Edge-List CSV Interoperability Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a first-class CSV/tabular edge-list file bridge so VGL can persist and restore simple homogeneous graphs without adding new dependencies.

**Architecture:** Build a small `vgl.compat.edge_list_csv` adapter on top of the existing in-memory `from_edge_list(...)` and `to_edge_list(...)` helpers. Keep the batch homogeneous-only, use the Python standard-library `csv` module plus `pathlib.Path`, infer numeric edge-feature columns conservatively, and expose matching `Graph` convenience methods.

**Tech Stack:** Python 3.11+, PyTorch, pytest, stdlib `csv`

---

### Task 1: Add failing CSV edge-list compatibility regressions

**Files:**
- Create: `tests/compat/test_edge_list_csv_adapter.py`
- Modify: `tests/test_package_layout.py`

**Step 1: Write the failing test**

Add regressions proving:

- homogeneous VGL graphs export to CSV and import back with ordered structure and rank-1 numeric edge features preserved
- custom column names and custom delimiters work
- explicit `num_nodes` preserves isolated nodes when importing a CSV edge list
- header-only CSV files import as empty graphs
- mixed or non-numeric edge feature columns fail clearly
- heterogeneous VGL graphs fail clearly on CSV export
- compat exports expose the new helper functions

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/compat/test_edge_list_csv_adapter.py tests/test_package_layout.py -k edge_list_csv`
Expected: FAIL because no CSV edge-list compatibility surface exists yet.

**Step 3: Write minimal implementation**

No implementation yet in this task.

**Step 4: Run test to verify it still fails for the intended reason**

Run: `python -m pytest -q tests/compat/test_edge_list_csv_adapter.py tests/test_package_layout.py -k edge_list_csv`
Expected: FAIL on missing imports or missing `Graph` methods.

**Step 5: Commit**

Do not commit yet.

### Task 2: Implement CSV adapter and graph methods

**Files:**
- Create: `vgl/compat/edge_list_csv.py`
- Modify: `vgl/compat/__init__.py`
- Modify: `vgl/graph/graph.py`

**Step 1: Write the failing test**

Use the regressions from Task 1.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/compat/test_edge_list_csv_adapter.py tests/test_package_layout.py -k edge_list_csv`
Expected: FAIL

**Step 3: Write minimal implementation**

Implement:

- `from_edge_list_csv(...)` using `csv.DictReader`
- `to_edge_list_csv(...)` using `csv.DictWriter`
- numeric edge-feature inference and validation
- homogeneous-only export validation
- `Graph.from_edge_list_csv(...)` and `graph.to_edge_list_csv()`
- compat export wiring consistent with the existing package layout

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/compat/test_edge_list_csv_adapter.py tests/test_package_layout.py -k edge_list_csv`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 3: Run focused and full verification

**Files:**
- Modify: any touched modules as needed

**Step 1: Focused verification**

Run: `python -m pytest -q tests/compat/test_edge_list_csv_adapter.py tests/compat/test_edge_list_adapter.py tests/compat/test_networkx_adapter.py tests/compat/test_dgl_adapter.py tests/compat/test_pyg_adapter.py tests/test_package_layout.py`
Expected: PASS

**Step 2: Full verification**

Run: `python -m pytest -q`
Expected: PASS across the repository.

**Step 3: Prepare integration**

Merge the worktree branch back to `main`, push `origin/main`, and remove the temporary worktree and branch so only `main` remains locally and remotely.

**Step 4: Commit**

Run: `git add vgl tests docs && git commit -m "feat: add edge-list csv interoperability"`
