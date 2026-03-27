# CSV Table Interoperability Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add homogeneous paired node/edge CSV table interoperability so VGL can persist and restore node features, edge features, isolates, and public node ids without adding new dependencies.

**Architecture:** Introduce a small `vgl.compat.csv_tables` adapter that reads and writes `nodes.csv` plus `edges.csv` using the Python standard-library `csv` module and `pathlib.Path`. Import will map public integer node ids into internal contiguous positions while retaining them as `n_id`, and export will prefer `n_id` when present so round-trips preserve public node identity.

**Tech Stack:** Python 3.11+, PyTorch, pytest, stdlib `csv`

---

### Task 1: Add failing paired-table compatibility regressions

**Files:**
- Create: `tests/compat/test_csv_table_adapter.py`
- Modify: `tests/test_package_layout.py`

**Step 1: Write the failing test**

Add regressions proving:

- homogeneous graphs round-trip through `nodes.csv` and `edges.csv` with node features, edge features, and public node ids preserved
- custom column names and custom delimiters work
- isolated nodes survive through explicit node rows
- import fails when an edge references a node id missing from the node table
- non-numeric node or edge feature columns fail clearly
- heterogeneous graphs fail clearly on export
- compat exports expose the new helper functions

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/compat/test_csv_table_adapter.py tests/test_package_layout.py -k csv_table`
Expected: FAIL because no paired-table compatibility surface exists yet.

**Step 3: Write minimal implementation**

No implementation yet in this task.

**Step 4: Run test to verify it still fails for the intended reason**

Run: `python -m pytest -q tests/compat/test_csv_table_adapter.py tests/test_package_layout.py -k csv_table`
Expected: FAIL on missing imports or missing `Graph` methods.

**Step 5: Commit**

Do not commit yet.

### Task 2: Implement CSV table adapter and graph methods

**Files:**
- Create: `vgl/compat/csv_tables.py`
- Modify: `vgl/compat/__init__.py`
- Modify: `vgl/graph/graph.py`

**Step 1: Write the failing test**

Use the regressions from Task 1.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/compat/test_csv_table_adapter.py tests/test_package_layout.py -k csv_table`
Expected: FAIL

**Step 3: Write minimal implementation**

Implement:

- `from_csv_tables(...)` using `csv.DictReader`
- `to_csv_tables(...)` using `csv.DictWriter`
- public-node-id mapping through `n_id`
- numeric scalar feature inference and validation
- homogeneous-only export validation
- `Graph.from_csv_tables(...)` and `graph.to_csv_tables()`
- compat export wiring consistent with the existing package layout

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/compat/test_csv_table_adapter.py tests/test_package_layout.py -k csv_table`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 3: Run focused and full verification

**Files:**
- Modify: any touched modules as needed

**Step 1: Focused verification**

Run: `python -m pytest -q tests/compat/test_csv_table_adapter.py tests/compat/test_edge_list_csv_adapter.py tests/compat/test_edge_list_adapter.py tests/compat/test_networkx_adapter.py tests/compat/test_dgl_adapter.py tests/compat/test_pyg_adapter.py tests/test_package_layout.py`
Expected: PASS

**Step 2: Full verification**

Run: `python -m pytest -q`
Expected: PASS across the repository.

**Step 3: Prepare integration**

Merge the worktree branch back to `main`, push `origin/main`, and remove the temporary worktree and branch so only `main` remains locally and remotely.

**Step 4: Commit**

Run: `git add vgl tests docs README.md && git commit -m "feat: add csv table interoperability"`
