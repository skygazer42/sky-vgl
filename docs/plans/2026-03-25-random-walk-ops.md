# Random Walk Ops Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add DGL-class `random_walk(...)` and `metapath_random_walk(...)` path-building operators to `vgl.ops` and bridge them through `Graph` convenience methods.

**Architecture:** Extend the existing `vgl.ops.path` module with two walk helpers that build dense node-trace tensors from one selected relation or from a typed metapath. Keep the implementation simple, tensor-oriented at the boundary, and non-mutating, while exporting the operations through both `vgl.ops` and `Graph`.

**Tech Stack:** Python 3.11, PyTorch, pytest

---

### Task 1: Add failing walk regressions and export assertions

**Files:**
- Modify: `tests/ops/test_path_ops.py`
- Modify: `tests/core/test_graph_ops_api.py`
- Modify: `tests/test_package_layout.py`

**Step 1: Write the failing test**

Add tests proving:

- `random_walk(...)` follows a deterministic homogeneous path and includes the seed in the trace
- `random_walk(...)` pads later steps with `-1` after a dead end
- `random_walk(...)` can target one selected relation on a single-node-type multi-relation graph
- `metapath_random_walk(...)` follows a heterogeneous metapath across node types
- `metapath_random_walk(...)` pads with `-1` after a missing relation step
- `random_walk(...)` rejects repeated walks over non-self-composable relations
- `metapath_random_walk(...)` rejects non-composable metapaths
- `Graph` bridges and `vgl.ops.__all__` expose the new functions

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/ops/test_path_ops.py tests/core/test_graph_ops_api.py tests/test_package_layout.py`
Expected: FAIL because the new walk operations and exports do not exist yet.

**Step 3: Write minimal implementation**

No implementation yet in this task.

**Step 4: Run test to verify it still fails for the intended reason**

Run: `python -m pytest -q tests/ops/test_path_ops.py tests/core/test_graph_ops_api.py tests/test_package_layout.py`
Expected: FAIL on missing walk functions or missing Graph bridges.

**Step 5: Commit**

Do not commit yet.

### Task 2: Implement random_walk and metapath_random_walk in vgl.ops

**Files:**
- Modify: `vgl/ops/path.py`
- Modify: `vgl/ops/__init__.py`
- Modify: `tests/ops/test_path_ops.py`

**Step 1: Write the failing test**

Use the tests from Task 1.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/ops/test_path_ops.py tests/core/test_graph_ops_api.py tests/test_package_layout.py`
Expected: FAIL

**Step 3: Write minimal implementation**

Implement:

- `random_walk(graph, seeds, *, length, edge_type=None)`
- `metapath_random_walk(graph, seeds, metapath)`

Implementation rules:

- return dense node-id traces with the seed in column 0
- pad later steps with `-1` after a walk terminates
- use uniform random sampling among outgoing neighbors
- validate seed ranges against the starting node type
- reject invalid negative lengths
- reject repeated multi-step walks over non-self-composable relations
- validate metapath edge-type chaining with the same rules used by `metapath_reachable_graph(...)`

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/ops/test_path_ops.py tests/core/test_graph_ops_api.py tests/test_package_layout.py`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 3: Add Graph bridges and refresh docs

**Files:**
- Modify: `vgl/graph/graph.py`
- Modify: `README.md`
- Modify: `docs/core-concepts.md`
- Modify: `docs/quickstart.md`

**Step 1: Write the failing test**

Use the Graph bridge and package export tests from Task 1.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/core/test_graph_ops_api.py tests/test_package_layout.py -k "walk or ops_all"`
Expected: FAIL until the Graph methods and export list are complete.

**Step 3: Write minimal implementation**

Add `Graph.random_walk(...)` and `Graph.metapath_random_walk(...)` convenience methods that delegate into `vgl.ops`, then document the new walk operators in the foundation-layer docs.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/ops/test_path_ops.py tests/core/test_graph_ops_api.py tests/test_package_layout.py`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 4: Run full regression and commit

**Files:**
- Modify: `docs/plans/2026-03-25-random-walk-ops-design.md`
- Modify: `docs/plans/2026-03-25-random-walk-ops.md`

**Step 1: Write the failing test**

No new code test. Review final behavior against the plan.

**Step 2: Run test to verify the code path state**

Run: `python -m pytest -q tests/ops/test_path_ops.py tests/core/test_graph_ops_api.py tests/test_package_layout.py`
Expected: PASS

**Step 3: Write minimal implementation**

No new code. Keep the plan docs aligned with the implemented batch.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q`
Expected: PASS across the repository.

**Step 5: Commit**

Run: `git add vgl tests docs README.md && git commit -m "feat: add random walk graph ops"`
