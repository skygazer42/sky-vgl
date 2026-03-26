# To-Block And Block Container Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a DGL-style relation-local `to_block(...)` transform and a lightweight `Block` container so VGL exposes a first-class message-flow block foundation.

**Architecture:** Keep the new transform graph-first and relation-local. `to_block(...)` will compact one selected relation into a bipartite block graph, while a new `Block` wrapper will carry the remapped graph plus source/destination id metadata and transfer helpers without changing current loader outputs.

**Tech Stack:** Python 3.11, PyTorch, pytest

---

### Task 1: Add failing graph-op regressions for to_block

**Files:**
- Create: `tests/ops/test_block_ops.py`

**Step 1: Write the failing test**

Add tests proving:

- `to_block(...)` builds a homogeneous block from incoming edges to destination seeds
- homogeneous blocks expose `src_n_id`, `dst_n_id`, and edge `e_id`
- `include_dst_in_src=False` drops destination seeds from the source frontier unless they are true predecessors
- relation-local heterogeneous `to_block(...)` preserves endpoint node types and feature slices
- an empty incoming frontier still returns a valid block with destination nodes intact
- ambiguous relation selection or out-of-range destination ids fail with clear `ValueError`s

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/ops/test_block_ops.py`
Expected: FAIL because `to_block(...)` and `Block` do not exist yet.

**Step 3: Write minimal implementation**

No implementation yet in this task.

**Step 4: Run test to verify it still fails for the intended reason**

Run: `python -m pytest -q tests/ops/test_block_ops.py`
Expected: FAIL on missing imports or missing functions/classes.

**Step 5: Commit**

Do not commit yet.

### Task 2: Implement Block and public to_block

**Files:**
- Create: `vgl/graph/block.py`
- Create: `vgl/ops/block.py`
- Modify: `vgl/graph/__init__.py`
- Modify: `vgl/ops/__init__.py`
- Modify: `tests/ops/test_block_ops.py`

**Step 1: Write the failing test**

Use the tests from Task 1.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/ops/test_block_ops.py`
Expected: FAIL

**Step 3: Write minimal implementation**

Implement:

- `Block(graph, edge_type, src_type, dst_type, src_n_id, dst_n_id, src_store_type, dst_store_type)`
- `to_block(graph, dst_nodes, *, edge_type=None, include_dst_in_src=True)`

Implementation rules:

- operate on exactly one selected relation
- preserve selected edge order from the source graph
- always attach `n_id` to source/destination node stores and `e_id` to the block edge store
- slice node- and edge-aligned features into the block graph
- use internal source/destination node store names only when the original relation has the same endpoint node type
- expose `srcdata`, `dstdata`, `edata`, `edge_index`, `to(...)`, and `pin_memory()` on `Block`
- keep temporal `time_attr` on the wrapped graph when the selected relation comes from a temporal graph

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/ops/test_block_ops.py`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 3: Bridge Graph methods and stable exports

**Files:**
- Modify: `vgl/graph/graph.py`
- Modify: `vgl/core/__init__.py`
- Modify: `vgl/__init__.py`
- Modify: `tests/core/test_graph_ops_api.py`
- Modify: `tests/test_package_layout.py`

**Step 1: Write the failing test**

Extend coverage to prove:

- `graph.to_block(...)` delegates correctly
- `vgl.graph.Block` and `vgl.ops.to_block` are stable exports
- top-level `vgl.Block` also resolves

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/core/test_graph_ops_api.py tests/test_package_layout.py -k "to_block or Block or ops_all"`
Expected: FAIL because the `Graph` bridge and exports are incomplete.

**Step 3: Write minimal implementation**

Add `Graph.to_block(...)`, re-export `Block` through `vgl.graph`, `vgl.core`, and top-level `vgl`, and include `to_block` in `vgl.ops.__all__` and package-layout expectations.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/ops/test_block_ops.py tests/core/test_graph_ops_api.py tests/test_package_layout.py`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 4: Refresh docs and run branch verification

**Files:**
- Modify: `README.md`
- Modify: `docs/core-concepts.md`
- Modify: `docs/quickstart.md`

**Step 1: Write the failing test**

No code test. Review docs for graph-ops and mini-batch sections that currently stop at sampled subgraphs and never mention blocks.

**Step 2: Run test to verify the code path state**

Run: `python -m pytest -q tests/ops/test_block_ops.py tests/core/test_graph_ops_api.py tests/test_package_layout.py`
Expected: PASS; documentation gap is manual.

**Step 3: Write minimal implementation**

Document:

- `to_block(...)` as a relation-local message-flow transform
- `Block` as the compact source/destination container
- `Graph.to_block(...)` as the convenience bridge

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q`
Expected: PASS across the repository.

**Step 5: Commit**

Run: `git add vgl tests docs README.md && git commit -m "feat: add to_block graph transform"`
