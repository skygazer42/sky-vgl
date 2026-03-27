# HeteroBlock DGL Interoperability Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add explicit DGL import/export support for multi-relation `HeteroBlock` while keeping existing single-relation `Block` compatibility behavior unchanged.

**Architecture:** Extend the DGL compatibility layer with dedicated `hetero_block_from_dgl(...)` / `hetero_block_to_dgl(...)` helpers, wire those helpers into `HeteroBlock.from_dgl(...)` / `HeteroBlock.to_dgl()`, and cover the new path with fake-DGL adapter tests that exercise same-type and bipartite relations together.

**Tech Stack:** Python 3.11, PyTorch, pytest, fake-DGL adapter tests, VGL graph/block compatibility layer

---

### Task 1: Add failing DGL adapter regressions for `HeteroBlock`

**Files:**
- Modify: `tests/compat/test_dgl_adapter.py`

**Step 1: Write the failing tests**

Add tests that verify:
- a multi-relation `HeteroBlock` round-trips through DGL
- an external multi-relation DGL block imports into `HeteroBlock`
- temporal `time_attr` survives the round-trip
- `Block.from_dgl(...)` still rejects multi-relation blocks

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/compat/test_dgl_adapter.py -k 'hetero_block or multi_relation_block'`
Expected: FAIL because `HeteroBlock` has no DGL adapter path yet

**Step 3: Write minimal implementation**

No production changes yet.

**Step 4: Re-run to confirm the red state**

Run: `python -m pytest -q tests/compat/test_dgl_adapter.py -k 'hetero_block or multi_relation_block'`
Expected: still FAIL for the new missing methods/helpers

**Step 5: Commit**

```bash
git add tests/compat/test_dgl_adapter.py
git commit -m "test: cover hetero block dgl interop"
```

### Task 2: Implement dedicated `HeteroBlock` DGL helpers

**Files:**
- Modify: `vgl/compat/dgl.py`
- Modify: `vgl/compat/__init__.py`
- Modify: `vgl/graph/block.py`

**Step 1: Write minimal implementation**

Add:
- `hetero_block_from_dgl(...)`
- `hetero_block_to_dgl(...)`
- `HeteroBlock.from_dgl(...)`
- `HeteroBlock.to_dgl()`

Keep:
- `block_from_dgl(...)` strict for single-relation `Block`

**Step 2: Run focused tests**

Run: `python -m pytest -q tests/compat/test_dgl_adapter.py`
Expected: PASS

**Step 3: Clean up**

Refactor any shared DGL block normalization helpers only if duplication is obvious.

**Step 4: Re-run focused tests**

Run: `python -m pytest -q tests/compat/test_dgl_adapter.py`
Expected: PASS

**Step 5: Commit**

```bash
git add vgl/compat/dgl.py vgl/compat/__init__.py vgl/graph/block.py
git commit -m "feat: add hetero block dgl interop"
```

### Task 3: Update docs and verify repository-wide behavior

**Files:**
- Modify: `README.md`
- Modify: `docs/core-concepts.md`
- Modify: `docs/quickstart.md`

**Step 1: Write doc updates**

Document that:
- relation-local `Block` still round-trips through DGL
- multi-relation `HeteroBlock` now has its own DGL interop path
- sampler-produced `HeteroBlock` layers can now be exported/imported through DGL

**Step 2: Run focused verification**

Run: `python -m pytest -q tests/compat/test_dgl_adapter.py tests/ops/test_block_ops.py tests/data/test_node_neighbor_sampler.py tests/data/test_link_neighbor_sampler.py`
Expected: PASS

**Step 3: Run full verification**

Run: `python -m pytest -q`
Expected: PASS

**Step 4: Commit**

```bash
git add README.md docs/core-concepts.md docs/quickstart.md
git commit -m "docs: describe hetero block dgl support"
```

**Step 5: Merge**

```bash
git checkout main
git merge --ff-only hetero-block-dgl
git push origin main
git branch -d hetero-block-dgl
git worktree remove .worktrees/hetero-block-dgl
```
