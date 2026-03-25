# DGL Compatibility Expansion Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Expand `vgl.compat.dgl` from basic homogeneous conversion into a broader adapter that round-trips heterogeneous and temporal graphs while safely bridging new foundation metadata.

**Architecture:** Keep the public `Graph.from_dgl(...)` / `graph.to_dgl()` entrypoints unchanged. Preserve the current simple homogeneous `dgl.graph(...)` path for plain homogeneous graphs, but use `dgl.heterograph(...)` whenever relation names, typed node spaces, or temporal metadata need to survive the round-trip.

**Tech Stack:** Python 3.11, PyTorch, pytest

---

### Task 1: Add failing DGL hetero/temporal compatibility regressions

**Files:**
- Modify: `tests/compat/test_dgl_adapter.py`

**Step 1: Write the failing test**

Add regressions proving:

- a heterogeneous VGL graph round-trips through `to_dgl()` / `from_dgl()` with canonical edge types and typed node/edge features intact
- `Graph.from_dgl(...)` can import a single-relation external DGL heterograph without collapsing it to the homogeneous default edge type
- a temporal VGL graph round-trips through the DGL adapter while preserving both the timestamp edge feature and VGL's `time_attr` metadata
- storage-backed graphs with sparse caches still round-trip safely

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/compat/test_dgl_adapter.py`
Expected: FAIL because the adapter still only understands the simple homogeneous DGLGraph path.

**Step 3: Write minimal implementation**

No implementation yet in this task.

**Step 4: Run test to verify it still fails for the intended reason**

Run: `python -m pytest -q tests/compat/test_dgl_adapter.py`
Expected: FAIL on missing heterograph or temporal metadata handling.

**Step 5: Commit**

Do not commit yet.

### Task 2: Implement hetero/temporal DGL compatibility support

**Files:**
- Modify: `vgl/compat/dgl.py`
- Modify: `tests/compat/test_dgl_adapter.py`

**Step 1: Write the failing test**

Use the regressions from Task 1.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/compat/test_dgl_adapter.py`
Expected: FAIL

**Step 3: Write minimal implementation**

Teach `from_dgl(...)` to detect heterograph-style typed node and edge spaces, import per-type node/edge data, preserve canonical edge types, and restore temporal graphs through a lightweight exported metadata field such as `vgl_time_attr`. Teach `to_dgl(...)` to keep the existing plain homogeneous `dgl.graph(...)` path but export typed or temporal graphs through `dgl.heterograph(...)` so relation names and temporal metadata are not lost.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/compat/test_dgl_adapter.py`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 3: Refresh compatibility docs and run branch verification

**Files:**
- Modify: `README.md`
- Modify: `docs/migration-guide.md`
- Create: `docs/plans/2026-03-25-dgl-compat-expansion.md`
- Create: `docs/plans/2026-03-25-dgl-compat-expansion-design.md`

**Step 1: Write the failing test**

No code test. Review docs for places that describe DGL compatibility too vaguely for hetero/temporal behavior.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/compat/test_dgl_adapter.py`
Expected: PASS; documentation gap is manual.

**Step 3: Write minimal implementation**

Document that the DGL adapter now round-trips heterogeneous graphs and preserves temporal `time_attr` through exported adapter metadata.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q`
Expected: PASS across the repository.

**Step 5: Commit**

Run: `git add vgl tests docs README.md && git commit -m "feat: expand dgl graph compatibility"`
