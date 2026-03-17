# Temporal Attention Encoders Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add one compact temporal encoder batch by replacing the placeholder temporal module with stable time encoding and TGAT-style temporal attention blocks.

**Architecture:** Keep temporal modules in `vgl.nn.temporal` and preserve the current training-loop contract where models receive a `TemporalEventBatch` and construct causal history views via `batch.history_graph(i)`. `TimeEncoder`, `TGATLayer`, and `TGATEncoder` should work on temporal homogeneous graphs without introducing a separate sampler or memory subsystem.

**Tech Stack:** Python, PyTorch, pytest

---

## Phase 8 Scope

- Replace the placeholder `TemporalEncoder` with `TimeEncoder`, `TGATLayer`, and `TGATEncoder`.
- Export the new temporal modules from `vgl.nn` and `vgl`.
- Update the temporal event prediction example to use `TGATEncoder`.
- Fix the single-relation temporal graph convenience path so `graph.edge_index` / `graph.edata` work for non-`("node", "to", "node")` edge names.

## Source Alignment

- `TimeEncoder` should stay close to the stable cosine time encoding shape used in PyG's TGN implementation.
- `TGATLayer` / `TGATEncoder` should stay close to the public TGAT family by combining time encoding with local temporal attention.

This phase should remain VGL-sized and deliberately avoid a full memory-based temporal subsystem.

### Task 1: Lock The New Surface With Tests

**Files:**
- Create: `tests/nn/test_temporal.py`
- Modify: `tests/test_package_exports.py`
- Modify: `tests/core/test_graph_multi_type.py`

**Step 1: Write the failing tests**

- Add shape tests for `TimeEncoder`, `TGATLayer`, and `TGATEncoder`.
- Extend package export tests so the temporal modules are visible from `vgl`.
- Add a regression test proving single-relation temporal graphs expose `edge_index` and `edata` through the convenience properties.

**Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/core/test_graph_multi_type.py tests/nn/test_temporal.py tests/integration/test_end_to_end_temporal.py tests/test_package_exports.py -q
```

Expected: FAIL because the new temporal modules and supporting graph convenience path do not exist yet.

### Task 2: Implement Temporal Modules

**Files:**
- Replace: `vgl/nn/temporal.py`
- Modify: `vgl/nn/__init__.py`
- Modify: `vgl/__init__.py`

**Step 1: Add `TimeEncoder`**

- Use a simple learnable linear projection followed by cosine activation.

**Step 2: Add `TGATLayer`**

- Combine node features, edge timestamps, and a query time through temporal attention.

**Step 3: Add `TGATEncoder`**

- Stack `TGATLayer` blocks with residual, normalization, and feed-forward sublayers.

### Task 3: Fix Temporal Graph Convenience Access

**Files:**
- Modify: `vgl/graph/graph.py`

**Step 1: Align `Graph` with `GraphView` convenience behavior**

- When a graph has exactly one edge type, `graph.edge_index` and `graph.edata` should resolve to that unique edge store even if its relation name is not `"to"`.

### Task 4: Integrate With Example And Docs

**Files:**
- Modify: `examples/temporal/event_prediction.py`
- Modify: `README.md`

**Step 1: Update the temporal example**

- Replace the hand-built history-count feature path with `TGATEncoder` + `TimeEncoder`.

**Step 2: Update public docs**

- Mention temporal encoders in the README highlights and package description.

### Task 5: Verify

**Files:**
- No code changes expected

**Step 1: Run targeted verification**

```bash
python -m pytest tests/core/test_graph_multi_type.py tests/nn/test_temporal.py tests/integration/test_end_to_end_temporal.py tests/test_package_exports.py -q
python -m ruff check vgl tests examples
```

Expected: PASS

**Step 2: Run the full suite**

```bash
python -m pytest -q
python -m ruff check vgl tests examples
python examples/temporal/event_prediction.py
```

Expected: PASS
