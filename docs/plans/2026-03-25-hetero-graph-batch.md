# Heterogeneous GraphBatch Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend `GraphBatch` so many-graph graph-classification flows can batch heterogeneous graphs without abandoning the current `batch.graphs` API.

**Architecture:** Preserve the existing homogeneous `graph_index` / `graph_ptr` contract, but add typed membership tensors for heterogeneous graphs via `graph_index_by_type` and `graph_ptr_by_type`. Keep `Loader` and `Trainer` unchanged by teaching `GraphBatch.from_graphs(...)`, `GraphBatch.from_samples(...)`, `to()`, and `pin_memory()` to handle the richer shape.

**Tech Stack:** Python 3.11, PyTorch, pytest

---

### Task 1: Add failing hetero GraphBatch contract tests

**Files:**
- Modify: `tests/core/test_graph_batch.py`
- Modify: `tests/core/test_graph_batch_graph_classification.py`
- Modify: `tests/core/test_batch_transfer.py`
- Modify: `tests/integration/test_graph_classification_many_graphs.py`

**Step 1: Write the failing test**

Add regressions proving:

- `GraphBatch.from_graphs(...)` produces per-node-type membership tensors for heterogeneous graphs
- `GraphBatch.from_samples(...)` keeps metadata labels and typed graph pointers aligned
- `GraphBatch.to()` / `pin_memory()` move those typed membership tensors
- a heterogeneous many-graph classification flow can pool with `graph_index_by_type[...]` end to end

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/core/test_graph_batch.py tests/core/test_graph_batch_graph_classification.py tests/core/test_batch_transfer.py tests/integration/test_graph_classification_many_graphs.py`
Expected: FAIL because `GraphBatch` still assumes homogeneous `graph.x` batching and lacks typed membership fields.

**Step 3: Write minimal implementation**

No implementation yet in this task.

**Step 4: Run test to verify it still fails for the intended reason**

Run: `python -m pytest -q tests/core/test_graph_batch.py tests/core/test_graph_batch_graph_classification.py tests/core/test_batch_transfer.py tests/integration/test_graph_classification_many_graphs.py`
Expected: FAIL on missing hetero GraphBatch support.

**Step 5: Commit**

Do not commit yet.

### Task 2: Implement typed hetero membership in GraphBatch

**Files:**
- Modify: `vgl/graph/batch.py`

**Step 1: Write the failing test**

Use the regressions from Task 1.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/core/test_graph_batch.py tests/core/test_graph_batch_graph_classification.py tests/core/test_batch_transfer.py tests/integration/test_graph_classification_many_graphs.py`
Expected: FAIL

**Step 3: Write minimal implementation**

Keep homogeneous batches unchanged, but let hetero batches populate `graph_index_by_type` / `graph_ptr_by_type`, keep `graph_index` / `graph_ptr` unset in that case, and preserve transfer / pin-memory semantics for the new fields.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/core/test_graph_batch.py tests/core/test_graph_batch_graph_classification.py tests/core/test_batch_transfer.py tests/integration/test_graph_classification_many_graphs.py`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 3: Refresh public example and docs

**Files:**
- Modify: `examples/hetero/graph_classification.py`
- Modify: `README.md`
- Modify: `docs/core-concepts.md`
- Modify: `docs/quickstart.md`

**Step 1: Write the failing test**

No code test. Review the current hetero graph-classification example and docs for places that still imply only homogeneous membership indexing exists.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/core/test_graph_batch.py tests/core/test_graph_batch_graph_classification.py tests/core/test_batch_transfer.py tests/integration/test_graph_classification_many_graphs.py`
Expected: PASS; documentation/example gap is manual.

**Step 3: Write minimal implementation**

Document and demonstrate that heterogeneous graph-classification models can keep using `batch.graphs` while pooling with `graph_index_by_type[...]`.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q`
Expected: PASS across the repository.

**Step 5: Commit**

Run: `git add vgl tests docs README.md examples && git commit -m "feat: support hetero graph batches"`
