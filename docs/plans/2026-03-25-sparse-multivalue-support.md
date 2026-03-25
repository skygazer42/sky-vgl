# Sparse Multi-Value Support Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Expand `vgl.sparse` so `SparseTensor.values` can carry trailing payload dimensions and the existing sparse ops preserve or validate those payloads correctly.

**Architecture:** Widen the sparse container contract from scalar edge values to `(nnz, ...)` payloads, keep layout conversion and structural ops payload-preserving, extend reductions and `sddmm(...)` over the edge axis, and keep `spmm(...)` intentionally scalar-only with a clear validation error.

**Tech Stack:** Python 3.11, PyTorch, pytest

---

### Task 1: Add failing sparse regressions for multi-value payloads

**Files:**
- Modify: `tests/sparse/test_sparse_base.py`
- Modify: `tests/sparse/test_sparse_convert.py`
- Modify: `tests/sparse/test_sparse_ops.py`

**Step 1: Write the failing test**

Add tests proving:

- `SparseTensor` accepts values shaped `(nnz, heads)`
- `SparseTensor` rejects values whose leading dimension does not match `nnz`
- COO/CSR/CSC conversions preserve trailing payload dimensions and ordering
- `select_rows(...)`, `select_cols(...)`, and `transpose(...)` preserve multi-value payloads
- `sum(...)` and `degree(...)` reduce over edges but keep trailing payload dimensions
- `sddmm(...)` returns `(nnz, heads)` values for multi-head node features
- `edge_softmax(...)` normalizes independently per trailing payload slice
- `spmm(...)` rejects multi-dimensional sparse values with a clear `ValueError`

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/sparse/test_sparse_base.py tests/sparse/test_sparse_convert.py tests/sparse/test_sparse_ops.py`
Expected: FAIL because the sparse container and ops still assume scalar-only edge values.

**Step 3: Write minimal implementation**

No implementation yet in this task.

**Step 4: Run test to verify it still fails for the intended reason**

Run: `python -m pytest -q tests/sparse/test_sparse_base.py tests/sparse/test_sparse_convert.py tests/sparse/test_sparse_ops.py`
Expected: FAIL on the current scalar-only sparse behavior.

**Step 5: Commit**

Do not commit yet.

### Task 2: Implement multi-value sparse container and op support

**Files:**
- Modify: `vgl/sparse/base.py`
- Modify: `vgl/sparse/convert.py`
- Modify: `vgl/sparse/ops.py`

**Step 1: Write the failing test**

Use the tests from Task 1.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/sparse/test_sparse_base.py tests/sparse/test_sparse_convert.py tests/sparse/test_sparse_ops.py`
Expected: FAIL

**Step 3: Write minimal implementation**

Implement:

- `SparseTensor.values` validation based on `values.shape[0] == nnz`
- payload-preserving conversions and structural selectors
- reduction over the sparse axis while preserving trailing payload dimensions
- multi-head `sddmm(...)` that reduces only the final feature dimension
- explicit `spmm(...)` validation rejecting non-scalar sparse values
- explicit multi-dimensional `edge_softmax(...)` support

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/sparse/test_sparse_base.py tests/sparse/test_sparse_convert.py tests/sparse/test_sparse_ops.py`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 3: Refresh sparse docs and run full regression

**Files:**
- Modify: `README.md`
- Modify: `docs/core-concepts.md`
- Modify: `docs/quickstart.md`

**Step 1: Write the failing test**

No code test. Review sparse backend docs for scalar-only wording or missing payload support notes.

**Step 2: Run test to verify the code path state**

Run: `python -m pytest -q tests/sparse/test_sparse_base.py tests/sparse/test_sparse_convert.py tests/sparse/test_sparse_ops.py`
Expected: PASS

**Step 3: Write minimal implementation**

Document that `vgl.sparse` now supports sparse edge payloads with trailing dimensions for conversions, reductions, `sddmm(...)`, and `edge_softmax(...)`, while `spmm(...)` remains scalar-weighted in this batch.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q`
Expected: PASS across the repository.

**Step 5: Commit**

Run: `git add vgl tests docs README.md && git commit -m "feat: support multi-value sparse payloads"`
