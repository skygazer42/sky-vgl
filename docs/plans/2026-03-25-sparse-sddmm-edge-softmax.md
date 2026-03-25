# Sparse SDDMM And Edge Softmax Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Expand `vgl.sparse` with public `sddmm(...)` and `edge_softmax(...)` primitives, and route the current convolution helper through the sparse backend.

**Architecture:** Add two tensor-first sparse operators in `vgl.sparse.ops`, keep them layout-agnostic by reading indices through `to_coo()`, and preserve existing sparse storage order when returning results. Reuse the current `_homo.edge_softmax(...)` entry point as a thin wrapper so model code stays unchanged while the sparse backend becomes the source of truth.

**Tech Stack:** Python 3.11, PyTorch, pytest

---

### Task 1: Add failing sparse regression tests and export assertions

**Files:**
- Modify: `tests/sparse/test_sparse_ops.py`
- Modify: `tests/test_package_layout.py`

**Step 1: Write the failing test**

Add tests proving:

- `sddmm(...)` computes sampled dot products on a COO sparse structure
- `sddmm(...)` preserves compressed layout metadata and shape
- `edge_softmax(...)` normalizes scores across shared destination indices
- `edge_softmax(...)` handles an empty sparse tensor
- `edge_softmax(...)` rejects score-length mismatches
- `vgl.sparse.__all__` exports `sddmm` and `edge_softmax`

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/sparse/test_sparse_ops.py tests/test_package_layout.py`
Expected: FAIL because the new sparse operators and exports do not exist yet.

**Step 3: Write minimal implementation**

No implementation yet in this task.

**Step 4: Run test to verify it still fails for the intended reason**

Run: `python -m pytest -q tests/sparse/test_sparse_ops.py tests/test_package_layout.py`
Expected: FAIL on missing imports or missing sparse functions.

**Step 5: Commit**

Do not commit yet.

### Task 2: Implement public sparse sddmm and edge_softmax

**Files:**
- Modify: `vgl/sparse/ops.py`
- Modify: `vgl/sparse/__init__.py`
- Modify: `tests/sparse/test_sparse_ops.py`

**Step 1: Write the failing test**

Use the tests from Task 1.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/sparse/test_sparse_ops.py tests/test_package_layout.py`
Expected: FAIL

**Step 3: Write minimal implementation**

Implement:

- `sddmm(sparse, lhs, rhs)` returning a `SparseTensor` with identical structure and computed values
- `edge_softmax(sparse, scores, *, dim=1)` returning dense edge-aligned normalized weights

Implementation rules:

- accept COO, CSR, and CSC inputs
- preserve sparse layout, shape, and storage order
- make `sddmm(...)` compute a dot product across the last feature dimension
- allow higher-rank node features so head dimensions survive the reduction
- treat empty sparse tensors as valid no-op inputs
- reject invalid dimensions and length mismatches with clear `ValueError`s

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/sparse/test_sparse_ops.py tests/test_package_layout.py`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 3: Refactor the convolution helper to delegate to the sparse backend

**Files:**
- Modify: `vgl/nn/conv/_homo.py`

**Step 1: Write the failing test**

Use the existing sparse tests from Task 2 and rely on the repository’s existing convolution coverage for regression protection.

**Step 2: Run test to verify the current behavior baseline**

Run: `python -m pytest -q tests/sparse/test_sparse_ops.py`
Expected: PASS before refactoring `_homo.py`.

**Step 3: Write minimal implementation**

Update `_homo.edge_softmax(...)` to:

- build a COO `SparseTensor` from the provided `edge_index`
- infer a safe row dimension from the current edge set
- call `vgl.sparse.edge_softmax(...)`

Keep the public `_homo.edge_softmax(scores, edge_index, num_nodes)` signature unchanged so all current convolution modules continue to work without edits.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/sparse/test_sparse_ops.py tests/nn`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 4: Refresh docs and run full regression

**Files:**
- Modify: `README.md`
- Modify: `docs/core-concepts.md`
- Modify: `docs/quickstart.md`

**Step 1: Write the failing test**

No code test. Review sparse backend documentation for missing operator coverage.

**Step 2: Run test to verify the code path state**

Run: `python -m pytest -q tests/sparse/test_sparse_ops.py tests/test_package_layout.py tests/nn`
Expected: PASS; documentation gap is manual.

**Step 3: Write minimal implementation**

Document that `vgl.sparse` now exposes:

- sparse-dense matmul
- sampled dense-dense matmul
- edge softmax over sparse adjacency structure

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q`
Expected: PASS across the repository.

**Step 5: Commit**

Run: `git add vgl tests docs README.md && git commit -m "feat: add sparse sddmm and edge softmax"`
