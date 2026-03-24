# Sparse Runtime Expansion Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Expand `vgl.sparse` with public CSC conversion, transpose, column selection, and reduction primitives so the sparse backend better supports DGL-style low-level graph workflows.

**Architecture:** Keep `SparseTensor` as the single sparse container and extend the existing conversion/ops modules in place. New behavior should be layout-agnostic at the public API boundary, using conversions internally for correctness and returning deterministic sparse outputs with explicit tests.

**Tech Stack:** Python 3.11, PyTorch, pytest

---

### Task 1: Expose public CSC conversion

**Files:**
- Modify: `vgl/sparse/__init__.py`
- Modify: `tests/test_package_layout.py`
- Modify: `tests/sparse/test_sparse_convert.py`

**Step 1: Write the failing test**

Add assertions that `to_csc` is importable from `vgl.sparse`, present in `__all__`, and converts COO input into a valid CSC tensor with preserved values/order.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/test_package_layout.py -k foundation tests/sparse/test_sparse_convert.py`
Expected: FAIL because `to_csc` is not exported publicly.

**Step 3: Write minimal implementation**

Re-export `to_csc` from `vgl.sparse.__init__` and rename the current private helper in `vgl.sparse.convert` into a public function.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/test_package_layout.py -k foundation tests/sparse/test_sparse_convert.py`
Expected: PASS

**Step 5: Commit**

Run: `git add tests/test_package_layout.py tests/sparse/test_sparse_convert.py vgl/sparse/__init__.py vgl/sparse/convert.py && git commit -m "feat: expose csc sparse conversion"`

### Task 2: Add sparse transpose

**Files:**
- Modify: `vgl/sparse/ops.py`
- Modify: `vgl/sparse/__init__.py`
- Modify: `tests/sparse/test_sparse_ops.py`

**Step 1: Write the failing test**

Add tests covering transpose for COO, CSR, and CSC inputs. Verify output shape is swapped, coordinates are transposed, layout is preserved sensibly (`COO` stays `COO`, `CSR` becomes `CSC`, `CSC` becomes `CSR`), and values are carried through.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/sparse/test_sparse_ops.py -k transpose`
Expected: FAIL because `transpose` does not exist.

**Step 3: Write minimal implementation**

Add `transpose(sparse)` to `vgl.sparse.ops`, normalizing each layout directly instead of densifying. Export it from `vgl.sparse`.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/sparse/test_sparse_ops.py -k transpose`
Expected: PASS

**Step 5: Commit**

Run: `git add tests/sparse/test_sparse_ops.py vgl/sparse/ops.py vgl/sparse/__init__.py && git commit -m "feat: add sparse transpose op"`

### Task 3: Add column selection

**Files:**
- Modify: `vgl/sparse/ops.py`
- Modify: `vgl/sparse/__init__.py`
- Modify: `tests/sparse/test_sparse_ops.py`

**Step 1: Write the failing test**

Add tests for `select_cols(sparse, cols)` that mirror `select_rows`, including empty selections and value-preserving behavior.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/sparse/test_sparse_ops.py -k select_cols`
Expected: FAIL because `select_cols` does not exist.

**Step 3: Write minimal implementation**

Implement `select_cols` by reindexing the chosen column ids on COO data and returning a new sparse tensor with updated shape.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/sparse/test_sparse_ops.py -k select_cols`
Expected: PASS

**Step 5: Commit**

Run: `git add tests/sparse/test_sparse_ops.py vgl/sparse/ops.py vgl/sparse/__init__.py && git commit -m "feat: add sparse column selection"`

### Task 4: Add additive sparse reductions

**Files:**
- Modify: `vgl/sparse/ops.py`
- Modify: `vgl/sparse/__init__.py`
- Modify: `tests/sparse/test_sparse_ops.py`

**Step 1: Write the failing test**

Add tests for a reduction helper such as `sum(sparse, dim=...)` on unweighted and weighted sparse tensors, and verify invalid dims fail clearly.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/sparse/test_sparse_ops.py -k "sum or degree"`
Expected: FAIL because the reduction helper does not exist.

**Step 3: Write minimal implementation**

Implement additive reduction over rows/cols on COO-normalized data. Use stored `values` when present and `1` otherwise.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/sparse/test_sparse_ops.py -k "sum or degree"`
Expected: PASS

**Step 5: Commit**

Run: `git add tests/sparse/test_sparse_ops.py vgl/sparse/ops.py vgl/sparse/__init__.py && git commit -m "feat: add sparse reductions"`

### Task 5: Make degree value-aware via reductions

**Files:**
- Modify: `vgl/sparse/ops.py`
- Modify: `tests/sparse/test_sparse_ops.py`

**Step 1: Write the failing test**

Add tests proving `degree()` still counts unweighted structure but respects weighted sparse tensors when values are provided.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/sparse/test_sparse_ops.py -k degree`
Expected: FAIL because `degree` currently ignores `values`.

**Step 3: Write minimal implementation**

Route `degree()` through the new reduction helper so the semantics stay centralized.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/sparse/test_sparse_ops.py -k degree`
Expected: PASS

**Step 5: Commit**

Run: `git add tests/sparse/test_sparse_ops.py vgl/sparse/ops.py && git commit -m "feat: make sparse degree value aware"`

### Task 6: Refresh docs and examples for the expanded sparse surface

**Files:**
- Modify: `README.md`
- Modify: `docs/core-concepts.md`
- Modify: `docs/quickstart.md`

**Step 1: Write the failing test**

No code test. Review the sparse sections and identify missing references to CSC, transpose, and column selection.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/test_package_layout.py -k foundation tests/sparse/test_sparse_convert.py tests/sparse/test_sparse_ops.py`
Expected: PASS; documentation gap is manual, not enforced.

**Step 3: Write minimal implementation**

Update docs to mention the expanded sparse surface and when row-vs-column layout selection matters.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/test_package_layout.py -k foundation tests/sparse/test_sparse_convert.py tests/sparse/test_sparse_ops.py`
Expected: PASS

**Step 5: Commit**

Run: `git add README.md docs/core-concepts.md docs/quickstart.md && git commit -m "docs: describe expanded sparse runtime"`

### Task 7: Add integration coverage through graph/storage entry points

**Files:**
- Modify: `tests/core/test_graph_sparse_cache.py`
- Modify: `tests/storage/test_graph_store.py`
- Modify: touched modules as needed

**Step 1: Write the failing test**

Add integration assertions that graph/store adjacency requests can produce CSC layouts and that transpose/select operations compose correctly with storage-backed adjacency views.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/core/test_graph_sparse_cache.py tests/storage/test_graph_store.py`
Expected: FAIL if any graph/store path still hides the expanded sparse surface.

**Step 3: Write minimal implementation**

Patch graph/storage helpers only if needed so the new sparse APIs remain reachable and cached correctly.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/core/test_graph_sparse_cache.py tests/storage/test_graph_store.py`
Expected: PASS

**Step 5: Commit**

Run: `git add tests/core/test_graph_sparse_cache.py tests/storage/test_graph_store.py vgl/graph vgl/storage && git commit -m "test: cover sparse runtime through graph storage entrypoints"`

### Task 8: Run full regression for the sparse expansion branch

**Files:**
- Modify: any touched modules as needed

**Step 1: Write the failing test**

No new test file. Use the full repository suite as the branch gate.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q`
Expected: PASS if previous tasks were implemented cleanly; otherwise use failures to close remaining integration gaps.

**Step 3: Write minimal implementation**

Fix only the regressions exposed by the full suite.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q`
Expected: PASS across the repository.

**Step 5: Commit**

Run: `git add vgl tests docs README.md && git commit -m "feat: expand sparse runtime foundations"`
