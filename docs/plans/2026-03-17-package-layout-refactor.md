# Package Layout Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reorganize the library around graph-centric domain packages while preserving the existing public API.

**Architecture:** Move core implementation entry points into explicit domain packages such as `vgl.graph`, `vgl.dataloading`, `vgl.engine`, `vgl.tasks`, `vgl.metrics`, and `vgl.transforms`. Keep `vgl.core`, `vgl.data`, and `vgl.train` as compatibility layers that re-export the new implementations so existing imports continue to work.

**Tech Stack:** Python 3.11+, PyTorch, pytest

---

### Task 1: Lock The Refactor With Tests

**Files:**
- Create: `tests/test_package_layout.py`
- Modify: `tests/data/test_loader.py`

**Step 1: Write the failing test**

Add import coverage for the new package layout and a dataset-protocol test proving the loader can iterate over generic sequence-style datasets.

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_package_layout.py tests/data/test_loader.py -q`
Expected: FAIL because the new packages do not exist yet and `Loader` still depends on `dataset.graphs`.

**Step 3: Write minimal implementation**

Create the new packages and update the loader to use `__len__`/`__getitem__` or standard iteration instead of reaching into `dataset.graphs`.

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_package_layout.py tests/data/test_loader.py -q`
Expected: PASS

### Task 2: Introduce The New Domain Packages

**Files:**
- Create: `vgl/graph/__init__.py`
- Create: `vgl/graph/batch.py`
- Create: `vgl/graph/errors.py`
- Create: `vgl/graph/graph.py`
- Create: `vgl/graph/schema.py`
- Create: `vgl/graph/stores.py`
- Create: `vgl/graph/view.py`
- Create: `vgl/dataloading/__init__.py`
- Create: `vgl/dataloading/dataset.py`
- Create: `vgl/dataloading/loader.py`
- Create: `vgl/dataloading/records.py`
- Create: `vgl/dataloading/sampler.py`
- Create: `vgl/engine/__init__.py`
- Create: `vgl/engine/evaluator.py`
- Create: `vgl/engine/trainer.py`
- Create: `vgl/tasks/__init__.py`
- Create: `vgl/tasks/base.py`
- Create: `vgl/tasks/node_classification.py`
- Create: `vgl/tasks/graph_classification.py`
- Create: `vgl/tasks/link_prediction.py`
- Create: `vgl/tasks/temporal_event_prediction.py`
- Create: `vgl/metrics/__init__.py`
- Create: `vgl/metrics/base.py`
- Create: `vgl/metrics/classification.py`
- Create: `vgl/transforms/__init__.py`
- Create: `vgl/transforms/identity.py`
- Modify: `vgl/__init__.py`
- Modify: `vgl/compat/dgl.py`
- Modify: `vgl/compat/pyg.py`

**Step 1: Write the minimal package shims**

Move implementation ownership to the new packages and keep names stable by re-exporting the same classes and functions.

**Step 2: Keep the root API stable**

Expose the new packages from `vgl/__init__.py`, add `DataLoader` as the preferred loader name, and preserve `Loader` as a compatibility alias.

**Step 3: Run focused tests**

Run: `python -m pytest tests/test_package_layout.py tests/test_package_exports.py -q`
Expected: PASS

### Task 3: Convert Legacy Packages Into Compatibility Layers

**Files:**
- Modify: `vgl/core/__init__.py`
- Modify: `vgl/core/batch.py`
- Modify: `vgl/core/errors.py`
- Modify: `vgl/core/graph.py`
- Modify: `vgl/core/schema.py`
- Modify: `vgl/core/stores.py`
- Modify: `vgl/core/view.py`
- Modify: `vgl/data/__init__.py`
- Modify: `vgl/data/dataset.py`
- Modify: `vgl/data/loader.py`
- Modify: `vgl/data/sample.py`
- Modify: `vgl/data/sampler.py`
- Modify: `vgl/data/transform.py`
- Modify: `vgl/train/__init__.py`
- Modify: `vgl/train/evaluator.py`
- Modify: `vgl/train/metrics.py`
- Modify: `vgl/train/task.py`
- Modify: `vgl/train/tasks.py`
- Modify: `vgl/train/trainer.py`

**Step 1: Replace legacy module bodies with imports from the new packages**

Keep the old module paths importable without duplicating behavior.

**Step 2: Update internal imports**

Switch the implementation modules to import through the new layout so future work stops depending on `core`, `data`, and `train`.

**Step 3: Run compatibility tests**

Run: `python -m pytest tests/core tests/data tests/train -q`
Expected: PASS

### Task 4: Update User-Facing Examples And Documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/quickstart.md`
- Modify: `examples/homo/graph_classification.py`
- Modify: `examples/hetero/graph_classification.py`
- Modify: `examples/temporal/event_prediction.py`
- Modify: any other example still importing from legacy paths

**Step 1: Update examples to prefer the new layout**

Use `vgl.dataloading`, `vgl.tasks`, and `vgl.engine` in examples and docs while keeping the root import examples simple.

**Step 2: Run smoke checks**

Run: `python -m pytest tests/integration -q`
Expected: PASS

### Task 5: Full Verification

**Files:**
- No code changes expected

**Step 1: Run the full test suite**

Run: `python -m pytest -q`
Expected: PASS

**Step 2: Optional static checks if environment permits**

Run: `python -m ruff check .`
Run: `python -m mypy vgl`

**Step 3: Commit**

```bash
git add docs/plans/2026-03-17-package-layout-refactor.md tests vgl README.md docs/quickstart.md examples
git commit -m "refactor: reorganize package layout around domain modules"
```
