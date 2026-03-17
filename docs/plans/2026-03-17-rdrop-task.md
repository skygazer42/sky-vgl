# R-Drop Task Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a built-in `RDropTask` wrapper that applies R-Drop consistency regularization to supported classification tasks.

**Architecture:** Implement `RDropTask` as a thin task wrapper in `vgl.tasks` that delegates targets and metrics to an underlying task while exposing a `paired_loss()` API for training. Extend `Trainer` with a minimal paired-forward branch that performs two forwards only when the configured task supports paired loss, while preserving existing metrics, scheduler, callback, and checkpoint semantics.

**Tech Stack:** Python, PyTorch, VGL tasks/trainer, pytest

---

### Task 1: Define R-Drop behavior with failing tests

**Files:**
- Modify: `tests/train/test_tasks.py`
- Modify: `tests/train/test_training_strategies.py`
- Modify: `tests/test_package_exports.py`
- Modify: `tests/test_package_layout.py`

**Step 1: Write the failing tests**

Add tests that verify:
- invalid `RDropTask` configuration is rejected
- `RDropTask` computes averaged supervised loss plus symmetric KL regularization
- trainer uses two forwards during training and one forward during evaluation for `RDropTask`
- root and domain package exports include `RDropTask`

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/train/test_tasks.py tests/train/test_training_strategies.py tests/test_package_exports.py tests/test_package_layout.py -q`
Expected: FAIL because `RDropTask` and trainer paired-forward support do not exist yet.

### Task 2: Implement the wrapper and trainer support

**Files:**
- Create: `vgl/tasks/rdrop.py`
- Modify: `vgl/tasks/__init__.py`
- Modify: `vgl/train/tasks.py`
- Modify: `vgl/engine/trainer.py`
- Modify: `vgl/__init__.py`

**Step 1: Write minimal implementation**

Implement:
- `RDropTask(base_task, alpha=1.0)` for `cross_entropy` tasks
- `paired_loss()` with symmetric KL regularization over task-filtered logits
- trainer-side paired-forward detection and double-forward loss path during training only

**Step 2: Run test to verify it passes**

Run: `python -m pytest tests/train/test_tasks.py tests/train/test_training_strategies.py tests/test_package_exports.py tests/test_package_layout.py -q`
Expected: PASS

### Task 3: Sync docs and run full verification

**Files:**
- Modify: `README.md`

**Step 1: Update README**

Mention `R-Drop` alongside the existing training strategies.

**Step 2: Run verification**

Run: `python -m pytest -q`
Expected: PASS

Run: `python -m ruff check vgl tests examples`
Expected: PASS
