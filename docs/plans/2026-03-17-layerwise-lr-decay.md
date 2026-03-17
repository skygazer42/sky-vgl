# Layerwise LR Decay Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a built-in layer-wise learning-rate decay strategy and a minimal trainer hook for custom optimizer parameter groups.

**Architecture:** Extend `Trainer` with an `optimizer_param_groups` input that can supply custom parameter groups without changing the training loop. Implement a stateless `LayerwiseLrDecay` helper that maps named module-prefix groups to optimizer param groups with progressively decayed learning rates, so it composes cleanly with existing callbacks and checkpoint/resume.

**Tech Stack:** Python, PyTorch, VGL engine trainer/parameter-groups, pytest

---

### Task 1: Define parameter-group behavior with failing tests

**Files:**
- Modify: `tests/train/test_training_strategies.py`
- Modify: `tests/train/test_trainer_evaluation.py`
- Modify: `tests/test_package_exports.py`
- Modify: `tests/test_package_layout.py`

**Step 1: Write the failing tests**

Add tests that verify:
- invalid `LayerwiseLrDecay` configuration is rejected
- `Trainer` can build optimizer param groups from `LayerwiseLrDecay` and assigns the expected per-group learning rates
- paused training resumed from a full training checkpoint matches uninterrupted training when both use custom optimizer param groups
- root and domain package exports include `LayerwiseLrDecay`

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/train/test_training_strategies.py tests/train/test_trainer_evaluation.py tests/test_package_exports.py tests/test_package_layout.py -q`
Expected: FAIL because `LayerwiseLrDecay` and `optimizer_param_groups` support do not exist yet.

### Task 2: Implement the helper, trainer hook, and exports

**Files:**
- Create: `vgl/engine/parameter_groups.py`
- Create: `vgl/train/parameter_groups.py`
- Modify: `vgl/engine/trainer.py`
- Modify: `vgl/engine/__init__.py`
- Modify: `vgl/train/__init__.py`
- Modify: `vgl/__init__.py`

**Step 1: Write minimal implementation**

Implement:
- `Trainer(..., optimizer_param_groups=None)` with callable/iterable support
- `LayerwiseLrDecay(module_name_groups, lr_decay=0.5, include_rest=True)`
- deterministic named-parameter grouping, overlap validation, and per-group LR assignment

**Step 2: Run test to verify it passes**

Run: `python -m pytest tests/train/test_training_strategies.py tests/train/test_trainer_evaluation.py tests/test_package_exports.py tests/test_package_layout.py -q`
Expected: PASS

### Task 3: Sync docs and run broader verification

**Files:**
- Modify: `README.md`

**Step 1: Update README**

Mention layer-wise learning-rate strategies alongside the existing scheduler/callback training surface.

**Step 2: Run targeted verification**

Run: `python -m pytest tests/train/test_training_strategies.py tests/train/test_trainer_evaluation.py tests/train/test_checkpoints.py tests/test_package_exports.py tests/test_package_layout.py -q`
Expected: PASS

### Task 4: Run full verification

**Files:**
- No further file changes required

**Step 1: Run full verification**

Run: `python -m pytest -q`
Expected: PASS

Run: `python -m ruff check vgl tests examples`
Expected: PASS
