# Adaptive Gradient Clipping Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add an Adaptive Gradient Clipping callback that clips gradients relative to parameter norms before each optimizer step.

**Architecture:** Extend the callback lifecycle with an `on_before_optimizer_step` hook so pre-step strategies can run on unscaled gradients. Build `AdaptiveGradientClipping` as a mostly stateless callback on top of that hook, and make trainer-side mixed precision unscale gradients before pre-step callbacks when required.

**Tech Stack:** Python, PyTorch, VGL engine callbacks/trainer, pytest

---

### Task 1: Define pre-step hook and AGC behavior with failing tests

**Files:**
- Modify: `tests/train/test_callbacks.py`
- Modify: `tests/train/test_trainer_evaluation.py`
- Modify: `tests/train/test_mixed_precision.py`
- Modify: `tests/test_package_exports.py`
- Modify: `tests/test_package_layout.py`

**Step 1: Write the failing tests**

Add tests that verify:
- trainer calls `on_before_optimizer_step` once per optimizer step
- invalid AGC arguments are rejected
- AGC clips updates relative to parameter norms before optimizer step
- mixed precision trainers unscale gradients before pre-step callbacks
- paused training resumed from a full training checkpoint matches uninterrupted training with AGC enabled
- root and domain package exports include `AdaptiveGradientClipping`

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/train/test_callbacks.py tests/train/test_trainer_evaluation.py tests/train/test_mixed_precision.py tests/test_package_exports.py tests/test_package_layout.py -q`
Expected: FAIL because the pre-step hook and AGC callback do not exist yet.

### Task 2: Implement the hook, callback, and exports

**Files:**
- Modify: `vgl/engine/callbacks.py`
- Modify: `vgl/engine/trainer.py`
- Modify: `vgl/engine/__init__.py`
- Modify: `vgl/train/callbacks.py`
- Modify: `vgl/train/__init__.py`
- Modify: `vgl/__init__.py`

**Step 1: Write minimal implementation**

Implement:
- `Callback.on_before_optimizer_step(...)`
- trainer-side pre-step callback dispatch on each optimizer step
- mixed-precision unscale before pre-step callbacks when required
- `AdaptiveGradientClipping(clipping=0.01, eps=1e-3)`

**Step 2: Run test to verify it passes**

Run: `python -m pytest tests/train/test_callbacks.py tests/train/test_trainer_evaluation.py tests/train/test_mixed_precision.py tests/test_package_exports.py tests/test_package_layout.py -q`
Expected: PASS

### Task 3: Sync docs and run broader verification

**Files:**
- Modify: `README.md`

**Step 1: Update README**

Mention AGC alongside the existing training callbacks in the training highlights.

**Step 2: Run targeted verification**

Run: `python -m pytest tests/train/test_callbacks.py tests/train/test_trainer_evaluation.py tests/train/test_mixed_precision.py tests/train/test_checkpoints.py tests/test_package_exports.py tests/test_package_layout.py -q`
Expected: PASS

### Task 4: Run full verification

**Files:**
- No further file changes required

**Step 1: Run full verification**

Run: `python -m pytest -q`
Expected: PASS

Run: `python -m ruff check vgl tests examples`
Expected: PASS
