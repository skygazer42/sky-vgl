# Gradient Centralization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a built-in Gradient Centralization callback that centralizes eligible parameter gradients before each optimizer step.

**Architecture:** Implement `GradientCentralization` as a stateless engine callback that runs in `on_before_optimizer_step`, so it composes with gradient clipping, sharpness-aware optimizers, checkpoint/resume, and existing callback exports. Keep the first version small: configurable eligibility for all weight tensors or convolution-only tensors.

**Tech Stack:** Python, PyTorch, VGL engine callbacks, pytest

---

### Task 1: Define callback behavior with failing tests

**Files:**
- Modify: `tests/train/test_callbacks.py`
- Modify: `tests/test_package_exports.py`
- Modify: `tests/test_package_layout.py`

**Step 1: Write the failing tests**

Add tests that verify:
- invalid `GradientCentralization` configuration is rejected
- eligible gradients are centralized before the optimizer step
- convolution-only mode leaves matrix gradients unchanged
- root and domain package exports include `GradientCentralization`

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/train/test_callbacks.py tests/test_package_exports.py tests/test_package_layout.py -q`
Expected: FAIL because `GradientCentralization` does not exist yet.

### Task 2: Implement the callback and exports

**Files:**
- Modify: `vgl/engine/callbacks.py`
- Modify: `vgl/engine/__init__.py`
- Modify: `vgl/train/callbacks.py`
- Modify: `vgl/train/__init__.py`
- Modify: `vgl/__init__.py`

**Step 1: Write minimal implementation**

Implement:
- `GradientCentralization(conv_only=False)` callback
- per-parameter centralization for gradients with rank > 1, or rank > 3 when `conv_only=True`
- package exports through engine, legacy train, and root namespaces

**Step 2: Run test to verify it passes**

Run: `python -m pytest tests/train/test_callbacks.py tests/test_package_exports.py tests/test_package_layout.py -q`
Expected: PASS

### Task 3: Sync docs and run broader verification

**Files:**
- Modify: `README.md`

**Step 1: Update README**

Mention gradient centralization in the training highlights.

**Step 2: Run targeted verification**

Run: `python -m pytest tests/train/test_callbacks.py tests/train/test_trainer_evaluation.py tests/test_package_exports.py tests/test_package_layout.py -q`
Expected: PASS

### Task 4: Run full verification

**Files:**
- No further file changes required

**Step 1: Run full verification**

Run: `python -m pytest -q`
Expected: PASS

Run: `python -m ruff check vgl tests examples`
Expected: PASS
