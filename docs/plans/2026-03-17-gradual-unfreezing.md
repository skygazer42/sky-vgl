# Gradual Unfreezing Callback Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a built-in gradual unfreezing callback for fine-tuning pretrained graph models over multiple epochs.

**Architecture:** Implement gradual unfreezing as a stateful callback on top of the existing fit-start and epoch-end hooks. The callback will freeze configured module-prefix groups at fit start, unfreeze one group at a time on a configurable epoch schedule, and persist its state through the existing callback checkpoint/resume payload.

**Tech Stack:** Python, PyTorch, VGL engine callbacks/trainer, pytest

---

### Task 1: Define callback behavior with failing tests

**Files:**
- Modify: `tests/train/test_callbacks.py`
- Modify: `tests/train/test_trainer_evaluation.py`
- Modify: `tests/test_package_exports.py`
- Modify: `tests/test_package_layout.py`

**Step 1: Write the failing tests**

Add tests that verify:
- invalid callback configuration is rejected
- configured module groups are frozen at fit start and unfrozen on the expected epoch schedule
- callback state round-trips through `state_dict()` / `load_state_dict()`
- root and domain package exports include `GradualUnfreezing`
- paused training resumed from a full training checkpoint matches uninterrupted training with gradual unfreezing enabled

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/train/test_callbacks.py tests/train/test_trainer_evaluation.py tests/test_package_exports.py tests/test_package_layout.py -q`
Expected: FAIL because `GradualUnfreezing` does not exist yet.

### Task 2: Implement the callback and public exports

**Files:**
- Modify: `vgl/engine/callbacks.py`
- Modify: `vgl/engine/__init__.py`
- Modify: `vgl/train/callbacks.py`
- Modify: `vgl/train/__init__.py`
- Modify: `vgl/__init__.py`

**Step 1: Write minimal implementation**

Implement `GradualUnfreezing` with:
- module-prefix group normalization and validation
- fit-start freezing of tracked parameters
- epoch-end incremental unfreezing by `start_epoch` and `frequency`
- resume-friendly `state_dict()` / `load_state_dict()`

**Step 2: Run test to verify it passes**

Run: `python -m pytest tests/train/test_callbacks.py tests/train/test_trainer_evaluation.py tests/test_package_exports.py tests/test_package_layout.py -q`
Expected: PASS

### Task 3: Sync docs and run broader verification

**Files:**
- Modify: `README.md`

**Step 1: Update README**

Mention gradual unfreezing alongside the existing training callbacks in the training highlights.

**Step 2: Run targeted verification**

Run: `python -m pytest tests/train/test_callbacks.py tests/train/test_trainer_evaluation.py tests/train/test_checkpoints.py tests/test_package_exports.py tests/test_package_layout.py -q`
Expected: PASS

### Task 4: Run full verification

**Files:**
- No further file changes required

**Step 1: Run full verification**

Run: `python -m pytest -q`
Expected: PASS

Run: `python -m ruff check vgl tests examples`
Expected: PASS
