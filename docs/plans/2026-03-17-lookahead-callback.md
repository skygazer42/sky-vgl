# Lookahead Callback Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a lightweight Lookahead training callback that periodically synchronizes fast model weights into slow weights and supports full checkpoint/resume.

**Architecture:** Build Lookahead as a stateful optimizer-step callback on top of the existing `on_after_optimizer_step` lifecycle hook and callback checkpoint machinery. Keep the API minimal with `sync_period`, `slow_step_size`, and optional `reset_state_on_fit_start`, tracking floating-point slow weights and synchronizing them back into the model at fixed optimizer-step intervals.

**Tech Stack:** Python, PyTorch, VGL engine callbacks/trainer, pytest

---

### Task 1: Define Lookahead behavior with failing tests

**Files:**
- Modify: `tests/train/test_callbacks.py`
- Modify: `tests/train/test_trainer_evaluation.py`
- Modify: `tests/test_package_exports.py`

**Step 1: Write the failing tests**

Add tests that verify:
- invalid `sync_period` and `slow_step_size` arguments are rejected
- Lookahead updates slow weights every `sync_period` optimizer steps and writes synchronized weights back into the model
- Lookahead `state_dict()` / `load_state_dict()` preserve tracked slow state and step count
- root package exports include `Lookahead`
- paused training resumed from a full training checkpoint matches uninterrupted training with Lookahead enabled

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/train/test_callbacks.py tests/train/test_trainer_evaluation.py tests/test_package_exports.py -q`
Expected: FAIL because `Lookahead` does not exist yet.

### Task 2: Implement the callback and public exports

**Files:**
- Modify: `vgl/engine/callbacks.py`
- Modify: `vgl/engine/__init__.py`
- Modify: `vgl/train/callbacks.py`
- Modify: `vgl/train/__init__.py`
- Modify: `vgl/__init__.py`

**Step 1: Write minimal implementation**

Implement `Lookahead` with:
- constructor validation for `sync_period` and `slow_step_size`
- floating-point slow state initialization from model state
- periodic sync on `on_after_optimizer_step`
- `apply_to()` helper
- resume-friendly `state_dict()` / `load_state_dict()`

**Step 2: Run test to verify it passes**

Run: `python -m pytest tests/train/test_callbacks.py tests/train/test_trainer_evaluation.py tests/test_package_exports.py -q`
Expected: PASS

### Task 3: Sync docs and run broader verification

**Files:**
- Modify: `README.md`

**Step 1: Update README**

Mention Lookahead alongside the existing training strategies in the training highlights.

**Step 2: Run targeted verification**

Run: `python -m pytest tests/train/test_callbacks.py tests/train/test_trainer_evaluation.py tests/train/test_checkpoints.py tests/test_package_exports.py -q`
Expected: PASS

### Task 4: Run full verification

**Files:**
- No further file changes required

**Step 1: Run full verification**

Run: `python -m pytest -q`
Expected: PASS

Run: `python -m ruff check vgl tests examples`
Expected: PASS
