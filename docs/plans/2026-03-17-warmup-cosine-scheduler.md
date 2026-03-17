# Warmup Cosine Scheduler Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a built-in warmup-plus-cosine learning-rate scheduler that works with the existing trainer scheduler hook and full checkpoint/resume flow.

**Architecture:** Implement a lightweight scheduler object in `vgl.engine` instead of changing trainer control flow. The scheduler will own its own state, apply the initial warmup LR at construction time, update optimizer param groups on each epoch-end `step()`, and serialize cleanly through the trainer checkpoint payload.

**Tech Stack:** Python, PyTorch, VGL engine trainer/schedulers, pytest

---

### Task 1: Define scheduler behavior with failing tests

**Files:**
- Modify: `tests/train/test_training_strategies.py`
- Modify: `tests/train/test_trainer_evaluation.py`
- Modify: `tests/test_package_exports.py`
- Modify: `tests/test_package_layout.py`

**Step 1: Write the failing tests**

Add tests that verify:
- invalid `warmup_epochs`, `max_epochs`, and `min_lr_ratio` values are rejected
- the scheduler applies the expected warmup-then-cosine LR sequence across epochs
- root and domain package exports include `WarmupCosineScheduler`
- paused training resumed from a full training checkpoint matches uninterrupted training when both use the built-in scheduler

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/train/test_training_strategies.py tests/train/test_trainer_evaluation.py tests/test_package_exports.py tests/test_package_layout.py -q`
Expected: FAIL because `WarmupCosineScheduler` does not exist yet.

### Task 2: Implement the scheduler and public exports

**Files:**
- Create: `vgl/engine/schedulers.py`
- Create: `vgl/train/schedulers.py`
- Modify: `vgl/engine/__init__.py`
- Modify: `vgl/train/__init__.py`
- Modify: `vgl/__init__.py`

**Step 1: Write minimal implementation**

Implement `WarmupCosineScheduler` with:
- constructor validation and optimizer/base-LR capture
- immediate initial LR application for the first warmup epoch
- epoch-end `step()` updates
- `state_dict()` / `load_state_dict()` and `get_last_lr()`

**Step 2: Run test to verify it passes**

Run: `python -m pytest tests/train/test_training_strategies.py tests/train/test_trainer_evaluation.py tests/test_package_exports.py tests/test_package_layout.py -q`
Expected: PASS

### Task 3: Sync docs and run broader verification

**Files:**
- Modify: `README.md`

**Step 1: Update README**

Mention the built-in warmup/cosine scheduler in the training highlights.

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
