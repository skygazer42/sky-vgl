# Step-Wise LR Scheduler Support Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend `Trainer` so learning-rate schedulers can step either once per epoch or once per optimizer step.

**Architecture:** Keep the scheduler surface small by adding a single `lr_scheduler_interval` trainer option with values `"epoch"` and `"step"`. Reuse the existing scheduler construction and checkpoint/resume flow, step the scheduler after each optimizer update when the interval is `"step"`, and reject monitor-driven schedulers such as `ReduceLROnPlateau` in step mode.

**Tech Stack:** Python, PyTorch, VGL engine trainer, pytest

---

### Task 1: Define step-wise scheduler behavior with failing tests

**Files:**
- Modify: `tests/train/test_training_strategies.py`
- Modify: `tests/train/test_trainer_evaluation.py`

**Step 1: Write the failing tests**

Add tests that verify:
- invalid `lr_scheduler_interval` is rejected
- step-wise schedulers update once per optimizer step, not once per batch
- monitor-driven schedulers are rejected in step mode
- paused training resumed from a full training checkpoint matches uninterrupted training when both use a step-wise scheduler

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/train/test_training_strategies.py tests/train/test_trainer_evaluation.py -q`
Expected: FAIL because `Trainer` does not support `lr_scheduler_interval="step"` yet.

### Task 2: Implement trainer support

**Files:**
- Modify: `vgl/engine/trainer.py`

**Step 1: Write minimal implementation**

Implement:
- `Trainer(..., lr_scheduler_interval="epoch")`
- validation for supported intervals
- step-wise scheduler stepping after each optimizer update
- rejection of `scheduler_monitor` / `ReduceLROnPlateau` in step mode

**Step 2: Run test to verify it passes**

Run: `python -m pytest tests/train/test_training_strategies.py tests/train/test_trainer_evaluation.py -q`
Expected: PASS

### Task 3: Sync docs and run broader verification

**Files:**
- Modify: `README.md`

**Step 1: Update README**

Mention step-wise scheduler support, including strategies such as `OneCycleLR`, in the training highlights.

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
