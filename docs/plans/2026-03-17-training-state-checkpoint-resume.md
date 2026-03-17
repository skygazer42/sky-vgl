# Training State Checkpoint Resume Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add full training-state checkpoint persistence and resume support for `Trainer`.

**Architecture:** Extend the checkpoint payload format with optional training sections for optimizer, scheduler, scaler, callbacks, trainer state, and history while preserving legacy model-only behavior. Add instance-level save/restore methods on `Trainer` and restore callback state after `on_fit_start` so step-based callbacks such as EMA can resume cleanly.

**Tech Stack:** Python, PyTorch, VGL engine/checkpoints/trainer/callbacks, pytest

---

### Task 1: Define checkpoint payload and resume behavior with failing tests

**Files:**
- Modify: `tests/train/test_checkpoints.py`
- Modify: `tests/train/test_trainer_evaluation.py`
- Modify: `vgl/engine/checkpoints.py`
- Modify: `vgl/engine/trainer.py`
- Modify: `vgl/engine/callbacks.py`
- Modify: `vgl/engine/history.py`

**Step 1: Write the failing tests**

Add tests that verify:
- structured checkpoints preserve optional training-state sections
- `Trainer.save_training_checkpoint()` writes optimizer / scheduler / callback / history state
- `Trainer.restore_training_checkpoint()` lets a paused run resume and match an uninterrupted run

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/train/test_checkpoints.py tests/train/test_trainer_evaluation.py -q`
Expected: FAIL because the richer payload and resume methods do not exist yet.

**Step 3: Write minimal implementation**

Implement:
- optional checkpoint sections in the checkpoint builder / normalizer
- callback `state_dict()` / `load_state_dict()` hooks
- `TrainingHistory.state_dict()` and `TrainingHistory.from_state_dict()`
- `Trainer.save_training_checkpoint()` and `Trainer.restore_training_checkpoint()`
- fit-loop resume logic that restores history, global step, best state, and callback state

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/train/test_checkpoints.py tests/train/test_trainer_evaluation.py -q`
Expected: PASS

### Task 2: Sync docs and run broader verification

**Files:**
- Modify: `README.md`

**Step 1: Update README**

Mention full training-state checkpoint/resume support in the training highlights.

**Step 2: Run targeted verification**

Run: `python -m pytest tests/train/test_checkpoints.py tests/train/test_trainer_evaluation.py tests/train/test_callbacks.py tests/train/test_training_strategies.py tests/train/test_mixed_precision.py -q`
Expected: PASS

### Task 3: Run full verification

**Files:**
- No further file changes required

**Step 1: Run full verification**

Run: `python -m pytest -q`
Expected: PASS

Run: `python -m ruff check vgl tests examples`
Expected: PASS
