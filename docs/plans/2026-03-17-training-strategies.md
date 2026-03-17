# Training Strategies Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add practical Trainer-level optimization strategies so VGL training loops support stronger real-world recipes without changing model code.

**Architecture:** Extend `vgl.engine.Trainer` with three lightweight, orthogonal controls: gradient accumulation, gradient clipping, and learning-rate schedulers. Keep the surface small by integrating them directly into the existing trainer loop instead of introducing a heavyweight strategy system.

**Tech Stack:** Python, PyTorch, VGL engine/trainer abstractions, pytest

---

### Task 1: Define Trainer strategy behavior with failing tests

**Files:**
- Create: `tests/train/test_training_strategies.py`
- Modify: `vgl/engine/trainer.py`

**Step 1: Write the failing tests**

Add tests covering:
- `accumulate_grad_batches` delays optimizer stepping and handles remainder batches
- `gradient_clip_val` clips gradients before optimizer step
- `lr_scheduler` steps each epoch for standard schedulers
- `ReduceLROnPlateau` receives the monitored metric automatically
- invalid accumulation values are rejected

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/train/test_training_strategies.py -q`
Expected: FAIL because the new Trainer arguments and behaviors do not exist yet.

**Step 3: Write minimal implementation**

Update `Trainer` to:
- validate new arguments
- accumulate scaled losses across micro-batches
- clip gradient norms before optimizer step when requested
- construct/accept LR schedulers and step them after each epoch

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/train/test_training_strategies.py -q`
Expected: PASS

### Task 2: Sync docs and broader trainer coverage

**Files:**
- Modify: `README.md`
- Modify: `tests/train/test_trainer_evaluation.py`

**Step 1: Update README**

Mention that `Trainer` supports gradient accumulation, gradient clipping, and LR scheduling.

**Step 2: Extend trainer tests if needed**

Add one assertion in existing trainer coverage if broader surface verification becomes useful.

**Step 3: Run targeted trainer verification**

Run: `python -m pytest tests/train/test_training_strategies.py tests/train/test_trainer_evaluation.py tests/train/test_callbacks.py -q`
Expected: PASS

### Task 3: Run full verification

**Files:**
- No further file changes required

**Step 1: Run full verification**

Run: `python -m pytest -q`
Expected: PASS

Run: `python -m ruff check vgl tests examples`
Expected: PASS
