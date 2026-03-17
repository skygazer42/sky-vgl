# SAM Optimizer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a built-in Sharpness-Aware Minimization optimizer and trainer support for SAM-style two-pass optimization.

**Architecture:** Implement `SAM` as an optimizer wrapper in `vgl.engine` that owns perturbation state and delegates the actual update to a base optimizer. Extend `Trainer` with a minimal sharpness-aware branch that replays each optimizer-step batch group for the second forward/backward pass, while preserving the existing history, scheduler, callback, and checkpoint/resume semantics.

**Tech Stack:** Python, PyTorch, VGL engine trainer/optimizers, pytest

---

### Task 1: Define SAM behavior with failing tests

**Files:**
- Modify: `tests/train/test_training_strategies.py`
- Modify: `tests/train/test_trainer_evaluation.py`
- Modify: `tests/test_package_exports.py`
- Modify: `tests/test_package_layout.py`

**Step 1: Write the failing tests**

Add tests that verify:
- invalid `SAM` configuration is rejected
- trainer can optimize with `SAM` and performs the expected sharpness-aware update
- root and domain package exports include `SAM`
- paused training resumed from a full training checkpoint matches uninterrupted training with `SAM`

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/train/test_training_strategies.py tests/train/test_trainer_evaluation.py tests/test_package_exports.py tests/test_package_layout.py -q`
Expected: FAIL because `SAM` and trainer support do not exist yet.

### Task 2: Implement the optimizer and trainer support

**Files:**
- Create: `vgl/engine/optimizers.py`
- Create: `vgl/train/optimizers.py`
- Modify: `vgl/engine/trainer.py`
- Modify: `vgl/engine/__init__.py`
- Modify: `vgl/train/__init__.py`
- Modify: `vgl/__init__.py`

**Step 1: Write minimal implementation**

Implement:
- `SAM(params, base_optimizer, rho=0.05, adaptive=False, **kwargs)`
- optimizer `first_step()` / `second_step()` / `state_dict()` / `load_state_dict()`
- trainer-side detection and two-pass optimizer-step replay for each accumulation group
- scheduler and callback dispatch once per completed SAM optimizer step

**Step 2: Run test to verify it passes**

Run: `python -m pytest tests/train/test_training_strategies.py tests/train/test_trainer_evaluation.py tests/test_package_exports.py tests/test_package_layout.py -q`
Expected: PASS

### Task 3: Sync docs and run broader verification

**Files:**
- Modify: `README.md`

**Step 1: Update README**

Mention SAM alongside the existing training strategies in the training highlights.

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
