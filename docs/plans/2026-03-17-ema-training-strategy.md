# EMA Training Strategy Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add an optimizer-step callback hook and an Exponential Moving Average strategy callback for stronger training-time weight smoothing.

**Architecture:** Extend the existing callback lifecycle with an `on_after_optimizer_step` hook so step-based strategies can integrate without modifying task or model APIs. Build `ExponentialMovingAverage` on top of that hook and export it through the engine, train, and root compatibility surfaces.

**Tech Stack:** Python, PyTorch, VGL engine callbacks/trainer, pytest

---

### Task 1: Define step-level callback and EMA behavior with failing tests

**Files:**
- Modify: `tests/train/test_callbacks.py`
- Modify: `tests/test_package_exports.py`
- Modify: `vgl/engine/callbacks.py`
- Modify: `vgl/engine/trainer.py`

**Step 1: Write the failing tests**

Add tests that verify:
- `on_after_optimizer_step` fires once per actual optimizer step, not per micro-batch
- `ExponentialMovingAverage` tracks shadow weights and can apply them at fit end
- root package exports include `ExponentialMovingAverage`

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/train/test_callbacks.py tests/test_package_exports.py -q`
Expected: FAIL because the hook and EMA callback do not exist yet.

**Step 3: Write minimal implementation**

Update the callback base class and trainer loop to emit optimizer-step events. Implement `ExponentialMovingAverage` as a callback that tracks floating-point state tensors and optionally applies them to the model at the end of training.

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/train/test_callbacks.py tests/test_package_exports.py -q`
Expected: PASS

### Task 2: Sync exports and full verification

**Files:**
- Modify: `vgl/engine/__init__.py`
- Modify: `vgl/train/__init__.py`
- Modify: `vgl/train/callbacks.py`
- Modify: `vgl/__init__.py`
- Modify: `README.md`

**Step 1: Export and document**

Re-export the new callback through engine/train/root packages and mention EMA in the training highlights.

**Step 2: Run targeted verification**

Run: `python -m pytest tests/train/test_callbacks.py tests/train/test_training_strategies.py tests/train/test_trainer_evaluation.py tests/test_package_exports.py -q`
Expected: PASS

**Step 3: Run full verification**

Run: `python -m pytest -q`
Expected: PASS

Run: `python -m ruff check vgl tests examples`
Expected: PASS
