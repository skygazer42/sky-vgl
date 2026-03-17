# Mixed Precision Training Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add practical mixed-precision training support to `Trainer` with autocast and optional gradient scaling.

**Architecture:** Extend `vgl.engine.Trainer` with a small `precision` surface that controls autocast contexts and an optional `grad_scaler` integration for scaled backward/optimizer steps. Keep it independent from model/task code and make CPU tests deterministic by allowing injected scaler objects.

**Tech Stack:** Python, PyTorch, VGL engine/trainer, pytest

---

### Task 1: Define mixed precision behavior with failing tests

**Files:**
- Create: `tests/train/test_mixed_precision.py`
- Modify: `vgl/engine/trainer.py`

**Step 1: Write the failing tests**

Add tests covering:
- invalid `precision` values are rejected
- `precision="bf16-mixed"` enters `torch.autocast` with CPU + `torch.bfloat16`
- injected `grad_scaler` is used for scale / backward / unscale / step / update

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/train/test_mixed_precision.py -q`
Expected: FAIL because the new Trainer precision arguments and scaler behavior do not exist yet.

**Step 3: Write minimal implementation**

Update `Trainer` to:
- validate `precision`
- infer autocast dtype/device from model parameters
- wrap forward/loss in autocast for mixed precision modes
- route backward and optimizer stepping through a scaler when one is configured

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/train/test_mixed_precision.py -q`
Expected: PASS

### Task 2: Sync docs and broader verification

**Files:**
- Modify: `README.md`

**Step 1: Update README**

Mention mixed precision in the training highlights.

**Step 2: Run targeted verification**

Run: `python -m pytest tests/train/test_mixed_precision.py tests/train/test_training_strategies.py tests/train/test_callbacks.py tests/train/test_trainer_evaluation.py -q`
Expected: PASS

### Task 3: Run full verification

**Files:**
- No further file changes required

**Step 1: Run full verification**

Run: `python -m pytest -q`
Expected: PASS

Run: `python -m ruff check vgl tests examples`
Expected: PASS
