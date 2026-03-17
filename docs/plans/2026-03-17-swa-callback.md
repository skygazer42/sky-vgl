# SWA Callback Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a lightweight Stochastic Weight Averaging callback for trainer-level model averaging.

**Architecture:** Build SWA as a stateful callback on top of the existing epoch-end hook and callback resume infrastructure. Keep the API small with `start_epoch`, `frequency`, and `apply_on_fit_end`, averaging floating-point state tensors and optionally applying the averaged weights when training finishes.

**Tech Stack:** Python, PyTorch, VGL engine callbacks/trainer, pytest

---

### Task 1: Define SWA behavior with failing tests

**Files:**
- Modify: `tests/train/test_callbacks.py`
- Modify: `tests/test_package_exports.py`
- Modify: `vgl/engine/callbacks.py`
- Modify: `vgl/engine/__init__.py`
- Modify: `vgl/train/callbacks.py`
- Modify: `vgl/train/__init__.py`
- Modify: `vgl/__init__.py`

**Step 1: Write the failing tests**

Add tests that verify:
- invalid SWA arguments are rejected
- SWA averages model weights starting at the requested epoch and can apply them at fit end
- root package exports include `StochasticWeightAveraging`

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/train/test_callbacks.py tests/test_package_exports.py -q`
Expected: FAIL because the callback and exports do not exist yet.

**Step 3: Write minimal implementation**

Implement `StochasticWeightAveraging` as an epoch-end callback with:
- averaged floating-point state tracking
- resume-friendly `state_dict()` / `load_state_dict()`
- `apply_to()` and optional `apply_on_fit_end`

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/train/test_callbacks.py tests/test_package_exports.py -q`
Expected: PASS

### Task 2: Sync docs and broader verification

**Files:**
- Modify: `README.md`

**Step 1: Update README**

Mention SWA/model averaging in the training highlights.

**Step 2: Run targeted verification**

Run: `python -m pytest tests/train/test_callbacks.py tests/train/test_trainer_evaluation.py tests/train/test_checkpoints.py tests/test_package_exports.py -q`
Expected: PASS

### Task 3: Run full verification

**Files:**
- No further file changes required

**Step 1: Run full verification**

Run: `python -m pytest -q`
Expected: PASS

Run: `python -m ruff check vgl tests examples`
Expected: PASS
