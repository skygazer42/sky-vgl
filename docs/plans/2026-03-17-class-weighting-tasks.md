# Class Weighting Task Support Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add built-in class weighting support for multiclass tasks and positive-class weighting support for link prediction.

**Architecture:** Keep the trainer unchanged and extend task-level `loss` handling, because supervised weighting belongs with the task's label semantics. Reuse the shared focal loss helpers so `class_weight` and `pos_weight` work for both standard and focal losses without duplicating formulas.

**Tech Stack:** Python, PyTorch, VGL tasks, pytest

---

### Task 1: Define weighting behavior with failing tests

**Files:**
- Modify: `tests/train/test_tasks.py`
- Modify: `tests/train/test_graph_classification_task.py`
- Modify: `tests/train/test_temporal_event_task.py`
- Modify: `tests/train/test_link_prediction_task.py`

**Step 1: Write the failing tests**

Add tests that verify:
- multiclass tasks apply `class_weight` for cross-entropy and focal loss
- link prediction applies `pos_weight` for BCE and focal loss
- invalid weighting configuration is rejected

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/train/test_tasks.py tests/train/test_graph_classification_task.py tests/train/test_temporal_event_task.py tests/train/test_link_prediction_task.py -q`
Expected: FAIL because weighting support does not exist yet.

### Task 2: Implement shared helper and task support

**Files:**
- Modify: `vgl/tasks/losses.py`
- Modify: `vgl/tasks/node_classification.py`
- Modify: `vgl/tasks/graph_classification.py`
- Modify: `vgl/tasks/temporal_event_prediction.py`
- Modify: `vgl/tasks/link_prediction.py`

**Step 1: Write minimal implementation**

Implement:
- class-weight and pos-weight tensor normalization helpers
- `class_weight` support in multiclass cross-entropy and focal loss paths
- `pos_weight` support in link-prediction BCE and focal BCE paths

**Step 2: Run test to verify it passes**

Run: `python -m pytest tests/train/test_tasks.py tests/train/test_graph_classification_task.py tests/train/test_temporal_event_task.py tests/train/test_link_prediction_task.py -q`
Expected: PASS

### Task 3: Sync docs and run full verification

**Files:**
- Modify: `README.md`

**Step 1: Update README**

Mention class weighting / positive weighting alongside the existing training strategies.

**Step 2: Run verification**

Run: `python -m pytest -q`
Expected: PASS

Run: `python -m ruff check vgl tests examples`
Expected: PASS
