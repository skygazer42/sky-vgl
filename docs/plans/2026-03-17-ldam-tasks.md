# LDAM Task Support Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add built-in LDAM loss support for multiclass node, graph, and temporal-event tasks.

**Architecture:** Keep the trainer unchanged and extend multiclass task-level `loss` handling, since LDAM only adjusts target-class logits before the existing cross-entropy path. Reuse the shared multiclass loss helpers so class-count validation and margin construction stay consistent across task types.

**Tech Stack:** Python, PyTorch, VGL tasks, pytest

---

### Task 1: Define LDAM behavior with failing tests

**Files:**
- Modify: `tests/train/test_tasks.py`
- Modify: `tests/train/test_graph_classification_task.py`
- Modify: `tests/train/test_temporal_event_task.py`

**Step 1: Write the failing tests**

Add tests that verify:
- multiclass tasks accept `loss="ldam"` and compute the expected margin-adjusted loss
- invalid LDAM configuration is rejected

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/train/test_tasks.py tests/train/test_graph_classification_task.py tests/train/test_temporal_event_task.py -q`
Expected: FAIL because LDAM support does not exist yet.

### Task 2: Implement shared helper and task support

**Files:**
- Modify: `vgl/tasks/losses.py`
- Modify: `vgl/tasks/node_classification.py`
- Modify: `vgl/tasks/graph_classification.py`
- Modify: `vgl/tasks/temporal_event_prediction.py`

**Step 1: Write minimal implementation**

Implement:
- LDAM helper using class-count-derived margins
- task support for `loss="ldam"` plus `ldam_max_margin`
- explicit validation that LDAM requires `class_count`

**Step 2: Run test to verify it passes**

Run: `python -m pytest tests/train/test_tasks.py tests/train/test_graph_classification_task.py tests/train/test_temporal_event_task.py -q`
Expected: PASS

### Task 3: Sync docs and run full verification

**Files:**
- Modify: `README.md`

**Step 1: Update README**

Mention LDAM alongside the existing long-tail strategies.

**Step 2: Run verification**

Run: `python -m pytest -q`
Expected: PASS

Run: `python -m ruff check vgl tests examples`
Expected: PASS
