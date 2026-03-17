# Balanced Softmax Task Support Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add built-in balanced softmax loss support for multiclass node, graph, and temporal-event tasks.

**Architecture:** Keep the trainer unchanged and extend multiclass task-level `loss` handling, since balanced softmax only changes how logits are normalized before cross-entropy. Reuse shared task loss helpers so class-count validation and the balanced-logit transform stay consistent across task types.

**Tech Stack:** Python, PyTorch, VGL tasks, pytest

---

### Task 1: Define balanced softmax behavior with failing tests

**Files:**
- Modify: `tests/train/test_tasks.py`
- Modify: `tests/train/test_graph_classification_task.py`
- Modify: `tests/train/test_temporal_event_task.py`

**Step 1: Write the failing tests**

Add tests that verify:
- multiclass tasks accept `loss="balanced_softmax"` and compute the expected balanced softmax loss
- invalid class-count configuration is rejected

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/train/test_tasks.py tests/train/test_graph_classification_task.py tests/train/test_temporal_event_task.py -q`
Expected: FAIL because balanced softmax support does not exist yet.

### Task 2: Implement shared helper and task support

**Files:**
- Modify: `vgl/tasks/losses.py`
- Modify: `vgl/tasks/node_classification.py`
- Modify: `vgl/tasks/graph_classification.py`
- Modify: `vgl/tasks/temporal_event_prediction.py`

**Step 1: Write minimal implementation**

Implement:
- class-count normalization helper
- balanced softmax helper as `cross_entropy(logits + log(class_count), targets, ...)`
- task support for `loss="balanced_softmax"` plus `class_count`

**Step 2: Run test to verify it passes**

Run: `python -m pytest tests/train/test_tasks.py tests/train/test_graph_classification_task.py tests/train/test_temporal_event_task.py -q`
Expected: PASS

### Task 3: Sync docs and run full verification

**Files:**
- Modify: `README.md`

**Step 1: Update README**

Mention balanced softmax alongside the existing long-tail strategies.

**Step 2: Run verification**

Run: `python -m pytest -q`
Expected: PASS

Run: `python -m ruff check vgl tests examples`
Expected: PASS
