# Focal Loss Task Support Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add built-in focal loss support across VGL's supervised node, graph, temporal-event, and link-prediction tasks.

**Architecture:** Keep the trainer unchanged and extend task-level `loss` handling, since the existing task abstraction already owns supervised loss computation. Add shared focal loss helpers under `vgl.tasks` so cross-entropy and binary-cross-entropy tasks can reuse one implementation with consistent validation.

**Tech Stack:** Python, PyTorch, VGL tasks, pytest

---

### Task 1: Define focal loss behavior with failing tests

**Files:**
- Modify: `tests/train/test_tasks.py`
- Modify: `tests/train/test_graph_classification_task.py`
- Modify: `tests/train/test_temporal_event_task.py`
- Modify: `tests/train/test_link_prediction_task.py`

**Step 1: Write the failing tests**

Add tests that verify:
- classification tasks accept `loss="focal"` and compute the expected focal loss
- link prediction accepts `loss="focal"` and computes the expected binary focal loss
- invalid focal configuration is rejected

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/train/test_tasks.py tests/train/test_graph_classification_task.py tests/train/test_temporal_event_task.py tests/train/test_link_prediction_task.py -q`
Expected: FAIL because focal loss support does not exist yet.

### Task 2: Implement shared helpers and task support

**Files:**
- Create: `vgl/tasks/losses.py`
- Modify: `vgl/tasks/node_classification.py`
- Modify: `vgl/tasks/graph_classification.py`
- Modify: `vgl/tasks/temporal_event_prediction.py`
- Modify: `vgl/tasks/link_prediction.py`

**Step 1: Write minimal implementation**

Implement:
- focal cross-entropy helper with `gamma >= 0`
- focal BCE-with-logits helper with `gamma >= 0`
- task support for `loss="focal"` plus `focal_gamma`

**Step 2: Run test to verify it passes**

Run: `python -m pytest tests/train/test_tasks.py tests/train/test_graph_classification_task.py tests/train/test_temporal_event_task.py tests/train/test_link_prediction_task.py -q`
Expected: PASS

### Task 3: Sync docs and run full verification

**Files:**
- Modify: `README.md`

**Step 1: Update README**

Mention focal loss alongside the existing training strategies.

**Step 2: Run verification**

Run: `python -m pytest -q`
Expected: PASS

Run: `python -m ruff check vgl tests examples`
Expected: PASS
