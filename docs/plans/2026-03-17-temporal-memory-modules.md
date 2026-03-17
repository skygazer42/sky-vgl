# Temporal Memory Modules Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a lightweight TGN-style temporal memory family that fits the current VGL temporal event workflow.

**Architecture:** Extend `vgl.nn.temporal` with a small stateful memory core built from message functions, message aggregators, and a GRU-based updater. Keep the API independent from heavyweight sampler/state infrastructure so it can be used directly with `TemporalEventBatch` and examples.

**Tech Stack:** Python, PyTorch, VGL graph/batch abstractions, pytest

---

### Task 1: Define the public temporal memory surface

**Files:**
- Modify: `vgl/nn/temporal.py`
- Modify: `vgl/nn/__init__.py`
- Modify: `vgl/__init__.py`
- Test: `tests/nn/test_temporal.py`
- Test: `tests/test_package_exports.py`

**Step 1: Write the failing tests**

Add tests that import and exercise:
- `IdentityTemporalMessage`
- `LastMessageAggregator`
- `MeanMessageAggregator`
- `TGNMemory`

Cover:
- message module output shape
- aggregator semantics on repeated node ids
- memory state update / reset / untouched-node behavior
- root-package exports

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/nn/test_temporal.py tests/test_package_exports.py -q`
Expected: FAIL because the new classes are not defined/exported yet.

**Step 3: Write minimal implementation**

Implement:
- a TGN-style identity message function using source memory, destination memory, raw message, and time encoding
- last/mean message aggregators returning node ids, aggregated messages, and timestamps
- a stateful `TGNMemory` with `reset_state()`, `detach()`, `forward()`, and `update()`

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/nn/test_temporal.py tests/test_package_exports.py -q`
Expected: PASS

### Task 2: Wire temporal event features through batching and add an example

**Files:**
- Modify: `vgl/graph/batch.py`
- Modify: `tests/core/test_temporal_event_batch.py`
- Modify: `tests/data/test_temporal_event_loader.py`
- Create: `examples/temporal/memory_event_prediction.py`
- Create: `tests/integration/test_end_to_end_temporal_memory.py`

**Step 1: Write the failing tests**

Add tests that verify:
- `TemporalEventBatch.from_records()` preserves stacked `event_features`
- the temporal memory example runs end-to-end for one epoch

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/core/test_temporal_event_batch.py tests/data/test_temporal_event_loader.py tests/integration/test_end_to_end_temporal_memory.py -q`
Expected: FAIL because `event_features` are not collated yet and the example does not exist.

**Step 3: Write minimal implementation**

Update `TemporalEventBatch` to stack optional per-event features when present. Add a compact sequential event-prediction example that reads memory before each event, predicts the label, then updates memory with the observed interaction.

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/core/test_temporal_event_batch.py tests/data/test_temporal_event_loader.py tests/integration/test_end_to_end_temporal_memory.py -q`
Expected: PASS

### Task 3: Sync docs and perform full verification

**Files:**
- Modify: `README.md`

**Step 1: Update docs**

Mention the new temporal memory family and the new example in the README temporal section.

**Step 2: Run full verification**

Run: `python -m pytest -q`
Expected: PASS

Run: `python -m ruff check vgl tests examples`
Expected: PASS

Run: `python examples/temporal/event_prediction.py`
Expected: 1 epoch completes successfully

Run: `python examples/temporal/memory_event_prediction.py`
Expected: 1 epoch completes successfully
