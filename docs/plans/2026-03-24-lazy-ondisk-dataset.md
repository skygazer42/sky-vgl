# Lazy On-Disk Dataset Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `OnDiskGraphDataset` load graph payloads lazily per item, expose manifest-backed split views, and keep older `graphs.pt` datasets readable.

**Architecture:** Newly written datasets will store one serialized graph payload per file under `graphs/`, while the dataset object tracks graph file paths and deserializes only the requested item on access. Split views will be lightweight contiguous-range wrappers over the same backing entries, and legacy `graphs.pt` artifacts will stay supported through a fallback path.

**Tech Stack:** Python 3.11, PyTorch, pytest

---

### Task 1: Add lazy on-disk layout regression coverage

**Files:**
- Modify: `tests/data/test_ondisk_dataset.py`

**Step 1: Write the failing test**

Add a test proving `OnDiskGraphDataset.write(...)` creates a per-graph `graphs/` directory and that indexing the dataset still round-trips the expected graph data.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/data/test_ondisk_dataset.py -k layout`
Expected: FAIL because the current implementation only writes `graphs.pt`.

**Step 3: Write minimal implementation**

No implementation yet in this task.

**Step 4: Run test to verify it still fails for the intended reason**

Run: `python -m pytest -q tests/data/test_ondisk_dataset.py -k layout`
Expected: FAIL on missing `graphs/` payload files.

**Step 5: Commit**

Do not commit yet.

### Task 2: Implement lazy per-graph storage and item loading

**Files:**
- Modify: `vgl/data/ondisk.py`
- Modify: `tests/data/test_ondisk_dataset.py`

**Step 1: Write the failing test**

Use the test from Task 1.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/data/test_ondisk_dataset.py -k layout`
Expected: FAIL

**Step 3: Write minimal implementation**

Write graph payloads to deterministic per-graph files, track those file entries inside `OnDiskGraphDataset`, and load/deserialise one graph per `__getitem__` call.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/data/test_ondisk_dataset.py -k layout`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 3: Add split-view and legacy-compatibility regression coverage

**Files:**
- Modify: `tests/data/test_ondisk_dataset.py`

**Step 1: Write the failing test**

Add one test proving `dataset.split("train")` and `dataset.split("test")` return the expected contiguous views, and one test proving a legacy `graphs.pt` dataset still loads correctly.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/data/test_ondisk_dataset.py -k "split or legacy"`
Expected: FAIL because split views do not exist and any new implementation may not preserve legacy loading.

**Step 3: Write minimal implementation**

No implementation yet in this task.

**Step 4: Run test to verify it still fails for the intended reason**

Run: `python -m pytest -q tests/data/test_ondisk_dataset.py -k "split or legacy"`
Expected: FAIL on missing split support and/or missing legacy fallback.

**Step 5: Commit**

Do not commit yet.

### Task 4: Implement split views and legacy fallback

**Files:**
- Modify: `vgl/data/ondisk.py`
- Modify: `tests/data/test_ondisk_dataset.py`
- Modify: `tests/integration/test_foundation_ondisk_sampling.py`

**Step 1: Write the failing test**

Use the tests from Task 3.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/data/test_ondisk_dataset.py -k "split or legacy" tests/integration/test_foundation_ondisk_sampling.py`
Expected: FAIL

**Step 3: Write minimal implementation**

Add `split(name)` support through lightweight contiguous-range dataset views and keep legacy `graphs.pt` artifacts readable when the new `graphs/` layout is absent. Update the integration path only if it helps exercise the new split-view flow.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/data/test_ondisk_dataset.py tests/integration/test_foundation_ondisk_sampling.py`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 5: Refresh docs and run full regression

**Files:**
- Modify: `README.md`
- Modify: `docs/core-concepts.md`
- Modify: `docs/quickstart.md`
- Modify: any touched modules as needed

**Step 1: Write the failing test**

No code test. Review docs for places that still describe `OnDiskGraphDataset` as an eager serialized graph list.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/data/test_ondisk_dataset.py tests/integration/test_foundation_ondisk_sampling.py`
Expected: PASS; documentation gap is manual.

**Step 3: Write minimal implementation**

Document that `OnDiskGraphDataset` now stores graphs as per-item payloads, loads them lazily, and exposes manifest-backed split views.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q`
Expected: PASS across the repository.

**Step 5: Commit**

Run: `git add vgl tests docs README.md && git commit -m "feat: add lazy on-disk dataset loading"`
