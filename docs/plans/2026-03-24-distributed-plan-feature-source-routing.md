# Distributed Plan Feature Source Routing Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Let plan-backed feature fetch stages resolve against either storage-backed feature stores or coordinator-backed distributed feature sources, and make `Loader` forward that source into plan execution.

**Architecture:** Keep the existing `fetch_node_features` / `fetch_edge_features` stage names and extend `PlanExecutor` with a tiny feature-source dispatch layer. Add an optional `feature_store` handle to `Loader` so plan execution can use a coordinator or feature store without changing current sampler APIs.

**Tech Stack:** Python 3.11, PyTorch, pytest

---

### Task 1: Add coordinator-backed feature fetch regression coverage

**Files:**
- Modify: `tests/data/test_feature_fetch_stage.py`

**Step 1: Write the failing test**

Add a regression proving `PlanExecutor` can execute `fetch_node_features` and `fetch_edge_features` stages when `feature_store=` is a `LocalSamplingCoordinator` built from partition shards, not a plain `FeatureStore`.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/data/test_feature_fetch_stage.py -k coordinator`
Expected: FAIL because the executor only calls `.fetch(...)` today.

**Step 3: Write minimal implementation**

No implementation yet in this task.

**Step 4: Run test to verify it still fails for the intended reason**

Run: `python -m pytest -q tests/data/test_feature_fetch_stage.py -k coordinator`
Expected: FAIL with an attribute error or equivalent call-shape failure on the coordinator source.

**Step 5: Commit**

Do not commit yet.

### Task 2: Implement executor feature-source dispatch

**Files:**
- Modify: `vgl/dataloading/executor.py`
- Modify: `tests/data/test_feature_fetch_stage.py`

**Step 1: Write the failing test**

Use the test from Task 1.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/data/test_feature_fetch_stage.py -k coordinator`
Expected: FAIL

**Step 3: Write minimal implementation**

Teach `_fetch_features(...)` to dispatch by feature-source capability:
- `.fetch(...)` for store-style sources
- `.fetch_node_features(...)` for node stages
- `.fetch_edge_features(...)` for edge stages

Preserve the current output shape in `context.state[output_key]`.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/data/test_feature_fetch_stage.py`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 3: Add loader forwarding regression coverage

**Files:**
- Modify: `tests/data/test_loader.py`

**Step 1: Write the failing test**

Add a regression proving `Loader(feature_store=source)` forwards that source into `executor.execute(...)` when a sampler returns a `SamplingPlan`. Use a tiny fake executor that records the forwarded argument.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/data/test_loader.py -k feature_store`
Expected: FAIL because `Loader` does not accept or forward `feature_store`.

**Step 3: Write minimal implementation**

No implementation yet in this task.

**Step 4: Run test to verify it still fails for the intended reason**

Run: `python -m pytest -q tests/data/test_loader.py -k feature_store`
Expected: FAIL on the missing constructor argument or missing forwarding behavior.

**Step 5: Commit**

Do not commit yet.

### Task 4: Implement loader feature-source forwarding

**Files:**
- Modify: `vgl/dataloading/loader.py`
- Modify: `tests/data/test_loader.py`

**Step 1: Write the failing test**

Use the test from Task 3.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/data/test_loader.py -k feature_store`
Expected: FAIL

**Step 3: Write minimal implementation**

Add an optional `feature_store=None` argument to `Loader`, store it, and pass it through in `_resolve_sampled(...)` / `_sample_item(...)` when executing a `SamplingPlan`. Keep all existing defaults and worker behavior unchanged.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/data/test_loader.py -k feature_store`
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

No code test. Review the data-loading and distributed docs for places that imply plan-backed feature fetch only works with a direct in-process store.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/data/test_feature_fetch_stage.py tests/data/test_loader.py tests/distributed/test_sampling_coordinator.py`
Expected: PASS; documentation gap is manual.

**Step 3: Write minimal implementation**

Document that plan-backed feature fetch stages can resolve against a direct feature store or a coordinator-backed distributed feature source, and that `Loader` can forward that source into plan execution.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q`
Expected: PASS across the repository.

**Step 5: Commit**

Run: `git add vgl tests docs README.md && git commit -m "feat: route plan feature fetch through distributed sources"`
