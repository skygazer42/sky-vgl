# Storage-Backed Plan Context Routing Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Let storage-backed graphs carry their feature source so plan-backed loading and direct plan execution can discover it automatically when no explicit `feature_store=` is provided.

**Architecture:** Keep the current explicit `feature_store=` override path intact and add a graph-level default set by `Graph.from_storage(...)`. `Loader` and `PlanExecutor` will both resolve the effective feature source from either the explicit argument or the plan graph's retained context.

**Tech Stack:** Python 3.11, PyTorch, pytest

---

### Task 1: Add graph-level storage context regression coverage

**Files:**
- Modify: `tests/core/test_feature_backed_graph.py`

**Step 1: Write the failing test**

Add a regression proving `Graph.from_storage(...)` retains the originating `FeatureStore` on the graph object so later runtime layers can reuse it.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/core/test_feature_backed_graph.py -k feature_store`
Expected: FAIL because storage-backed graphs do not currently expose retained feature-source context.

**Step 3: Write minimal implementation**

No implementation yet in this task.

**Step 4: Run test to verify it still fails for the intended reason**

Run: `python -m pytest -q tests/core/test_feature_backed_graph.py -k feature_store`
Expected: FAIL on missing graph attribute or mismatched context retention.

**Step 5: Commit**

Do not commit yet.

### Task 2: Implement graph storage context retention

**Files:**
- Modify: `vgl/graph/graph.py`
- Modify: `tests/core/test_feature_backed_graph.py`

**Step 1: Write the failing test**

Use the test from Task 1.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/core/test_feature_backed_graph.py -k feature_store`
Expected: FAIL

**Step 3: Write minimal implementation**

Add an optional graph-level feature-source field, set it in `Graph.from_storage(...)`, and preserve it across additive graph rebuild helpers where needed.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/core/test_feature_backed_graph.py`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 3: Add plan-backed loader and executor fallback regressions

**Files:**
- Modify: `tests/data/test_loader.py`
- Modify: `tests/integration/test_foundation_large_graph_flow.py`

**Step 1: Write the failing test**

Add one loader regression proving a plan-backed load forwards a storage-backed graph's retained feature source when `Loader(..., feature_store=None)` is used, and one integration regression proving a storage-backed graph can satisfy a `fetch_node_features` stage through the public loader path without manual `feature_store=` wiring.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/data/test_loader.py -k storage_context tests/integration/test_foundation_large_graph_flow.py -k plan_feature`
Expected: FAIL because automatic graph-context resolution does not exist yet.

**Step 3: Write minimal implementation**

No implementation yet in this task.

**Step 4: Run test to verify it still fails for the intended reason**

Run: `python -m pytest -q tests/data/test_loader.py -k storage_context tests/integration/test_foundation_large_graph_flow.py -k plan_feature`
Expected: FAIL because loader and executor still require manual wiring.

**Step 5: Commit**

Do not commit yet.

### Task 4: Implement loader and executor storage-context fallback

**Files:**
- Modify: `vgl/dataloading/loader.py`
- Modify: `vgl/dataloading/executor.py`
- Modify: `tests/data/test_loader.py`
- Modify: `tests/integration/test_foundation_large_graph_flow.py`

**Step 1: Write the failing test**

Use the tests from Task 3.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/data/test_loader.py -k storage_context tests/integration/test_foundation_large_graph_flow.py -k plan_feature`
Expected: FAIL

**Step 3: Write minimal implementation**

Teach `Loader` to resolve the effective feature source from the explicit `feature_store=` argument or `SamplingPlan.graph.feature_store`. Teach `PlanExecutor.execute(...)` to do the same fallback when `feature_store` is omitted. Keep explicit arguments higher priority than graph-retained context.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/data/test_loader.py tests/integration/test_foundation_large_graph_flow.py`
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

No code test. Review the storage-backed graph and plan-executor docs for places that still imply automatic feature-source routing requires manual `feature_store=` plumbing.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/core/test_feature_backed_graph.py tests/data/test_loader.py tests/integration/test_foundation_large_graph_flow.py`
Expected: PASS; documentation gap is manual.

**Step 3: Write minimal implementation**

Document that `Graph.from_storage(...)` retains a feature source and that plan-backed loading/execution can reuse it automatically when no explicit override is supplied.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q`
Expected: PASS across the repository.

**Step 5: Commit**

Run: `git add vgl tests docs README.md && git commit -m "feat: route plan context through storage-backed graphs"`
