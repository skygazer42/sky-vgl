# Link and Temporal Feature Materialization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend plan-backed node and edge feature fetch so sampled link-prediction and temporal-event subgraphs expose the fetched tensors through the same public record and batch graphs users already consume.

**Architecture:** Keep the current sampling-plan model intact. `sample_link_neighbors` and `sample_temporal_neighbors` will record the sampled subgraph's global `n_id` / `e_id` state for later fetch stages, `LinkNeighborSampler` and `TemporalNeighborSampler` will grow the same opt-in feature-stage emission surface already shipped on `NodeNeighborSampler`, and link/temporal materialization will overlay fetched slices onto sampled record graphs before direct `sample(...)` results or loader-built batches are returned.

**Tech Stack:** Python 3.11, PyTorch, pytest

---

### Task 1: Add failing regressions for link and temporal feature materialization

**Files:**
- Modify: `tests/data/test_link_neighbor_sampler.py`
- Modify: `tests/data/test_temporal_neighbor_sampler.py`

**Step 1: Write the failing test**

Add regressions proving:

- `LinkNeighborSampler(node_feature_names=..., edge_feature_names=...)` can rehydrate sampled homogeneous and heterogeneous link subgraphs from a feature store
- `TemporalNeighborSampler(node_feature_names=..., edge_feature_names=...)` can rehydrate sampled strict-history subgraphs in both direct `sample(...)` and loader paths

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/data/test_link_neighbor_sampler.py tests/data/test_temporal_neighbor_sampler.py`
Expected: FAIL because the samplers do not yet expose feature-prefetch configuration.

**Step 3: Write minimal implementation**

No implementation yet in this task.

**Step 4: Run test to verify it still fails for the intended reason**

Run: `python -m pytest -q tests/data/test_link_neighbor_sampler.py tests/data/test_temporal_neighbor_sampler.py`
Expected: FAIL on missing sampler feature configuration and materialization support.

**Step 5: Commit**

Do not commit yet.

### Task 2: Teach sampling stages to expose sampled global ids for fetch stages

**Files:**
- Modify: `vgl/dataloading/executor.py`

**Step 1: Write the failing test**

Use the regressions from Task 1.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/data/test_link_neighbor_sampler.py tests/data/test_temporal_neighbor_sampler.py`
Expected: FAIL

**Step 3: Write minimal implementation**

Record `node_ids` / `edge_ids` or typed variants from the sampled link/temporal subgraph so later fetch stages can address the correct global ids without changing the stage protocol.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/data/test_link_neighbor_sampler.py tests/data/test_temporal_neighbor_sampler.py`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 3: Materialize fetched features into link and temporal record graphs

**Files:**
- Modify: `vgl/dataloading/materialize.py`

**Step 1: Write the failing test**

Use the regressions from Task 1.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/data/test_link_neighbor_sampler.py tests/data/test_temporal_neighbor_sampler.py`
Expected: FAIL

**Step 3: Write minimal implementation**

Overlay fetched node and edge features onto sampled link / temporal record graphs using the sampled graph's public `n_id` / `e_id` ordering, and reuse that same path for both direct `sample(...)` returns and `Loader` batch materialization.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/data/test_link_neighbor_sampler.py tests/data/test_temporal_neighbor_sampler.py`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 4: Add opt-in feature-stage emission on link and temporal samplers

**Files:**
- Modify: `vgl/dataloading/sampler.py`

**Step 1: Write the failing test**

Use the regressions from Task 1.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/data/test_link_neighbor_sampler.py tests/data/test_temporal_neighbor_sampler.py`
Expected: FAIL

**Step 3: Write minimal implementation**

Add `node_feature_names=...` and `edge_feature_names=...` to `LinkNeighborSampler` and `TemporalNeighborSampler`, normalize homogeneous vs heterogeneous configuration, and append the correct fetch stages after the sampling stage without changing default behavior.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/data/test_link_neighbor_sampler.py tests/data/test_temporal_neighbor_sampler.py tests/data/test_node_neighbor_sampler.py tests/data/test_batch_materialize.py tests/data/test_feature_fetch_stage.py tests/data/test_neighbor_expansion_stage.py`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 5: Refresh docs and run full regression

**Files:**
- Modify: `README.md`
- Modify: `docs/core-concepts.md`
- Modify: `docs/quickstart.md`

**Step 1: Write the failing test**

No code test. Review docs for places that still describe feature materialization as node-sampling-only.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/data/test_link_neighbor_sampler.py tests/data/test_temporal_neighbor_sampler.py tests/data/test_node_neighbor_sampler.py tests/data/test_batch_materialize.py tests/data/test_feature_fetch_stage.py tests/data/test_neighbor_expansion_stage.py`
Expected: PASS; documentation gap is manual.

**Step 3: Write minimal implementation**

Document that link and temporal sampled subgraphs can also opt into plan-backed node/edge feature materialization and reuse a retained graph feature store automatically.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q`
Expected: PASS across the repository.

**Step 5: Commit**

Run: `git add vgl tests docs README.md && git commit -m "feat: materialize plan fetched link and temporal features"`
