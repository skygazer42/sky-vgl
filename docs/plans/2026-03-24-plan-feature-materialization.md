# Plan Feature Materialization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Materialize plan-fetched node and edge features into node-sampled subgraphs and add an opt-in `NodeNeighborSampler` fetch-stage path.

**Architecture:** Keep the current plan stage model intact. `expand_neighbors` will record induced edge ids, fetch stages will accumulate reserved materialization payloads in addition to their current `output_key`, and node-context materialization will overlay aligned fetched tensors onto the rebuilt subgraph. `NodeNeighborSampler` will gain opt-in node/edge feature stage emission without changing default behavior.

**Tech Stack:** Python 3.11, PyTorch, pytest

---

### Task 1: Add executor regressions for induced edge ids and typed fetch indices

**Files:**
- Modify: `tests/data/test_neighbor_expansion_stage.py`
- Modify: `tests/data/test_feature_fetch_stage.py`

**Step 1: Write the failing test**

Add regressions proving:

- homogeneous `expand_neighbors` stores induced `edge_ids` alongside `node_ids`
- heterogeneous `expand_neighbors` stores relation-local `edge_ids_by_type`
- `fetch_node_features` and `fetch_edge_features` can resolve their indices from typed dict state such as `node_ids_by_type` / `edge_ids_by_type`

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/data/test_neighbor_expansion_stage.py tests/data/test_feature_fetch_stage.py`
Expected: FAIL because executor expansion does not record edge ids and fetch stages only resolve tensor state entries.

**Step 3: Write minimal implementation**

No implementation yet in this task.

**Step 4: Run test to verify it still fails for the intended reason**

Run: `python -m pytest -q tests/data/test_neighbor_expansion_stage.py tests/data/test_feature_fetch_stage.py`
Expected: FAIL on missing edge-id state and typed index resolution.

**Step 5: Commit**

Do not commit yet.

### Task 2: Implement executor-side edge-id capture and fetched-feature tracking

**Files:**
- Modify: `vgl/dataloading/executor.py`
- Modify: `tests/data/test_neighbor_expansion_stage.py`
- Modify: `tests/data/test_feature_fetch_stage.py`

**Step 1: Write the failing test**

Use the tests from Task 1.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/data/test_neighbor_expansion_stage.py tests/data/test_feature_fetch_stage.py`
Expected: FAIL

**Step 3: Write minimal implementation**

Teach `expand_neighbors` to store induced `edge_ids` / `edge_ids_by_type`, teach feature-fetch stages to resolve typed dict indices, and accumulate fetched `TensorSlice`s under reserved node/edge materialization keys while preserving the existing `output_key` behavior.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/data/test_neighbor_expansion_stage.py tests/data/test_feature_fetch_stage.py`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 3: Materialize fetched features into node samples

**Files:**
- Modify: `vgl/dataloading/materialize.py`
- Modify: `tests/data/test_node_neighbor_sampler.py`

**Step 1: Write the failing test**

Add regressions proving a node-sampling plan can fetch replacement node/edge tensors from an external feature source and that the resulting `NodeBatch.graph` carries those fetched tensors in both homogeneous and hetero cases.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/data/test_node_neighbor_sampler.py -k feature`
Expected: FAIL because materialization ignores fetched feature payloads.

**Step 3: Write minimal implementation**

Overlay fetched node and edge features onto the rebuilt subgraph using aligned `n_id` / `e_id` order, while preserving existing graph-derived data when no fetched payload is present.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/data/test_node_neighbor_sampler.py -k feature`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 4: Add opt-in sampler fetch-stage emission

**Files:**
- Modify: `vgl/dataloading/sampler.py`
- Modify: `tests/data/test_node_neighbor_sampler.py`

**Step 1: Write the failing test**

Extend the new node-sampler feature regression so `NodeNeighborSampler(node_feature_names=..., edge_feature_names=...)` appends the right fetch stages automatically without requiring a custom subclassed sampler.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/data/test_node_neighbor_sampler.py -k feature`
Expected: FAIL because `NodeNeighborSampler` does not yet expose fetch-stage configuration.

**Step 3: Write minimal implementation**

Add opt-in node/edge feature configuration to `NodeNeighborSampler`, normalize homo vs hetero inputs, and append fetch stages after `expand_neighbors`. Keep the default `None` path unchanged.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/data/test_node_neighbor_sampler.py`
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

No code test. Review docs for places that still imply plan-backed feature fetch stops at executor state or requires custom sampler subclasses to reach public batches.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/data/test_neighbor_expansion_stage.py tests/data/test_feature_fetch_stage.py tests/data/test_node_neighbor_sampler.py`
Expected: PASS; documentation gap is manual.

**Step 3: Write minimal implementation**

Document that node-sampling plans can materialize fetched node/edge features into sampled subgraphs and that `NodeNeighborSampler` can append those fetch stages opt-in.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q`
Expected: PASS across the repository.

**Step 5: Commit**

Run: `git add vgl tests docs README.md && git commit -m "feat: materialize plan fetched node features"`
