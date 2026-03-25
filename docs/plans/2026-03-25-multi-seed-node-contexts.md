# Multi-Seed Node Contexts Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend the node sampling/materialization path so one node context can legally carry multiple supervision seeds while still producing the existing flat `NodeBatch.seed_index` contract.

**Architecture:** Keep `NodeBatch`, `NodeClassificationTask`, and trainer behavior unchanged. Normalize multi-seed metadata into `NodeSeedRequest.node_ids`, materialize one sampled subgraph per context, then expand that context into multiple `SampleRecord` entries that share the same sampled graph but carry one local `subgraph_seed` each.

**Tech Stack:** Python 3.11, PyTorch, pytest

---

### Task 1: Add failing multi-seed node materialization regressions

**Files:**
- Modify: `tests/data/test_batch_materialize.py`
- Modify: `tests/data/test_node_neighbor_sampler.py`

**Step 1: Write the failing test**

Add regressions proving:

- `materialize_batch(...)` expands one homogeneous node context with multiple seeds into one `NodeBatch` with multiple `seed_index` entries
- the same flattening works for heterogeneous node contexts keyed by `node_type`
- `NodeNeighborSampler.sample(...)` accepts multi-seed node metadata and returns multiple samples that share the sampled subgraph
- `Loader(..., sampler=NodeNeighborSampler(...))` flattens one multi-seed dataset item into multiple node-supervision entries without duplicating the sampled graph

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/data/test_batch_materialize.py tests/data/test_node_neighbor_sampler.py -k multi_seed`
Expected: FAIL because node materialization still assumes exactly one seed per context and the sampler still coerces `metadata['seed']` to one scalar.

**Step 3: Write minimal implementation**

No implementation yet in this task.

**Step 4: Run test to verify it still fails for the intended reason**

Run: `python -m pytest -q tests/data/test_batch_materialize.py tests/data/test_node_neighbor_sampler.py -k multi_seed`
Expected: FAIL on the one-seed-only guard or scalar seed coercion path.

**Step 5: Commit**

Do not commit yet.

### Task 2: Implement multi-seed node sampler/materialization support

**Files:**
- Modify: `vgl/dataloading/materialize.py`
- Modify: `vgl/dataloading/sampler.py`

**Step 1: Write the failing test**

Use the regressions from Task 1.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/data/test_batch_materialize.py tests/data/test_node_neighbor_sampler.py -k multi_seed`
Expected: FAIL

**Step 3: Write minimal implementation**

Teach `NodeNeighborSampler` to normalize scalar or rank-1 multi-seed metadata into `NodeSeedRequest.node_ids`, keep node-type validation on every seed, and let node materialization expand one context into multiple `SampleRecord` values that share the sampled graph while carrying per-seed metadata and local `subgraph_seed` values.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/data/test_batch_materialize.py tests/data/test_node_neighbor_sampler.py -k multi_seed`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 3: Refresh docs and run branch verification

**Files:**
- Modify: `README.md`
- Modify: `docs/core-concepts.md`
- Modify: `docs/quickstart.md`
- Create: `docs/plans/2026-03-25-multi-seed-node-contexts.md`
- Create: `docs/plans/2026-03-25-multi-seed-node-contexts-design.md`

**Step 1: Write the failing test**

No code test. Review docs for places that only demonstrate scalar node seeds.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/data/test_batch_materialize.py tests/data/test_node_neighbor_sampler.py`
Expected: PASS; documentation gap is manual.

**Step 3: Write minimal implementation**

Document that `NodeNeighborSampler` can accept one seed or a rank-1 seed collection per dataset item and that materialization preserves the flat `NodeBatch.seed_index` contract by expanding one sampled context into multiple seed entries.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q`
Expected: PASS across the repository.

**Step 5: Commit**

Run: `git add vgl tests docs README.md && git commit -m "feat: support multi-seed node contexts"`
