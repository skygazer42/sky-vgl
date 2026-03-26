# Heterogeneous Node Block Output Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add relation-local heterogeneous `output_blocks=True` support to `NodeNeighborSampler` when the supervised node type has exactly one inbound relation, covering both local and stitched sampling.

**Architecture:** Keep the existing `NodeBatch.blocks: list[Block]` contract intact by supporting only the unambiguous single-inbound-relation case. Reuse `to_block(...)` for relation-local hetero blocks, retain cumulative per-type hop snapshots during heterogeneous expansion, and build blocks from the final sampled graph after feature overlays.

**Tech Stack:** Python 3.10+, PyTorch, pytest

---

### Task 1: Add failing heterogeneous node block regressions

**Files:**
- Modify: `tests/data/test_node_neighbor_sampler.py`
- Modify: `tests/core/test_node_batch.py`
- Modify: `tests/integration/test_foundation_partition_local.py`

**Step 1: Write the failing test**

Add regressions proving:

- local heterogeneous `NodeNeighborSampler(..., output_blocks=True)` emits relation-local blocks when the supervised `node_type` has one inbound relation
- stitched heterogeneous node sampling through `LocalSamplingCoordinator` emits the same relation-local blocks
- fixed hop count is preserved even when later heterogeneous expansions add no new supervised-type nodes
- coordinator-backed fetched node and edge features remain visible on stitched heterogeneous block graphs
- `NodeBatch.from_samples(...)` batches heterogeneous block layers correctly
- ambiguous multi-inbound-relation heterogeneous node block requests fail clearly

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/data/test_node_neighbor_sampler.py -k "hetero and output_blocks" tests/core/test_node_batch.py -k hetero tests/integration/test_foundation_partition_local.py -k "hetero and block"`
Expected: FAIL because heterogeneous node block output is still rejected.

**Step 3: Write minimal implementation**

No implementation yet in this task.

**Step 4: Run test to verify it still fails for the intended reason**

Run: `python -m pytest -q tests/data/test_node_neighbor_sampler.py -k "hetero and output_blocks" tests/core/test_node_batch.py -k hetero tests/integration/test_foundation_partition_local.py -k "hetero and block"`
Expected: FAIL on the current heterogeneous `output_blocks` guard or missing hetero node block state.

**Step 5: Commit**

Do not commit yet.

### Task 2: Implement local and stitched heterogeneous node block materialization

**Files:**
- Modify: `vgl/ops/khop.py`
- Modify: `vgl/dataloading/sampler.py`
- Modify: `vgl/dataloading/executor.py`
- Modify: `vgl/dataloading/materialize.py`

**Step 1: Write the failing test**

Use the regressions from Task 1.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/data/test_node_neighbor_sampler.py -k "hetero and output_blocks" tests/integration/test_foundation_partition_local.py -k "hetero and block"`
Expected: FAIL

**Step 3: Write minimal implementation**

Implement:

- one helper that resolves the single inbound `edge_type` for the supervised `node_type` or fails clearly
- heterogeneous `return_hops` support for local neighbor expansion
- stitched heterogeneous node-hop tracking through the coordinator path
- relation-local block construction from the sampled graph using the supervised destination-type hop snapshots
- storage of `node_block_edge_type` and per-type hop state through executor/materialization

Implementation rules:

- preserve current homogeneous behavior exactly
- only support the single-inbound-relation heterogeneous case
- build blocks after feature overlays, not before
- keep `NodeBatch.graph` and `seed_index` semantics unchanged

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/data/test_node_neighbor_sampler.py -k "hetero and output_blocks" tests/integration/test_foundation_partition_local.py -k "hetero and block"`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 3: Verify heterogeneous block batching

**Files:**
- Modify: `tests/core/test_node_batch.py`
- Modify: `vgl/graph/batch.py` only if the regression reveals a real batching gap

**Step 1: Write the failing test**

Use the heterogeneous block batching regression from Task 1.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/core/test_node_batch.py -k hetero`
Expected: FAIL if heterogeneous block batching needs adjustment.

**Step 3: Write minimal implementation**

Prefer using the existing `_batch_block_layer(...)` behavior unchanged. Only patch `vgl/graph/batch.py` if the new regression exposes a concrete batching bug.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/core/test_node_batch.py -k hetero`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 4: Refresh docs and run full regression

**Files:**
- Modify: `README.md`
- Modify: `docs/core-concepts.md`
- Modify: `docs/quickstart.md`

**Step 1: Write the failing test**

No code test. Review docs for places that still say node block output is homogeneous-only.

**Step 2: Run test to verify the code path state**

Run: `python -m pytest -q tests/data/test_node_neighbor_sampler.py tests/core/test_node_batch.py tests/integration/test_foundation_partition_local.py`
Expected: PASS if the heterogeneous node block path is complete.

**Step 3: Write minimal implementation**

Document that:

- `NodeNeighborSampler(..., output_blocks=True)` now supports relation-local heterogeneous node blocks when the supervised `node_type` has one inbound relation
- the stitched coordinator-backed path supports the same contract
- ambiguous multi-inbound-relation node types still fail clearly

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q`
Expected: PASS across the repository.

**Step 5: Commit**

Run: `git add vgl tests docs README.md && git commit -m "feat: add hetero node block output"`
