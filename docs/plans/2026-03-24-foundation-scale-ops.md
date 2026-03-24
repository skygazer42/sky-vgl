# Foundation Scale Ops Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build VGL-native foundation layers for sparse execution, graph operations, large-graph storage, dataset/runtime plumbing, and distributed training primitives while preserving the current public modeling API.

**Architecture:** Add new `vgl.sparse`, `vgl.storage`, `vgl.ops`, `vgl.data` catalog, and `vgl.distributed` layers first, then migrate current graph and dataloading code to consume them internally. Keep `Graph`, existing samplers, and `Trainer` as the user-facing entry points during the migration.

**Tech Stack:** Python 3.11+, PyTorch, pytest, existing VGL package layout

---

### Task 1: Create the new foundation package namespaces

**Files:**
- Create: `vgl/sparse/__init__.py`
- Create: `vgl/storage/__init__.py`
- Create: `vgl/ops/__init__.py`
- Test: `tests/test_package_layout.py`

**Step 1: Write the failing test**
Add package-layout assertions that `vgl.sparse`, `vgl.storage`, and `vgl.ops` can be imported and expose stable `__all__` lists.

**Step 2: Run test to verify it fails**
Run: `python -m pytest -q tests/test_package_layout.py -k foundation`
Expected: FAIL with import errors for the new namespaces.

**Step 3: Write minimal implementation**
Create empty package modules that expose placeholder `__all__` definitions.

**Step 4: Run test to verify it passes**
Run: `python -m pytest -q tests/test_package_layout.py -k foundation`
Expected: PASS

**Step 5: Commit**
Run: `git add tests/test_package_layout.py vgl/sparse/__init__.py vgl/storage/__init__.py vgl/ops/__init__.py && git commit -m "feat: add foundation package namespaces"`

### Task 2: Add sparse layout and sparse tensor primitives

**Files:**
- Create: `vgl/sparse/base.py`
- Modify: `vgl/sparse/__init__.py`
- Test: `tests/sparse/test_sparse_base.py`

**Step 1: Write the failing test**
Add tests for `SparseLayout`, `SparseTensor`, `nnz`, and invalid index/shape combinations.

**Step 2: Run test to verify it fails**
Run: `python -m pytest -q tests/sparse/test_sparse_base.py`
Expected: FAIL because the module and types do not exist.

**Step 3: Write minimal implementation**
Add a small immutable sparse container with indices, shape, layout, and validation helpers.

**Step 4: Run test to verify it passes**
Run: `python -m pytest -q tests/sparse/test_sparse_base.py`
Expected: PASS

**Step 5: Commit**
Run: `git add tests/sparse/test_sparse_base.py vgl/sparse/base.py vgl/sparse/__init__.py && git commit -m "feat: add sparse tensor primitives"`

### Task 3: Add sparse conversion helpers

**Files:**
- Create: `vgl/sparse/convert.py`
- Modify: `vgl/sparse/__init__.py`
- Test: `tests/sparse/test_sparse_convert.py`

**Step 1: Write the failing test**
Add tests for COO edge index to `SparseTensor`, layout conversion, and round-trip preservation.

**Step 2: Run test to verify it fails**
Run: `python -m pytest -q tests/sparse/test_sparse_convert.py`
Expected: FAIL because conversion helpers do not exist.

**Step 3: Write minimal implementation**
Add `from_edge_index(...)`, `to_coo(...)`, and `to_csr(...)` helpers.

**Step 4: Run test to verify it passes**
Run: `python -m pytest -q tests/sparse/test_sparse_convert.py`
Expected: PASS

**Step 5: Commit**
Run: `git add tests/sparse/test_sparse_convert.py vgl/sparse/convert.py vgl/sparse/__init__.py && git commit -m "feat: add sparse conversion helpers"`

### Task 4: Add sparse graph operations

**Files:**
- Create: `vgl/sparse/ops.py`
- Modify: `vgl/sparse/__init__.py`
- Test: `tests/sparse/test_sparse_ops.py`

**Step 1: Write the failing test**
Add tests for sparse degree computation, row selection, and sparse-dense matmul on toy graphs.

**Step 2: Run test to verify it fails**
Run: `python -m pytest -q tests/sparse/test_sparse_ops.py`
Expected: FAIL because sparse ops do not exist.

**Step 3: Write minimal implementation**
Implement `degree`, `select_rows`, and `spmm`.

**Step 4: Run test to verify it passes**
Run: `python -m pytest -q tests/sparse/test_sparse_ops.py`
Expected: PASS

**Step 5: Commit**
Run: `git add tests/sparse/test_sparse_ops.py vgl/sparse/ops.py vgl/sparse/__init__.py && git commit -m "feat: add sparse graph ops"`

### Task 5: Add graph adjacency cache support

**Files:**
- Modify: `vgl/graph/stores.py`
- Modify: `vgl/graph/graph.py`
- Test: `tests/core/test_graph_sparse_cache.py`

**Step 1: Write the failing test**
Add tests proving a graph can build and cache sparse adjacency views per edge type.

**Step 2: Run test to verify it fails**
Run: `python -m pytest -q tests/core/test_graph_sparse_cache.py`
Expected: FAIL because graph caches are missing.

**Step 3: Write minimal implementation**
Add cache slots and `adjacency(layout="coo")` helpers.

**Step 4: Run test to verify it passes**
Run: `python -m pytest -q tests/core/test_graph_sparse_cache.py`
Expected: PASS

**Step 5: Commit**
Run: `git add tests/core/test_graph_sparse_cache.py vgl/graph/stores.py vgl/graph/graph.py && git commit -m "feat: cache graph adjacency layouts"`

### Task 6: Add graph operation pipeline primitives

**Files:**
- Create: `vgl/ops/pipeline.py`
- Modify: `vgl/ops/__init__.py`
- Test: `tests/ops/test_pipeline.py`

**Step 1: Write the failing test**
Add tests for a composable transform pipeline that applies operations in order.

**Step 2: Run test to verify it fails**
Run: `python -m pytest -q tests/ops/test_pipeline.py`
Expected: FAIL because pipeline types do not exist.

**Step 3: Write minimal implementation**
Implement `GraphTransform` protocol and `TransformPipeline`.

**Step 4: Run test to verify it passes**
Run: `python -m pytest -q tests/ops/test_pipeline.py`
Expected: PASS

**Step 5: Commit**
Run: `git add tests/ops/test_pipeline.py vgl/ops/pipeline.py vgl/ops/__init__.py && git commit -m "feat: add graph transform pipeline"`

### Task 7: Add self-loop and bidirectional graph operations

**Files:**
- Create: `vgl/ops/structure.py`
- Modify: `vgl/ops/__init__.py`
- Test: `tests/ops/test_structure_ops.py`

**Step 1: Write the failing test**
Add tests for `add_self_loops`, `remove_self_loops`, and `to_bidirected`.

**Step 2: Run test to verify it fails**
Run: `python -m pytest -q tests/ops/test_structure_ops.py`
Expected: FAIL because structure ops do not exist.

**Step 3: Write minimal implementation**
Implement structure transforms for homogeneous graphs first.

**Step 4: Run test to verify it passes**
Run: `python -m pytest -q tests/ops/test_structure_ops.py`
Expected: PASS

**Step 5: Commit**
Run: `git add tests/ops/test_structure_ops.py vgl/ops/structure.py vgl/ops/__init__.py && git commit -m "feat: add graph structure transforms"`

### Task 8: Add induced subgraph extraction

**Files:**
- Create: `vgl/ops/subgraph.py`
- Modify: `vgl/ops/__init__.py`
- Test: `tests/ops/test_subgraph_ops.py`

**Step 1: Write the failing test**
Add tests for node-induced and edge-induced subgraph extraction, including retained features.

**Step 2: Run test to verify it fails**
Run: `python -m pytest -q tests/ops/test_subgraph_ops.py`
Expected: FAIL because subgraph ops do not exist.

**Step 3: Write minimal implementation**
Implement `node_subgraph` and `edge_subgraph`.

**Step 4: Run test to verify it passes**
Run: `python -m pytest -q tests/ops/test_subgraph_ops.py`
Expected: PASS

**Step 5: Commit**
Run: `git add tests/ops/test_subgraph_ops.py vgl/ops/subgraph.py vgl/ops/__init__.py && git commit -m "feat: add induced subgraph ops"`

### Task 9: Add k-hop extraction utilities

**Files:**
- Create: `vgl/ops/khop.py`
- Modify: `vgl/ops/__init__.py`
- Test: `tests/ops/test_khop_ops.py`

**Step 1: Write the failing test**
Add tests for inbound/outbound k-hop node expansion and edge frontier extraction.

**Step 2: Run test to verify it fails**
Run: `python -m pytest -q tests/ops/test_khop_ops.py`
Expected: FAIL because k-hop ops do not exist.

**Step 3: Write minimal implementation**
Implement `khop_nodes` and `khop_subgraph`.

**Step 4: Run test to verify it passes**
Run: `python -m pytest -q tests/ops/test_khop_ops.py`
Expected: PASS

**Step 5: Commit**
Run: `git add tests/ops/test_khop_ops.py vgl/ops/khop.py vgl/ops/__init__.py && git commit -m "feat: add k-hop graph ops"`

### Task 10: Add node compaction and relabel helpers

**Files:**
- Create: `vgl/ops/compact.py`
- Modify: `vgl/ops/__init__.py`
- Test: `tests/ops/test_compact_ops.py`

**Step 1: Write the failing test**
Add tests proving extracted subgraphs can be compacted and mapped back to original ids.

**Step 2: Run test to verify it fails**
Run: `python -m pytest -q tests/ops/test_compact_ops.py`
Expected: FAIL because compaction helpers do not exist.

**Step 3: Write minimal implementation**
Implement `compact_nodes` returning relabeled graph plus id mapping metadata.

**Step 4: Run test to verify it passes**
Run: `python -m pytest -q tests/ops/test_compact_ops.py`
Expected: PASS

**Step 5: Commit**
Run: `git add tests/ops/test_compact_ops.py vgl/ops/compact.py vgl/ops/__init__.py && git commit -m "feat: add graph compaction helpers"`

### Task 11: Bridge graph methods to the new ops layer

**Files:**
- Modify: `vgl/graph/graph.py`
- Test: `tests/core/test_graph_ops_api.py`

**Step 1: Write the failing test**
Add tests for convenience methods such as `graph.add_self_loops()` and `graph.node_subgraph(...)`.

**Step 2: Run test to verify it fails**
Run: `python -m pytest -q tests/core/test_graph_ops_api.py`
Expected: FAIL because graph helpers do not exist.

**Step 3: Write minimal implementation**
Delegate graph methods to the new `vgl.ops` implementations.

**Step 4: Run test to verify it passes**
Run: `python -m pytest -q tests/core/test_graph_ops_api.py`
Expected: PASS

**Step 5: Commit**
Run: `git add tests/core/test_graph_ops_api.py vgl/graph/graph.py && git commit -m "feat: bridge graph helpers to ops layer"`

### Task 12: Add tensor store protocols

**Files:**
- Create: `vgl/storage/base.py`
- Modify: `vgl/storage/__init__.py`
- Test: `tests/storage/test_tensor_store.py`

**Step 1: Write the failing test**
Add tests for `TensorStore` contract: shape lookup, row fetch, and dtype/device preservation.

**Step 2: Run test to verify it fails**
Run: `python -m pytest -q tests/storage/test_tensor_store.py`
Expected: FAIL because storage primitives do not exist.

**Step 3: Write minimal implementation**
Add abstract `TensorStore` and a small fetch result contract.

**Step 4: Run test to verify it passes**
Run: `python -m pytest -q tests/storage/test_tensor_store.py`
Expected: PASS

**Step 5: Commit**
Run: `git add tests/storage/test_tensor_store.py vgl/storage/base.py vgl/storage/__init__.py && git commit -m "feat: add tensor store protocols"`

### Task 13: Add in-memory tensor stores

**Files:**
- Create: `vgl/storage/memory.py`
- Modify: `vgl/storage/__init__.py`
- Test: `tests/storage/test_memory_store.py`

**Step 1: Write the failing test**
Add tests for an in-memory store that returns row slices by id.

**Step 2: Run test to verify it fails**
Run: `python -m pytest -q tests/storage/test_memory_store.py`
Expected: FAIL because `InMemoryTensorStore` does not exist.

**Step 3: Write minimal implementation**
Implement `InMemoryTensorStore`.

**Step 4: Run test to verify it passes**
Run: `python -m pytest -q tests/storage/test_memory_store.py`
Expected: PASS

**Step 5: Commit**
Run: `git add tests/storage/test_memory_store.py vgl/storage/memory.py vgl/storage/__init__.py && git commit -m "feat: add in-memory tensor stores"`

### Task 14: Add memory-mapped tensor stores

**Files:**
- Create: `vgl/storage/mmap.py`
- Modify: `vgl/storage/__init__.py`
- Test: `tests/storage/test_mmap_store.py`

**Step 1: Write the failing test**
Add tests for persisting a tensor to a file-backed store and reloading it.

**Step 2: Run test to verify it fails**
Run: `python -m pytest -q tests/storage/test_mmap_store.py`
Expected: FAIL because mmap storage does not exist.

**Step 3: Write minimal implementation**
Implement a local file-backed tensor store using PyTorch serialization or mmap-compatible tensor loading.

**Step 4: Run test to verify it passes**
Run: `python -m pytest -q tests/storage/test_mmap_store.py`
Expected: PASS

**Step 5: Commit**
Run: `git add tests/storage/test_mmap_store.py vgl/storage/mmap.py vgl/storage/__init__.py && git commit -m "feat: add mmap tensor stores"`

### Task 15: Add feature store composition

**Files:**
- Create: `vgl/storage/feature_store.py`
- Modify: `vgl/storage/__init__.py`
- Test: `tests/storage/test_feature_store.py`

**Step 1: Write the failing test**
Add tests for a multi-table feature store keyed by `(entity_kind, type_name, feature_name)`.

**Step 2: Run test to verify it fails**
Run: `python -m pytest -q tests/storage/test_feature_store.py`
Expected: FAIL because feature store composition does not exist.

**Step 3: Write minimal implementation**
Implement `FeatureStore` backed by named `TensorStore` instances.

**Step 4: Run test to verify it passes**
Run: `python -m pytest -q tests/storage/test_feature_store.py`
Expected: PASS

**Step 5: Commit**
Run: `git add tests/storage/test_feature_store.py vgl/storage/feature_store.py vgl/storage/__init__.py && git commit -m "feat: add feature store composition"`

### Task 16: Add graph storage backends

**Files:**
- Create: `vgl/storage/graph_store.py`
- Modify: `vgl/storage/__init__.py`
- Test: `tests/storage/test_graph_store.py`

**Step 1: Write the failing test**
Add tests for edge index lookup, edge count, and type-aware adjacency retrieval from a storage backend.

**Step 2: Run test to verify it fails**
Run: `python -m pytest -q tests/storage/test_graph_store.py`
Expected: FAIL because graph storage backends do not exist.

**Step 3: Write minimal implementation**
Implement an in-memory graph store and a protocol for future on-disk/remote graph stores.

**Step 4: Run test to verify it passes**
Run: `python -m pytest -q tests/storage/test_graph_store.py`
Expected: PASS

**Step 5: Commit**
Run: `git add tests/storage/test_graph_store.py vgl/storage/graph_store.py vgl/storage/__init__.py && git commit -m "feat: add graph storage backends"`

### Task 17: Add storage-backed graph construction

**Files:**
- Modify: `vgl/graph/graph.py`
- Modify: `vgl/graph/stores.py`
- Test: `tests/core/test_feature_backed_graph.py`

**Step 1: Write the failing test**
Add tests for building a graph that resolves node/edge features through `FeatureStore` and adjacency through a graph store backend.

**Step 2: Run test to verify it fails**
Run: `python -m pytest -q tests/core/test_feature_backed_graph.py`
Expected: FAIL because storage-backed graphs are not supported.

**Step 3: Write minimal implementation**
Add optional store-backed constructors while preserving current in-memory graph behavior.

**Step 4: Run test to verify it passes**
Run: `python -m pytest -q tests/core/test_feature_backed_graph.py`
Expected: PASS

**Step 5: Commit**
Run: `git add tests/core/test_feature_backed_graph.py vgl/graph/graph.py vgl/graph/stores.py && git commit -m "feat: add storage-backed graph support"`

### Task 18: Add normalized seed request models

**Files:**
- Create: `vgl/dataloading/requests.py`
- Test: `tests/data/test_seed_requests.py`

**Step 1: Write the failing test**
Add tests for typed seed requests for node, link, temporal, and graph-classification entry points.

**Step 2: Run test to verify it fails**
Run: `python -m pytest -q tests/data/test_seed_requests.py`
Expected: FAIL because request models do not exist.

**Step 3: Write minimal implementation**
Implement request dataclasses with validation and metadata preservation.

**Step 4: Run test to verify it passes**
Run: `python -m pytest -q tests/data/test_seed_requests.py`
Expected: PASS

**Step 5: Commit**
Run: `git add tests/data/test_seed_requests.py vgl/dataloading/requests.py && git commit -m "feat: add normalized seed requests"`

### Task 19: Add sampling plan primitives

**Files:**
- Create: `vgl/dataloading/plan.py`
- Test: `tests/data/test_sampling_plan.py`

**Step 1: Write the failing test**
Add tests for a plan object that preserves ordered execution stages and plan metadata.

**Step 2: Run test to verify it fails**
Run: `python -m pytest -q tests/data/test_sampling_plan.py`
Expected: FAIL because plan primitives do not exist.

**Step 3: Write minimal implementation**
Implement `SamplingPlan` and stage records.

**Step 4: Run test to verify it passes**
Run: `python -m pytest -q tests/data/test_sampling_plan.py`
Expected: PASS

**Step 5: Commit**
Run: `git add tests/data/test_sampling_plan.py vgl/dataloading/plan.py && git commit -m "feat: add sampling plan primitives"`

### Task 20: Add sampling plan executor contracts

**Files:**
- Create: `vgl/dataloading/executor.py`
- Test: `tests/data/test_sampling_executor.py`

**Step 1: Write the failing test**
Add tests proving a plan executor runs stages in order and returns a materialization context.

**Step 2: Run test to verify it fails**
Run: `python -m pytest -q tests/data/test_sampling_executor.py`
Expected: FAIL because plan executors do not exist.

**Step 3: Write minimal implementation**
Implement executor and a simple registry of stage handlers.

**Step 4: Run test to verify it passes**
Run: `python -m pytest -q tests/data/test_sampling_executor.py`
Expected: PASS

**Step 5: Commit**
Run: `git add tests/data/test_sampling_executor.py vgl/dataloading/executor.py && git commit -m "feat: add sampling plan executor"`

### Task 21: Add feature fetch stages

**Files:**
- Modify: `vgl/dataloading/executor.py`
- Test: `tests/data/test_feature_fetch_stage.py`

**Step 1: Write the failing test**
Add tests for plan stages that fetch node/edge features via `FeatureStore`.

**Step 2: Run test to verify it fails**
Run: `python -m pytest -q tests/data/test_feature_fetch_stage.py`
Expected: FAIL because feature fetch stages are missing.

**Step 3: Write minimal implementation**
Implement feature fetch stage handlers.

**Step 4: Run test to verify it passes**
Run: `python -m pytest -q tests/data/test_feature_fetch_stage.py`
Expected: PASS

**Step 5: Commit**
Run: `git add tests/data/test_feature_fetch_stage.py vgl/dataloading/executor.py && git commit -m "feat: add feature fetch stages"`

### Task 22: Add neighbor expansion stages

**Files:**
- Modify: `vgl/dataloading/executor.py`
- Modify: `vgl/ops/khop.py`
- Test: `tests/data/test_neighbor_expansion_stage.py`

**Step 1: Write the failing test**
Add tests for staged neighbor expansion from node seeds.

**Step 2: Run test to verify it fails**
Run: `python -m pytest -q tests/data/test_neighbor_expansion_stage.py`
Expected: FAIL because neighbor expansion stages are missing.

**Step 3: Write minimal implementation**
Implement neighbor expansion stage handlers using `vgl.ops`/`vgl.sparse`.

**Step 4: Run test to verify it passes**
Run: `python -m pytest -q tests/data/test_neighbor_expansion_stage.py`
Expected: PASS

**Step 5: Commit**
Run: `git add tests/data/test_neighbor_expansion_stage.py vgl/dataloading/executor.py vgl/ops/khop.py && git commit -m "feat: add neighbor expansion stages"`

### Task 23: Add batch materialization helpers

**Files:**
- Create: `vgl/dataloading/materialize.py`
- Test: `tests/data/test_batch_materialize.py`

**Step 1: Write the failing test**
Add tests for materializing executor context into `GraphBatch`, `NodeBatch`, `LinkPredictionBatch`, and `TemporalEventBatch`.

**Step 2: Run test to verify it fails**
Run: `python -m pytest -q tests/data/test_batch_materialize.py`
Expected: FAIL because materialization helpers do not exist.

**Step 3: Write minimal implementation**
Extract the current `_build_batch` logic into reusable materializers.

**Step 4: Run test to verify it passes**
Run: `python -m pytest -q tests/data/test_batch_materialize.py`
Expected: PASS

**Step 5: Commit**
Run: `git add tests/data/test_batch_materialize.py vgl/dataloading/materialize.py && git commit -m "feat: add batch materializers"`

### Task 24: Rewire `NodeNeighborSampler` to build plans

**Files:**
- Modify: `vgl/dataloading/sampler.py`
- Modify: `vgl/dataloading/loader.py`
- Modify: `vgl/dataloading/__init__.py`
- Test: `tests/data/test_node_neighbor_sampler.py`

**Step 1: Write the failing test**
Add or tighten tests proving node neighbor sampling produces the same visible batches through the new plan path.

**Step 2: Run test to verify it fails**
Run: `python -m pytest -q tests/data/test_node_neighbor_sampler.py`
Expected: FAIL because sampler still uses the legacy direct path.

**Step 3: Write minimal implementation**
Have the sampler emit `SamplingPlan` objects and let the loader execute them.

**Step 4: Run test to verify it passes**
Run: `python -m pytest -q tests/data/test_node_neighbor_sampler.py`
Expected: PASS

**Step 5: Commit**
Run: `git add tests/data/test_node_neighbor_sampler.py vgl/dataloading/sampler.py vgl/dataloading/loader.py vgl/dataloading/__init__.py && git commit -m "refactor: route node sampling through plans"`

### Task 25: Rewire `LinkNeighborSampler` to build plans

**Files:**
- Modify: `vgl/dataloading/sampler.py`
- Modify: `vgl/dataloading/loader.py`
- Test: `tests/data/test_link_neighbor_sampler.py`

**Step 1: Write the failing test**
Add or tighten tests proving link sampling and negative sampling still behave correctly after plan execution.

**Step 2: Run test to verify it fails**
Run: `python -m pytest -q tests/data/test_link_neighbor_sampler.py`
Expected: FAIL because link sampling still uses the legacy path.

**Step 3: Write minimal implementation**
Route link neighbor sampling through plan execution.

**Step 4: Run test to verify it passes**
Run: `python -m pytest -q tests/data/test_link_neighbor_sampler.py`
Expected: PASS

**Step 5: Commit**
Run: `git add tests/data/test_link_neighbor_sampler.py vgl/dataloading/sampler.py vgl/dataloading/loader.py && git commit -m "refactor: route link sampling through plans"`

### Task 26: Rewire `TemporalNeighborSampler` to build plans

**Files:**
- Modify: `vgl/dataloading/sampler.py`
- Modify: `vgl/dataloading/loader.py`
- Test: `tests/data/test_temporal_neighbor_sampler.py`

**Step 1: Write the failing test**
Add or tighten tests proving temporal filtering and history expansion still behave correctly.

**Step 2: Run test to verify it fails**
Run: `python -m pytest -q tests/data/test_temporal_neighbor_sampler.py`
Expected: FAIL because temporal sampling still uses the legacy path.

**Step 3: Write minimal implementation**
Route temporal sampling through plan execution.

**Step 4: Run test to verify it passes**
Run: `python -m pytest -q tests/data/test_temporal_neighbor_sampler.py`
Expected: PASS

**Step 5: Commit**
Run: `git add tests/data/test_temporal_neighbor_sampler.py vgl/dataloading/sampler.py vgl/dataloading/loader.py && git commit -m "refactor: route temporal sampling through plans"`

### Task 27: Add loader prefetch support

**Files:**
- Modify: `vgl/dataloading/loader.py`
- Test: `tests/data/test_loader_prefetch.py`

**Step 1: Write the failing test**
Add tests for bounded prefetch behavior without changing output order.

**Step 2: Run test to verify it fails**
Run: `python -m pytest -q tests/data/test_loader_prefetch.py`
Expected: FAIL because prefetch options are missing.

**Step 3: Write minimal implementation**
Add optional prefetch iterator hooks that remain single-process for now.

**Step 4: Run test to verify it passes**
Run: `python -m pytest -q tests/data/test_loader_prefetch.py`
Expected: PASS

**Step 5: Commit**
Run: `git add tests/data/test_loader_prefetch.py vgl/dataloading/loader.py && git commit -m "feat: add loader prefetch hooks"`

### Task 28: Add dataset manifest models

**Files:**
- Create: `vgl/data/catalog.py`
- Test: `tests/data/test_dataset_catalog.py`

**Step 1: Write the failing test**
Add tests for dataset manifest metadata, fingerprinting, and split declarations.

**Step 2: Run test to verify it fails**
Run: `python -m pytest -q tests/data/test_dataset_catalog.py`
Expected: FAIL because dataset catalog models do not exist.

**Step 3: Write minimal implementation**
Implement manifest dataclasses and a small catalog registry.

**Step 4: Run test to verify it passes**
Run: `python -m pytest -q tests/data/test_dataset_catalog.py`
Expected: PASS

**Step 5: Commit**
Run: `git add tests/data/test_dataset_catalog.py vgl/data/catalog.py && git commit -m "feat: add dataset catalog models"`

### Task 29: Add dataset cache and fingerprint utilities

**Files:**
- Create: `vgl/data/cache.py`
- Test: `tests/data/test_data_cache.py`

**Step 1: Write the failing test**
Add tests for cache directory resolution, manifest fingerprinting, and cache hits.

**Step 2: Run test to verify it fails**
Run: `python -m pytest -q tests/data/test_data_cache.py`
Expected: FAIL because cache helpers do not exist.

**Step 3: Write minimal implementation**
Implement local cache helpers and manifest fingerprints.

**Step 4: Run test to verify it passes**
Run: `python -m pytest -q tests/data/test_data_cache.py`
Expected: PASS

**Step 5: Commit**
Run: `git add tests/data/test_data_cache.py vgl/data/cache.py && git commit -m "feat: add dataset cache helpers"`

### Task 30: Add built-in dataset wrappers

**Files:**
- Create: `vgl/data/datasets.py`
- Modify: `vgl/data/__init__.py`
- Test: `tests/data/test_builtin_datasets.py`

**Step 1: Write the failing test**
Add tests for a built-in dataset base class and one toy graph dataset wrapper.

**Step 2: Run test to verify it fails**
Run: `python -m pytest -q tests/data/test_builtin_datasets.py`
Expected: FAIL because built-in dataset wrappers do not exist.

**Step 3: Write minimal implementation**
Implement a minimal dataset base and one fixture-backed built-in dataset.

**Step 4: Run test to verify it passes**
Run: `python -m pytest -q tests/data/test_builtin_datasets.py`
Expected: PASS

**Step 5: Commit**
Run: `git add tests/data/test_builtin_datasets.py vgl/data/datasets.py vgl/data/__init__.py && git commit -m "feat: add built-in dataset wrappers"`

### Task 31: Add on-disk graph dataset format

**Files:**
- Create: `vgl/data/ondisk.py`
- Test: `tests/data/test_ondisk_dataset.py`

**Step 1: Write the failing test**
Add tests for writing a small graph dataset manifest to disk and reloading it.

**Step 2: Run test to verify it fails**
Run: `python -m pytest -q tests/data/test_ondisk_dataset.py`
Expected: FAIL because on-disk datasets do not exist.

**Step 3: Write minimal implementation**
Implement a local manifest-plus-tensor-file format.

**Step 4: Run test to verify it passes**
Run: `python -m pytest -q tests/data/test_ondisk_dataset.py`
Expected: PASS

**Step 5: Commit**
Run: `git add tests/data/test_ondisk_dataset.py vgl/data/ondisk.py && git commit -m "feat: add on-disk graph dataset format"`

### Task 32: Add distributed namespace and partition metadata

**Files:**
- Create: `vgl/distributed/__init__.py`
- Create: `vgl/distributed/partition.py`
- Test: `tests/distributed/test_partition_metadata.py`

**Step 1: Write the failing test**
Add tests for partition metadata, shard counts, owner resolution, and manifest validation.

**Step 2: Run test to verify it fails**
Run: `python -m pytest -q tests/distributed/test_partition_metadata.py`
Expected: FAIL because distributed partition models do not exist.

**Step 3: Write minimal implementation**
Implement partition metadata dataclasses and validation.

**Step 4: Run test to verify it passes**
Run: `python -m pytest -q tests/distributed/test_partition_metadata.py`
Expected: PASS

**Step 5: Commit**
Run: `git add tests/distributed/test_partition_metadata.py vgl/distributed/__init__.py vgl/distributed/partition.py && git commit -m "feat: add partition metadata models"`

### Task 33: Add partition writing utilities

**Files:**
- Create: `vgl/distributed/writer.py`
- Test: `tests/distributed/test_partition_writer.py`

**Step 1: Write the failing test**
Add tests for splitting a toy graph into partition files with metadata.

**Step 2: Run test to verify it fails**
Run: `python -m pytest -q tests/distributed/test_partition_writer.py`
Expected: FAIL because partition writers do not exist.

**Step 3: Write minimal implementation**
Implement a deterministic local partition writer.

**Step 4: Run test to verify it passes**
Run: `python -m pytest -q tests/distributed/test_partition_writer.py`
Expected: PASS

**Step 5: Commit**
Run: `git add tests/distributed/test_partition_writer.py vgl/distributed/writer.py && git commit -m "feat: add local graph partition writer"`

### Task 34: Add local shard loading

**Files:**
- Create: `vgl/distributed/shard.py`
- Test: `tests/distributed/test_local_shard.py`

**Step 1: Write the failing test**
Add tests for loading one partition into a shard-local graph/store view.

**Step 2: Run test to verify it fails**
Run: `python -m pytest -q tests/distributed/test_local_shard.py`
Expected: FAIL because local shard loading does not exist.

**Step 3: Write minimal implementation**
Implement `LocalGraphShard` and partition manifest loading.

**Step 4: Run test to verify it passes**
Run: `python -m pytest -q tests/distributed/test_local_shard.py`
Expected: PASS

**Step 5: Commit**
Run: `git add tests/distributed/test_local_shard.py vgl/distributed/shard.py && git commit -m "feat: add local shard loading"`

### Task 35: Add distributed store protocols

**Files:**
- Create: `vgl/distributed/store.py`
- Modify: `vgl/distributed/__init__.py`
- Test: `tests/distributed/test_store_protocol.py`

**Step 1: Write the failing test**
Add tests for distributed feature/table lookup contracts, with a local stub backend.

**Step 2: Run test to verify it fails**
Run: `python -m pytest -q tests/distributed/test_store_protocol.py`
Expected: FAIL because distributed stores do not exist.

**Step 3: Write minimal implementation**
Implement base protocol and a local passthrough adapter.

**Step 4: Run test to verify it passes**
Run: `python -m pytest -q tests/distributed/test_store_protocol.py`
Expected: PASS

**Step 5: Commit**
Run: `git add tests/distributed/test_store_protocol.py vgl/distributed/store.py vgl/distributed/__init__.py && git commit -m "feat: add distributed store protocols"`

### Task 36: Add distributed sampling coordination contracts

**Files:**
- Create: `vgl/distributed/coordinator.py`
- Modify: `vgl/distributed/__init__.py`
- Test: `tests/distributed/test_sampling_coordinator.py`

**Step 1: Write the failing test**
Add tests for planning shard-local seed routing and feature fetch coordination.

**Step 2: Run test to verify it fails**
Run: `python -m pytest -q tests/distributed/test_sampling_coordinator.py`
Expected: FAIL because sampling coordination contracts do not exist.

**Step 3: Write minimal implementation**
Implement coordinator interfaces and a single-process local coordinator.

**Step 4: Run test to verify it passes**
Run: `python -m pytest -q tests/distributed/test_sampling_coordinator.py`
Expected: PASS

**Step 5: Commit**
Run: `git add tests/distributed/test_sampling_coordinator.py vgl/distributed/coordinator.py vgl/distributed/__init__.py && git commit -m "feat: add sampling coordination contracts"`

### Task 37: Add trainer compatibility with plan-backed and distributed-aware loaders

**Files:**
- Modify: `vgl/engine/trainer.py`
- Test: `tests/train/test_trainer_distributed_loader.py`

**Step 1: Write the failing test**
Add tests proving `Trainer.fit/evaluate/test` accept the upgraded loader outputs without API changes.

**Step 2: Run test to verify it fails**
Run: `python -m pytest -q tests/train/test_trainer_distributed_loader.py`
Expected: FAIL because trainer integration is incomplete.

**Step 3: Write minimal implementation**
Adjust trainer assumptions only where needed to accept richer batch metadata.

**Step 4: Run test to verify it passes**
Run: `python -m pytest -q tests/train/test_trainer_distributed_loader.py`
Expected: PASS

**Step 5: Commit**
Run: `git add tests/train/test_trainer_distributed_loader.py vgl/engine/trainer.py && git commit -m "feat: support plan-backed loaders in trainer"`

### Task 38: Add compatibility bridges for new foundation layers

**Files:**
- Modify: `vgl/compat/dgl.py`
- Modify: `vgl/compat/__init__.py`
- Test: `tests/compat/test_dgl_adapter.py`

**Step 1: Write the failing test**
Add compatibility tests covering conversion of graphs with sparse caches and storage-backed metadata.

**Step 2: Run test to verify it fails**
Run: `python -m pytest -q tests/compat/test_dgl_adapter.py`
Expected: FAIL because new foundation metadata is not bridged.

**Step 3: Write minimal implementation**
Extend compatibility adapters to ignore or translate new metadata safely.

**Step 4: Run test to verify it passes**
Run: `python -m pytest -q tests/compat/test_dgl_adapter.py`
Expected: PASS

**Step 5: Commit**
Run: `git add tests/compat/test_dgl_adapter.py vgl/compat/dgl.py vgl/compat/__init__.py && git commit -m "feat: bridge new foundation metadata in compat adapters"`

### Task 39: Update docs for the new foundation model

**Files:**
- Modify: `docs/core-concepts.md`
- Modify: `docs/quickstart.md`
- Modify: `README.md`

**Step 1: Write the failing test**
Add or tighten doc assertions in existing repo-identity/package-layout tests if documentation contracts are enforced there.

**Step 2: Run test to verify it fails**
Run: `python -m pytest -q tests/test_repo_identity.py tests/test_package_layout.py`
Expected: FAIL only if doc-linked contracts were added; otherwise skip directly to implementation and verify with full suite later.

**Step 3: Write minimal implementation**
Document the new foundation layers, migration path, and advanced loader/storage concepts.

**Step 4: Run test to verify it passes**
Run: `python -m pytest -q tests/test_repo_identity.py tests/test_package_layout.py`
Expected: PASS

**Step 5: Commit**
Run: `git add docs/core-concepts.md docs/quickstart.md README.md && git commit -m "docs: describe foundation layers and migration path"`

### Task 40: Add end-to-end regression coverage and run the full suite

**Files:**
- Create: `tests/integration/test_foundation_large_graph_flow.py`
- Create: `tests/integration/test_foundation_ondisk_sampling.py`
- Create: `tests/integration/test_foundation_partition_local.py`
- Modify: any touched modules as needed

**Step 1: Write the failing test**
Add integration tests covering storage-backed graph sampling, on-disk dataset loading into loaders, and local partition metadata driving shard-aware sampling.

**Step 2: Run test to verify it fails**
Run: `python -m pytest -q tests/integration/test_foundation_large_graph_flow.py tests/integration/test_foundation_ondisk_sampling.py tests/integration/test_foundation_partition_local.py`
Expected: FAIL because the end-to-end flows are not complete yet.

**Step 3: Write minimal implementation**
Patch the remaining integration gaps until the new flows work through public APIs.

**Step 4: Run test to verify it passes**
Run: `python -m pytest -q`
Expected: PASS across the full repository.

**Step 5: Commit**
Run: `git add tests/integration/test_foundation_large_graph_flow.py tests/integration/test_foundation_ondisk_sampling.py tests/integration/test_foundation_partition_local.py && git add vgl docs README.md && git commit -m "feat: complete foundation scale ops rollout"`
