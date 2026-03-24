# Foundation Scale Ops Design

**Problem**

`vgl` already covers a strong single-process modeling surface: one `Graph` abstraction, many convolution layers, several graph tasks, and a trainer stack. What it does not yet have is the lower-level systems foundation needed to compete with DGL-class infrastructure for large-graph training, graph structure manipulation, sparse execution, and dataset/runtime plumbing.

**Design Goal**

Extend VGL with native foundation layers for sparse graph execution, graph operations, large-graph storage, dataset catalog/on-disk access, and distributed training primitives without breaking the current `Graph`, `Trainer`, and `dataloading` APIs. New APIs should be VGL-native first, with compatibility bridges only at ecosystem boundaries.

## Architecture

The long-term structure adds six foundation layers:

1. `vgl.sparse`
   Defines sparse layouts, conversion helpers, adjacency caches, and sparse compute helpers. This layer owns COO/CSR/CSC representation and layout-aware kernels.

2. `vgl.storage`
   Defines `TensorStore`, `FeatureStore`, and graph storage backends for in-memory, mmap-backed, and future remote feature access. This layer lets graphs reference tensors without forcing all features into process memory.

3. `vgl.ops`
   Defines graph structure operations and transform pipelines, including self-loop transforms, bidirectionality, node/edge induced subgraphs, k-hop extraction, compaction, relabeling, and transform composition.

4. `vgl.dataloading`
   Evolves from direct synchronous samplers into a request/plan/executor model. User input becomes a normalized seed request, then flows through expansion, compaction, feature fetch, and batch materialization stages.

5. `vgl.data`
   Gains dataset catalog and cache responsibilities: manifests, fingerprints, cache directories, on-disk dataset loading, and small built-in dataset wrappers.

6. `vgl.distributed`
   Provides partition metadata, local shard loading, distributed feature/store protocols, and sampling coordination contracts. Early milestones stay single-node and local-first, but APIs should leave room for remote execution.

## Data Flow

Current loaders call `sampler.sample(item)` and batch the return value immediately. The new design introduces three neutral contracts:

- `SeedRequest`: normalized user intent such as node seeds, link seeds, temporal seeds, ranking candidates, or metadata labels
- `SamplingPlan`: an ordered description of stages to execute
- `PlanExecutor`: the runtime that resolves a plan against graph/storage/distributed backends

The target data flow is:

`record or graph -> SeedRequest -> SamplingPlan -> stage execution -> compacted subgraph/features -> batch`

This keeps node, link, temporal, homogeneous, heterogeneous, in-memory, on-disk, and eventually distributed workloads on one conceptual path. A sampler becomes a plan builder rather than a bespoke batch constructor.

## Migration Strategy

The current API remains the user-facing surface during migration.

Phase 1 adds new packages and internal primitives without changing behavior.  
Phase 2 rewires existing samplers and graph methods to use the new primitives under the hood.  
Phase 3 introduces explicit advanced APIs such as storage-backed graphs, dataset manifests, partitioned graphs, and sampling plans.  
Phase 4 adds compatibility bridges and deprecation notes for any legacy code paths that become thin wrappers.

This approach keeps the existing test suite valuable. Each new layer gets its own focused tests, while integration tests prove that old user workflows still work.

## Error Handling

Each new subsystem should fail with typed, domain-specific errors:

- sparse layout mismatches and invalid indices
- unavailable features or tensors in storage backends
- invalid transform requests such as missing node types or impossible compactions
- malformed dataset manifests or partition metadata
- unsupported distributed execution modes

Errors should be raised as early as possible with actionable messages, and cross-layer adapters should preserve the original failure cause.

## Testing Strategy

Implementation should proceed with TDD. Every capability lands with:

- focused unit tests for the new module
- package/export tests when new namespaces are exposed
- regression tests for current API compatibility
- integration coverage when a feature changes loader/trainer behavior

Verification remains `python -m pytest -q` at the repository root, plus targeted test commands per task during development.
