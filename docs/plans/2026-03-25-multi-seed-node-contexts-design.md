# Multi-Seed Node Contexts Design

## Problem

The plan/executor stack already accepts `NodeSeedRequest.node_ids` as a rank-1 tensor and neighbor expansion already works over multiple node seeds. The public node sampling path still collapses that flexibility at two higher layers: `NodeNeighborSampler` coerces metadata seeds to one scalar, and node-context materialization raises when one context contains more than one seed. That leaves an avoidable data-loading gap for node mini-batches where one dataset item already represents a small supervision set.

## Goals

- Preserve the current `NodeBatch` and `NodeClassificationTask` contract
- Allow one node context to carry one or many seeds without changing loader or trainer APIs
- Keep homogeneous and heterogeneous node sampling behavior aligned
- Avoid duplicating sampled subgraphs when several seeds share the same context graph

## Design

### 1. Normalize node seeds as rank-1 tensors at the sampler boundary

`NodeNeighborSampler` should accept scalar ints, Python sequences, or tensors in `metadata['seed']` or `SampleRecord.subgraph_seed`, normalize them to one rank-1 `torch.long` tensor, and validate every seed against the selected node type.

### 2. Materialize one sampled graph per context, then expand supervision records

Node context materialization should still build the sampled subgraph once. Instead of forcing one `SampleRecord` per context, it should emit one `SampleRecord` per requested seed, all referencing the same sampled graph object and carrying the corresponding local `subgraph_seed`.

### 3. Preserve the flat `NodeBatch.seed_index` contract

`materialize_batch(...)` should flatten multi-seed node contexts into a normal `NodeBatch.from_samples(...)` call. This keeps downstream tasks and models unchanged because `NodeBatch.seed_index` already represents a flat list of supervised nodes.

### 4. Keep heterogeneous behavior keyed by node type

For heterogeneous node sampling, multi-seed validation and local seed remapping stay scoped to `request.node_type`. The sampled graph still uses per-type `n_id` mappings; each emitted sample simply stores the local seed index for that node type.

## Non-goals

- No change to `NodeBatch` shape or metadata schema beyond repeating per-seed metadata entries
- No redesign of `NodeClassificationTask` or trainer counting semantics
- No change to link or temporal batch materialization

## Verification

- Focused multi-seed regressions in node materialization and node neighbor sampling tests
- Fresh full repository regression before merge
