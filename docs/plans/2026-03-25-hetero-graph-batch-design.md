# Heterogeneous GraphBatch Design

## Problem

`GraphBatch` started as a homogeneous graph-classification batch and only tracked one flat node-to-graph membership tensor: `graph_index`. That worked for `graph.x`-centric models, but it broke as soon as a graph-classification dataset contained true heterogeneous graphs. The rest of the stack could already represent hetero graphs, transfer them across devices, store them on disk, and sample them relation-aware, but the graph-classification batch contract still collapsed at collation time.

## Goals

- Preserve the current `batch.graphs` public API
- Keep homogeneous graph classification unchanged
- Add enough typed membership information for hetero pooling
- Avoid widening `Loader` or `Trainer` with graph-kind-specific branches

## Design

### 1. Preserve the homogeneous fast path

Homogeneous `GraphBatch` instances continue to expose `graph_index` and `graph_ptr` exactly as before. Existing models and tests should not need changes.

### 2. Add per-node-type membership for heterogeneous batches

Heterogeneous batches expose `graph_index_by_type` and `graph_ptr_by_type`, keyed by node type. Each tensor describes how nodes of that type map back to graphs inside `batch.graphs`. This mirrors how hetero models already think: pooling is usually performed on one node type at a time.

### 3. Keep batch transfer semantics symmetric

`GraphBatch.to()` and `GraphBatch.pin_memory()` now carry the typed membership tensors alongside the existing homogeneous tensors. That keeps trainer-managed device movement and loader pinning consistent across homo and hetero graph-classification paths.

## Non-goals

- No new batched-heterograph object replacing `batch.graphs`
- No change to `Loader` or `Trainer` public arguments
- No attempt to infer a universal single `graph_index` across hetero node types
- No redesign of graph-level label storage beyond current graph/metadata sources

## Verification

- Core GraphBatch regressions for hetero membership and sample labels
- Batch transfer regressions for typed membership tensors
- End-to-end hetero graph-classification training regression
- Full repository test suite before merge
