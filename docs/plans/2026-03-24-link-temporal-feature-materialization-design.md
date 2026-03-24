# Link and Temporal Feature Materialization Design

## Problem

Plan-backed feature fetch had already been wired through executor state and node-sample materialization, but sampled link-prediction and temporal-event flows still stopped at the local subgraph extracted by their custom sampling stages. That left a public API gap: `LinkNeighborSampler` and `TemporalNeighborSampler` could build sampled graphs, but there was no opt-in way to rehydrate those graphs from an external feature store without custom sampler subclasses or ad-hoc post-processing.

## Goals

- Preserve the existing sampling-plan / executor contract
- Keep default sampler behavior unchanged
- Reuse the same reserved materialization state used by node sampling
- Support both direct `sample(...)` and loader-driven batch materialization

## Design

### 1. Reuse stage-local sampled graph ids

`sample_link_neighbors` and `sample_temporal_neighbors` already produce sampled subgraphs whose public `n_id` / `e_id` fields point back to global ids. The executor now records those ids into `context.state` immediately after the sampling stage runs. That lets the existing generic `fetch_node_features` / `fetch_edge_features` stages work unchanged.

### 2. Mirror the node sampler opt-in surface

`LinkNeighborSampler` now owns the same `node_feature_names=...` / `edge_feature_names=...` configuration pattern as `NodeNeighborSampler`. Homogeneous link and temporal samplers accept flat feature-name iterables; heterogeneous link sampling accepts dictionaries keyed by node type and edge type. `TemporalNeighborSampler` forwards the same options to its link-sampler base.

### 3. Materialize onto record graphs, not new batch-specific side channels

Link prediction and temporal event APIs are centered on `LinkPredictionRecord.graph` and `TemporalEventRecord.graph`, so fetched features should appear there. Materialization rebuilds the sampled graph with fetched tensors overlaid in `n_id` / `e_id` order before records are returned directly or collated into batches. This keeps models oblivious to whether features came from the source graph or an external store.

## Non-goals

- No new automatic feature fetch defaults
- No change to negative-sampling semantics
- No temporal hetero sampling redesign
- No cross-partition stitching beyond the existing routed feature-source support

## Verification

- Focused sampler regressions covering homogeneous / heterogeneous link sampling and direct / loader temporal sampling
- Adjacent node/materialization/fetch regressions to guard the already-landed node path
- Full repository test suite before merge
