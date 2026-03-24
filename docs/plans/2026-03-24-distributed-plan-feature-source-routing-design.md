# Distributed Plan Feature Source Routing Design

**Problem**

VGL now has local partition shards, a `LocalSamplingCoordinator`, and explicit `fetch_node_features` / `fetch_edge_features` plan stages. But those pieces do not meet in the plan runtime. `PlanExecutor` only understands one narrow `feature_store.fetch(key, index)` shape, while `LocalSamplingCoordinator` exposes typed `fetch_node_features(...)` and `fetch_edge_features(...)` entry points. `Loader` also does not forward any configured feature source into `executor.execute(...)`, so plan-backed feature fetch only works when a caller manually invokes `PlanExecutor.execute(..., feature_store=...)`. That leaves the distributed substrate disconnected from the request/plan/executor path that is supposed to underpin larger-graph training.

**Design Goal**

Make plan-backed feature fetch stages resolve against either a storage-backed feature store or a coordinator-backed routed feature source without changing existing stage names or request contracts. The result should stay additive: current direct `FeatureStore` behavior must keep working, and new coordinator-backed fetch should only activate when the caller passes a compatible source.

## Recommended Approach

Use one narrow routing layer inside `PlanExecutor._fetch_features(...)` instead of introducing a new abstraction family. The executor will inspect the configured feature source and choose the appropriate call shape:

- if the source exposes `fetch(key, index)`, keep the current behavior
- if the stage is node-scoped and the source exposes `fetch_node_features(key, index)`, use that
- if the stage is edge-scoped and the source exposes `fetch_edge_features(key, index)`, use that

This keeps the existing stage model intact, avoids wrapping the coordinator in a fake `FeatureStore`, and preserves future room for remote backends that may expose either protocol.

## Loader Integration

`Loader` should accept an optional `feature_store` argument and pass it into `executor.execute(...)` whenever it resolves a `SamplingPlan`. This is intentionally named the same as the current executor argument to stay source-compatible with tests and call sites, even though the runtime will now accept either a real feature store or a coordinator-like routed source. Existing loaders that do not specify a feature source should behave exactly as they do today.

## Testing Strategy

Add three layers of coverage:

1. Extend `tests/data/test_feature_fetch_stage.py` with a coordinator-backed regression using a partitioned hetero graph and `LocalSamplingCoordinator`.
2. Extend `tests/data/test_loader.py` with a lightweight fake executor that proves `Loader(feature_store=...)` forwards the configured source while resolving a `SamplingPlan`.
3. Run the focused data/distributed tests plus the full repository suite.

This is a deliberately small bridge step. It does not yet make built-in samplers emit feature fetch stages automatically, and it does not redesign materialization to surface fetched feature payloads in batches. Those can build on top once the executor path can route through distributed feature sources at all.
