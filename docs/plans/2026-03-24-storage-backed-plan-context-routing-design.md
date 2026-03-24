# Storage-Backed Plan Context Routing Design

**Problem**

VGL can already build storage-backed graphs through `Graph.from_storage(...)`, and the public loader/trainer path can train on those graphs because lazy node and edge stores fetch from the underlying `FeatureStore` on demand. But the newer request/plan/executor path still treats feature sources as out-of-band wiring: callers must pass `feature_store=` explicitly into `Loader` or `PlanExecutor.execute(...)` if a plan contains `fetch_node_features` or `fetch_edge_features` stages. That means storage-backed graphs do not carry enough runtime context to let plan execution discover the same source automatically, which keeps large-graph flows more manual than they need to be.

**Design Goal**

Make storage-backed graphs retain their originating feature source so plan-backed execution can reuse it automatically. The change should stay additive: direct `Loader(feature_store=...)` and `PlanExecutor.execute(..., feature_store=...)` must continue to override everything, while storage-backed graphs should simply provide a safe default when no explicit source is supplied.

## Recommended Approach

Add an optional graph-level feature-source reference and set it in `Graph.from_storage(...)`. Then teach both `Loader` and `PlanExecutor` to resolve the effective feature source in this order:

1. explicit `feature_store=` argument
2. `SamplingPlan.graph.feature_store` when present
3. error only if a feature-fetch stage actually executes without either source

Keeping the fallback in both loader and executor is deliberate. Loader-level resolution makes the wiring visible and testable for plan-backed public workflows, while executor-level resolution keeps direct `PlanExecutor.execute(plan, graph=...)` calls ergonomic.

## Scope Boundaries

This batch does not redesign `SamplingPlan`, add graph-store-backed neighbor expansion, or make samplers emit new feature-fetch stages automatically. It only removes one layer of manual plumbing for cases where storage-backed graphs already know where their features live.

## Testing Strategy

Cover the change in three layers:

1. unit regression in `tests/core/test_feature_backed_graph.py` that `Graph.from_storage(...)` retains the originating feature source
2. data-loading regression in `tests/data/test_loader.py` that plan-backed loading forwards a storage-backed graph's source when no explicit `feature_store` is passed
3. integration regression in `tests/integration/test_foundation_large_graph_flow.py` that a storage-backed graph can satisfy plan feature fetch without manual wiring

Then run the focused graph/data/integration tests plus the full repository suite.
