# Plan Feature Materialization Design

**Problem**

VGL can now route `fetch_node_features` and `fetch_edge_features` stages through either a direct `FeatureStore` or a coordinator-backed distributed source, but the fetched payloads stop in executor state. `materialize_context(...)` rebuilds sampled subgraphs from `node_ids` alone, so fetched tensors never override or populate the resulting `SampleRecord.graph`. That leaves plan-backed feature fetch effectively disconnected from the public node-sampling path, and it is why recent batches explicitly stopped short of making samplers emit feature-fetch stages automatically.

**Design Goal**

Make node-sampling plans able to fetch node and edge features through the plan runtime and materialize those tensors back into the sampled subgraph, while preserving current default behavior when no fetch stages are present. This should also give `NodeNeighborSampler` an opt-in way to append those fetch stages automatically so storage-backed and routed feature sources can participate in large-graph node sampling without custom wrapper code.

## Recommended Approach

Keep the current stage names and `feature_store=` surface intact, but tighten the contract between executor and materializer:

1. extend `expand_neighbors` to record induced edge ids alongside expanded node ids
2. let feature-fetch stages resolve typed indices from either tensor state entries (`node_ids`, `edge_ids`) or typed dict entries (`node_ids_by_type`, `edge_ids_by_type`)
3. keep writing fetched slices to the requested `output_key`, but also accumulate them under reserved materialization state keyed by node type / edge type
4. update node-context materialization to align those fetched slices with the selected `n_id` / `e_id` order and overlay them onto the rebuilt subgraph
5. add opt-in `node_feature_names` and `edge_feature_names` configuration to `NodeNeighborSampler` so it can append the matching fetch stages automatically

This stays additive. Existing plans that never add fetch stages behave exactly as they do today, while custom plans and new sampler options finally make routed feature fetch visible in the public batch output.

## Scope Boundaries

This batch stays focused on node-sampling materialization. It does not:

- redesign link or temporal materialization
- add cross-partition graph stitching
- change the meaning of the public `feature_store=` argument
- make every sampler auto-fetch features by default

## Testing Strategy

Cover the change in four layers:

1. executor regressions that `expand_neighbors` records induced edge ids for homo and hetero contexts
2. feature-fetch regressions that typed dict state entries can drive node/edge fetch stages
3. node-sampler regressions that fetched node/edge tensors override the materialized subgraph in homo and hetero cases
4. full-suite verification plus docs updates describing the new opt-in sampler fetch path
