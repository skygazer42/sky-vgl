# Degree Query Ops Design

## Context

`vgl.ops.query` already covers edge identity lookups (`find_edges(...)`, `edge_ids(...)`, `has_edges_between(...)`) plus ordered adjacency views (`in_edges(...)`, `out_edges(...)`, `predecessors(...)`, `successors(...)`). The next DGL-style gap directly above that layer is degree inspection: callers still cannot ask for inbound or outbound degree counts through the public graph API, either for one node, many nodes, or the entire declared node space.

That gap matters for several of the broader bottom-layer goals the project is chasing. Degree queries are a small but important substrate for sampler heuristics, neighborhood filtering, block construction diagnostics, sparse operator validation, and DGL adapter parity. They are also a clean follow-on batch because the adjacency-query work already established stable node validation and storage-backed node-count handling.

## Scope Choice

There were three realistic slice options:

1. Fold degree queries into a larger neighborhood-inspection batch with more set operations.
2. Add only `in_degrees(...)` and `out_degrees(...)` now, on top of the new adjacency-query substrate.
3. Skip degree queries and jump straight to larger training or sparse-backend work.

Option 2 is the right slice. Degree queries are coherent, high leverage, and easy to verify in isolation. Pulling in broader neighborhood or sparse-backend work would expand the surface area too much for one batch and make DGL parity progress harder to audit.

## Recommended API

Add two public ops:

- `in_degrees(graph, v=None, *, edge_type=None)`
- `out_degrees(graph, u=None, *, edge_type=None)`

Add matching `Graph` convenience methods:

- `graph.in_degrees(...)`
- `graph.out_degrees(...)`

The API should follow the existing VGL relation selection convention: if `edge_type` is omitted, use the default or only edge type; otherwise require the canonical edge-type tuple.

## DGL-Aligned Semantics

These operators should mirror the common DGL behavior:

- if no node ids are provided, return degrees for the full declared node space of the relevant endpoint type
- if a single scalar node id is provided, return a Python `int`
- if a tensor, list, or other iterable of node ids is provided, return a one-dimensional `torch.Tensor`
- duplicate node ids in the query should be preserved in the output order

Unlike adjacency edge queries, degree queries never expose `e_id`; they only count stored edges incident to the selected endpoint.

## Validation And Storage Context

Degree queries must validate requested node ids against `graph._node_count(...)`, not only against the maximum id observed in `edge_index`. That is essential for featureless storage-backed graphs where isolated-but-declared nodes exist in `graph_store` but never appear in the relation's edge list.

The “all nodes” case should also use `graph._node_count(...)` to size the returned tensor, so a graph with four declared nodes and only two connected ones still returns a length-four degree vector with zeros in the isolated positions.

Empty node selections should return empty tensors on the relation's device.

## Implementation Shape

The implementation can stay inside `vgl.ops.query` and reuse the current helper stack:

- resolve the relation through `_resolve_edge_type(...)`
- normalize optional node ids into either scalar or vector form
- validate ids with `_validate_node_ids(...)`
- count incident edges by comparing `store.edge_index[0]` or `store.edge_index[1]` against the requested nodes

For the full-node-space path, `torch.bincount(...)` with `minlength=graph._node_count(...)` is the simplest fit. For selected-node queries, reuse the full count vector and gather from it, which keeps the behavior consistent and naturally preserves duplicates in the request order.

## Testing Strategy

Extend coverage in:

- `tests/ops/test_query_ops.py` for homogeneous, heterogeneous, scalar, vector, and all-node queries
- `tests/core/test_graph_ops_api.py` for `Graph.in_degrees(...)` and `Graph.out_degrees(...)`
- `tests/core/test_feature_backed_graph.py` for featureless storage-backed graphs with declared isolated nodes
- `tests/test_package_layout.py` for stable `vgl.ops` exports

Key regressions:

- scalar input returns `int`
- omitted node ids return full-length degree tensors
- heterogeneous relations count against the correct source or destination node type
- invalid node ids raise `ValueError`
- isolated declared nodes return zero degree instead of failing or disappearing

## Non-Goals

This batch should not include:

- unique-neighbor degree semantics
- weighted degree accumulation
- mutation APIs such as `add_edges(...)`
- sampler changes
- sparse-kernel rewrites
- broader graph-statistics helpers such as `num_edges(...)` or `in_neighbors(...)`
