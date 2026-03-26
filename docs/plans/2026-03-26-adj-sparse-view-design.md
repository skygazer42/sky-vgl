# Weighted Adjacency Sparse View Design

## Context

VGL already has two adjacent structural exports:

- `Graph.adjacency(layout=...)` for sparse adjacency views
- `adj_tensors(...)` for raw COO / CSR / CSC adjacency tensors

What is still missing is the DGL-style `adj(...)` entry point. In DGL this is the obvious API for "give me the adjacency matrix", and it also supports edge-feature-backed nonzero values through `eweight_name`. Right now VGL users have to choose between unweighted `Graph.adjacency(...)` and manual sparse reconstruction when they want a weighted adjacency view.

## Scope Choice

There were three reasonable slices:

1. Add `adj(...)` as a weighted sparse-view API on top of the existing backend.
2. Rework `Graph.adjacency(...)` itself to absorb weighting and DGL naming.
3. Jump to broader sparse export or mutation work.

Option 1 is the right slice. It improves DGL migration ergonomics, adds one meaningful capability over `Graph.adjacency(...)`, and does not require breaking or redefining the existing adjacency cache contract.

## Recommended API

Add one public op:

- `adj(graph, *, edge_type=None, eweight_name=None, layout="coo")`

Add one matching `Graph` method:

- `graph.adj(...)`

This keeps the DGL-style name while allowing one VGL-specific extension: `layout`. DGL's native `adj(...)` always returns its backend sparse matrix format, while VGL already has a stable COO / CSR / CSC sparse abstraction. Keeping `layout="coo"` by default preserves the DGL mental model without giving up VGL's layout control.

## Semantics

Return type:

- always `vgl.sparse.SparseTensor`

Value rules:

- if `eweight_name` is omitted, the nonzero values should behave as all ones
- if `eweight_name` is provided, use that edge feature tensor as the sparse values

Ordering rules:

- visible COO row / col order should follow public `e_id` when present
- CSR / CSC compressed order should preserve that public edge order within each row or column bucket

Shape rules:

- homogeneous relations use `(num_nodes, num_nodes)`
- heterogeneous relations use `(num_src_nodes, num_dst_nodes)`
- storage-backed declared node counts must flow through to shape and pointer lengths

## Implementation Approach

The query layer already has the right primitives:

- `_resolve_edge_type(...)`
- `_ordered_edge_tensors(...)`
- `_compress_edge_tensors(...)`
- `_normalize_sparse_layout(...)`

The new op should:

1. resolve the relation
2. fetch ordered edge endpoints in public-`e_id` order
3. choose sparse values from either `eweight_name` or implicit ones
4. build `SparseTensor` directly in COO / CSR / CSC so compressed layouts keep the same stable ordering model as `adj_tensors(...)`

This should stay independent from `Graph.adjacency(...)` caching. The existing cache is unweighted and keyed only by layout, so reusing it would either ignore weights or blur cache semantics.

## Testing Strategy

Extend coverage in:

- `tests/ops/test_query_ops.py`
- `tests/core/test_graph_ops_api.py`
- `tests/core/test_feature_backed_graph.py`
- `tests/test_package_layout.py`

Key regressions:

- unweighted `adj(...)` returns a `SparseTensor` with the expected dense form
- weighted `adj(..., eweight_name=...)` propagates edge values
- COO ordering follows public `e_id`
- CSR / CSC preserve public edge order within compressed buckets
- heterogeneous relation selection works
- missing `eweight_name` fails clearly
- featureless storage-backed graphs preserve declared shape / pointer lengths
- `Graph` bridge and `vgl.ops` exports stay stable

## Non-Goals

This batch should not include:

- changing `Graph.adjacency(...)` cache semantics
- torch-native sparse tensor export
- SciPy export
- adjacency mutation APIs
- new sparse layouts beyond COO / CSR / CSC
