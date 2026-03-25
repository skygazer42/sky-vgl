# Sparse Multi-Value Support Design

## Context

`vgl.sparse` now exposes the basic runtime surface users expect for adjacency-centric work: COO/CSR/CSC layouts, sparse-dense matmul, sampled dense-dense matmul, and edge softmax. One structural limitation still keeps the backend thinner than DGL-class sparse systems: `SparseTensor.values` is effectively scalar-only.

That limitation leaks into multiple places:

- `SparseTensor` validation requires `values.numel() == nnz`
- reductions flatten edge values instead of preserving trailing dimensions
- `sddmm(...)` can only return scalar edge scores
- multi-head edge softmax works on dense tensors, but the sparse container still cannot hold multi-head edge values

This is a substrate issue, not a model-specific issue. Fixing it strengthens the sparse layer directly and makes later generalized sparse ops more plausible.

## Scope Choice

Three plausible next slices were considered:

1. Add generalized GSPMM/GSDDMM immediately.
2. First make `SparseTensor` support multi-dimensional edge values and extend the existing sparse ops that naturally fit that model.
3. Ignore the sparse container and add more graph operations first.

Option 2 is the right batch. It is smaller than a full generalized sparse kernel surface, but it removes the key representation bottleneck that currently blocks multi-head sparse values and cleaner backend evolution.

## Recommended Design

Keep `SparseTensor` rank-2 structurally, but allow `values` to have shape `(nnz, ...)` instead of only `(nnz,)`.

Update validation so:

- `values` may be rank-1 or higher
- `values.shape[0]` must match sparse `nnz`
- the sparse structure still owns only the first edge axis; trailing dimensions are payload

Then extend sparse behavior accordingly:

- `to_coo()`, `to_csr()`, and `to_csc()` preserve trailing value dimensions while reordering edge entries
- `select_rows(...)`, `select_cols(...)`, and `transpose(...)` preserve multi-value payloads unchanged except for edge reindexing
- `sum(...)` and `degree(...)` reduce over the edge axis while preserving trailing payload dimensions
- `sddmm(...)` reduces only the last feature dimension, so inputs like `(num_nodes, heads, channels)` return sparse values with shape `(nnz, heads)`
- `edge_softmax(...)` already accepts extra trailing score dimensions, but should keep that behavior explicit in tests

## Explicit Non-Extension

`spmm(...)` should stay scalar-weighted in this batch. Multi-dimensional sparse values create ambiguous semantics for the current API because the dense input is only `(num_cols, features)`. Rather than silently broadcasting or guessing a contract, `spmm(...)` should fail early when sparse values are not scalar per edge.

## Validation Rules

`SparseTensor.values` should fail early when:

- `values.ndim == 0`
- `values.shape[0] != nnz`

`sddmm(...)` should additionally validate:

- matching node counts on axis 0
- matching payload prefix dimensions except for the final feature axis
- matching final feature axis size

`sum(...)` and `degree(...)` should still accept value-less sparse tensors and return scalar counts.

## Testing Strategy

Add focused coverage for:

- multi-dimensional value acceptance in `SparseTensor`
- rejection when the leading value dimension does not match `nnz`
- conversion round-trips preserving trailing payload dimensions
- row/column selection and transpose preserving multi-value payloads
- reduction over the sparse axis while keeping trailing payload dimensions
- multi-head `sddmm(...)`
- multi-head `edge_softmax(...)`
- `spmm(...)` rejecting multi-dimensional sparse values

Then run the full suite to make sure current convolution and graph-op paths still pass.

## Non-Goals

This batch should not include:

- generalized GSPMM/GSDDMM APIs
- multi-dimensional sparse matrix multiplication semantics
- custom sparse autograd kernels
- distributed sparse execution changes
