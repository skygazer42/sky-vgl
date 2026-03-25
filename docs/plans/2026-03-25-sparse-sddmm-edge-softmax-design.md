# Sparse SDDMM And Edge Softmax Design

## Context

`vgl.sparse` already covers layout containers, COO/CSR/CSC conversion, transpose, structural selection, reductions, and sparse-dense matmul. It still lacks two substrate primitives that appear repeatedly in DGL-class graph runtimes:

- sampled dense-dense multiplication over a sparse structure
- edge-wise softmax normalized over a structural axis

The codebase already contains a private `edge_softmax(...)` helper in `vgl.nn.conv._homo`, but that keeps an important sparse primitive buried inside one model helper module and prevents the sparse backend from owning the behavior directly.

## Scope Choice

Three plausible next slices were considered:

1. Add only a public `edge_softmax(...)` and leave `sddmm(...)` for later.
2. Add both `sddmm(...)` and `edge_softmax(...)` to `vgl.sparse`, then refactor `_homo` to delegate to the new public primitive.
3. Jump further into fused sparse kernels such as generalized SPMM/SDDMM operators and segment reductions.

Option 2 is the right batch. It closes a clear sparse-backend gap, gives the package a more credible low-level operator surface, and stays compact enough to verify with focused sparse tests plus the existing convolution regression suite.

## Recommended Design

Add two public operators to `vgl.sparse.ops` and export them through `vgl.sparse`:

- `sddmm(sparse, lhs, rhs)`
- `edge_softmax(sparse, scores, *, dim=1)`

`sddmm(...)` should treat the sparse tensor as the sampling structure, gather `lhs` by sparse row indices and `rhs` by sparse column indices, and compute a dot product across the last feature dimension. For rank-2 inputs, the output is one scalar per edge. For higher-rank inputs such as `(num_nodes, heads, channels)`, the dot product should reduce only the last dimension so the result stays edge-aligned with any head dimension preserved.

`edge_softmax(...)` should accept any current sparse layout, normalize scores over the chosen sparse axis, and return a dense tensor aligned with the sparse storage order. The default should be `dim=1` to preserve current message-passing semantics, where incoming edges to the same destination node are normalized together.

## Layout And Ordering Semantics

The new operators should accept COO, CSR, and CSC inputs by normalizing through `to_coo()` for index access. They should preserve the caller-visible storage order:

- `edge_softmax(...)` returns dense values in the current sparse storage order
- `sddmm(...)` returns a new `SparseTensor` with the same layout, shape, and structural indices as the input, with only `values` replaced

For compressed layouts, the simplest stable behavior is to keep the original pointer/index arrays and write back the computed values without re-sorting.

## Validation Rules

`sddmm(...)` should fail early when:

- `lhs` or `rhs` has zero rank
- `lhs.size(0)` does not match the sparse row count
- `rhs.size(0)` does not match the sparse column count
- the feature dimensions needed for the dot product do not line up

`edge_softmax(...)` should fail early when:

- `dim` is not `0` or `1`
- `scores.size(0)` does not match sparse `nnz`

Empty sparse tensors should remain valid inputs for both operators.

## Integration Plan

Keep `vgl.nn.conv._homo.edge_softmax(...)` as a compatibility wrapper for the current convolution modules, but reimplement it by constructing a small COO `SparseTensor` and calling `vgl.sparse.edge_softmax(...)`. That keeps the public model API unchanged while moving the primitive into the sparse backend where it belongs.

## Testing Strategy

Add focused coverage in `tests/sparse/test_sparse_ops.py` for:

- `sddmm(...)` on COO inputs with expected sampled dot products
- layout and shape preservation for compressed sparse inputs
- `edge_softmax(...)` normalization over destination groups
- empty sparse inputs
- input-length validation failures

Then extend `tests/test_package_layout.py` so the new sparse exports are part of the stable namespace. Existing convolution tests will provide regression coverage for the `_homo.edge_softmax(...)` wrapper once the full suite runs.

## Non-Goals

This batch should not include:

- generalized binary operators such as add/sub/mul/div variants for SDDMM
- fused sparse kernels or custom CUDA paths
- graph-level `edge_softmax(graph, logits)` convenience APIs
- generalized segment reductions
- distributed sparse execution changes
