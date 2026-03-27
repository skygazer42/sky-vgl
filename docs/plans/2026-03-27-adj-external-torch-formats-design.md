# External Adjacency Torch Format Export Design

## Context

`Graph.adj_external(...)` already exports adjacency into two external families:

- native PyTorch sparse COO tensors by default
- SciPy `coo` / `csr` matrices when `scipy_fmt=` is selected

That was the right first slice, but it leaves one narrow sparse-backend gap after the new `vgl.sparse` <-> `torch.sparse_*` interoperability layer landed: callers still cannot ask `adj_external(...)` for native PyTorch CSR or CSC adjacency directly.

That matters because graph-level export is the main high-level escape hatch for downstream sparse tooling. Once VGL can already convert internal sparse layouts into `torch.sparse_csr_tensor` and `torch.sparse_csc_tensor`, keeping `adj_external(...)` locked to COO means the graph surface still does not expose the native compressed layouts PyTorch users actually want to feed into row- or column-oriented sparse kernels.

## Scope Choice

Three slices were considered:

1. Change the default torch export from COO to compressed layouts opportunistically.
2. Add an explicit torch-format selector while preserving the current default.
3. Redesign `adj_external(...)` around one generic external-format enum for both torch and SciPy.

Option 2 is the right slice.

It is additive, keeps current callers stable, and maps cleanly onto the new low-level torch sparse conversion helpers. Option 1 would be a behavior break, and option 3 would widen API churn for little gain.

## Recommended API

Extend the public op to:

- `adj_external(graph, transpose=False, *, scipy_fmt=None, torch_fmt=None, edge_type=None)`

Keep the matching `Graph` bridge:

- `graph.adj_external(...)`

Rules:

- when both `scipy_fmt` and `torch_fmt` are `None`, preserve the current default and return `torch.sparse_coo_tensor`
- when `torch_fmt` is one of `"coo"`, `"csr"`, or `"csc"`, return the matching native PyTorch sparse tensor layout
- when `scipy_fmt` is one of `"coo"` or `"csr"`, keep the current SciPy behavior
- reject calls that set both `scipy_fmt` and `torch_fmt`

## Semantics

### Torch Export

For torch exports:

- `"coo"` should preserve visible public-`e_id` COO order exactly
- `"csr"` should compress rows in canonical row order while preserving public edge order within each row bucket
- `"csc"` should compress columns in canonical column order while preserving public edge order within each column bucket
- values remain explicit ones, matching current `adj_external(...)` behavior

### Shape and Orientation

- default orientation is `(num_src_nodes, num_dst_nodes)`
- `transpose=True` swaps both orientation and shape
- featureless storage-backed declared node counts must still carry through
- heterogeneous relation selection should still work with the same `edge_type=` path

## Implementation Approach

The new torch sparse export can stay local to `adj_external(...)`:

1. resolve the edge type
2. fetch ordered endpoints in public-`e_id` order
3. optionally transpose endpoints and shape
4. if `torch_fmt` is requested, build a `SparseTensor` with unit values in the requested layout and hand it to `vgl.sparse.to_torch_sparse(...)`
5. otherwise preserve the existing torch-COO default or SciPy path

This keeps graph export logic thin and reuses the new sparse-backend interop instead of duplicating compressed-layout construction inside query code.

## Non-Goals

- weighted external adjacency export
- SciPy CSC export
- changes to `adj(...)`, `adj_tensors(...)`, or graph sparse-format state
- new graph methods beyond the existing `adj_external(...)` bridge

## Testing Strategy

Add focused coverage for:

- `torch_fmt="csr"` and `"csc"` native exports
- coexistence with the existing default COO behavior
- validation that `scipy_fmt` and `torch_fmt` cannot be combined
- heterogeneous relation selection with compressed torch export
- featureless storage-backed graphs preserving declared node space under compressed export
- `Graph.adj_external(...)` bridge forwarding the new keyword

Then run focused ops/core tests and the full suite.
