# Sparse Runtime Expansion Design

## Goal

Deepen `vgl.sparse` from a minimal adjacency helper into a more complete sparse runtime layer that better matches the low-level capabilities users expect from DGL-style graph systems. The next slice focuses on public CSC support, sparse transpose, column-oriented structural selection, and lightweight reduction primitives.

## Why This Next

The previous foundation rollout created the package boundaries and minimal sparse tensor model, but the runtime is still skewed toward row-oriented COO/CSR flows. DGL-class graph systems rely on both row- and column-oriented sparse views because in-neighbor traversal, transpose-based message flow, and layout-aware graph storage all build on them. This is a better next step than remote distributed execution because it is lower-risk, directly reusable by loaders/stores, and strengthens the substrate that distributed sampling will eventually consume.

## Scope

This phase keeps `SparseTensor` as the only tensor container and expands around it:

- make CSC conversion a public API
- add `transpose(sparse)` that swaps shape and indices while preserving values
- add `select_cols(sparse, cols)` as the column-dual of `select_rows`
- add a simple additive reduction primitive over rows/cols and let `degree` build on it
- document and test the new public surface

## Non-Goals

This phase does not attempt to add fused kernels, autograd-specialized sparse ops, remote graph stores, multi-process distributed execution, or a full GraphBolt-style runtime. It also does not redesign `Graph` or `Loader`.

## Design Notes

All new ops should accept any current layout by normalizing through `to_coo()` or layout converters first, then return deterministic outputs. `transpose()` should preserve layout where possible by swapping CSR<->CSC and transposing COO directly. Structural selectors should return reindexed sparse tensors over the chosen axis. Reductions should respect `values` when present and otherwise behave like unweighted edge counts.
