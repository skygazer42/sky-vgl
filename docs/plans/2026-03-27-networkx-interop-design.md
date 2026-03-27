# NetworkX Interoperability Design

## Context

VGL already exposes compatibility bridges for PyG and DGL, but it still has no public path for NetworkX. That is a data-ecosystem gap rather than a model-layer gap: users can move graphs between major tensor-first frameworks, yet they cannot round-trip a graph through the most common Python graph-construction library.

This is a good next slice because the compatibility shape already exists in the codebase. `Graph` has `from_pyg(...)`, `to_pyg()`, `from_dgl(...)`, and `to_dgl()`, and `vgl.compat` already hosts adapter modules. Adding NetworkX follows the established integration pattern without widening the training or sparse-runtime surfaces.

## Scope Choice

Three slices were considered:

1. Full heterogeneous and temporal NetworkX interoperability.
2. Homogeneous directed interoperability first, with a stable additive API.
3. Export-only support.

Option 2 is the right slice.

NetworkX does not have a native heterogeneous graph model comparable to DGL heterographs, and undirected import semantics are easy to make surprising. Export-only support would leave the ecosystem bridge half-finished. A homogeneous directed slice gives VGL a useful, testable adapter now while keeping semantics explicit.

## Recommended API

Add:

- `vgl.compat.from_networkx(nx_graph)`
- `vgl.compat.to_networkx(graph)`
- `Graph.from_networkx(nx_graph)`
- `Graph.to_networkx()`

Keep the API intentionally narrow in this batch:

- import accepts `networkx.DiGraph` and `networkx.MultiDiGraph`
- export always returns `networkx.MultiDiGraph`
- heterogeneous graphs are rejected on export
- undirected NetworkX graphs are rejected on import

That keeps the public surface small and avoids silent structural reinterpretation.

## Conversion Semantics

### Export

`to_networkx(graph)` should:

- require a homogeneous graph
- create a `networkx.MultiDiGraph`
- add one node for every declared node id `0..num_nodes-1`
- attach node-aligned tensor features per node by slicing along dimension 0
- add one directed edge per stored edge in graph order
- attach edge-aligned tensor features per edge by slicing along dimension 0
- omit `edge_index` from edge attributes because it is represented structurally

Using `MultiDiGraph` is deliberate: it preserves parallel edges and stable edge iteration order.

### Import

`from_networkx(nx_graph)` should:

- reject undirected graphs
- normalize node labels into contiguous local ids using the graph’s iteration order
- build `edge_index` from the normalized directed edges
- reconstruct node and edge attributes by stacking values with the same key across nodes or edges
- require attribute values for a given key to be tensor-like and shape-compatible across all rows

This keeps the imported graph tensor-first and predictable. Missing attributes for some rows should fail clearly rather than inventing defaults.

## Non-Goals

- heterogeneous NetworkX import/export
- undirected graph auto-symmetrization
- block interoperability through NetworkX
- conversion of arbitrary Python object attributes into VGL features
- graph-level metadata round-tripping

## Testing Strategy

Add focused coverage for:

- homogeneous graph export to `networkx.MultiDiGraph`
- parallel edges surviving round-trip through `MultiDiGraph`
- import from external `DiGraph` and `MultiDiGraph`
- clear failure on heterogeneous export
- clear failure on undirected import
- package and compat exports

Then run focused compatibility/package tests plus the full repository suite.
