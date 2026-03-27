# Laplacian Sparse View Design

## Context

`vgl.ops` already exposes the main adjacency and incidence sparse surfaces:

- `adj(...)` and `adj_tensors(...)` for sparse adjacency materialization
- `adj_external(...)` for SciPy and torch sparse export
- `inc(...)` for incidence views
- degree and neighborhood queries through `in_degrees(...)`, `out_degrees(...)`, `in_edges(...)`, and `out_edges(...)`

One foundation gap still remains on the graph-ops and sparse-backend floor: there is no public Laplacian operator.

That matters because Laplacians are a basic graph primitive for spectral preprocessing, diagnostics, positional encodings, and classical graph algorithms. Without a first-class surface, callers have to rebuild degree-normalized sparse matrices manually from `adj(...)`, which is repetitive and easy to get wrong around loops, duplicate edges, and isolated nodes.

## Scope Choice

Three slices were considered:

1. Add a wide spectral package with Laplacian, normalized adjacency, eigenvalue helpers, and positional encodings.
2. Add one bounded `laplacian(...)` sparse view that matches the current `adj(...)` style and reuses the existing sparse tensor abstractions.
3. Skip a public Laplacian API and leave callers to compose it externally.

Option 2 is the right slice.

It closes the obvious primitive gap directly, fits the current `vgl.ops` graph-query surface, and keeps this batch additive. Option 1 is too wide for one batch, and option 3 leaves a clear hole in the sparse foundation.

## Recommended API

Add one public function:

- `laplacian(graph, *, edge_type=None, normalization=None, eweight_name=None, layout="coo")`

Add one matching graph method:

- `graph.laplacian(...)`

Rules:

- only square relations are supported, so the selected edge type must have matching source and destination node types
- `normalization` accepts `None`, `"rw"`, or `"sym"`
- `eweight_name` selects a 1-D per-edge scalar weight tensor; when omitted, unit weights are used
- `layout` returns a `SparseTensor` in COO, CSR, or CSC form, matching the existing `adj(...)` convention

## Semantics

### Degree Convention

`vgl.adj(...)` uses source nodes as sparse rows and destination nodes as sparse columns.

This batch keeps that convention and defines degree from the sparse row sum:

- unnormalized Laplacian: `L = D_out - A`
- random-walk normalized Laplacian: `L_rw = I - D_out^{-1} A`
- symmetric normalized Laplacian: `L_sym = I - D_out^{-1/2} A D_out^{-1/2}`

For directed graphs this yields a row-oriented Laplacian. For undirected or bidirected graphs it matches the usual structure.

### Duplicate Edges And Self-Loops

The Laplacian should aggregate repeated structural entries instead of emitting duplicate sparse coordinates.

Concretely:

- parallel edges between the same visible pair contribute one summed matrix entry
- self-loops subtract from the diagonal after the degree term is added
- the returned sparse tensor therefore contains one visible coordinate per nonzero matrix position

This keeps dense conversion predictable and avoids leaking duplicate sparse coordinates to callers.

### Isolated Nodes

Zero-degree nodes stay part of the declared node space.

Behavior:

- for `normalization=None`, isolated-node rows and columns remain all zero
- for normalized variants, isolated nodes also remain all zero instead of receiving a synthetic identity entry

That matches the sparse-first interpretation already used elsewhere in VGL and avoids undefined inverse-degree behavior.

### Storage-Backed And Heterogeneous Graphs

The new API should work for:

- homogeneous graphs
- homogeneous relations inside heterogeneous graphs
- featureless storage-backed graphs with declared node counts in the graph store

For storage-backed graphs, the returned sparse shape must still reflect the declared node space even when some nodes are absent from the visible edge list.

## Implementation Approach

Implement the operator in `vgl.ops.query`:

1. resolve and validate the selected relation
2. fetch edge endpoints in public edge order and resolve scalar edge weights
3. compute row-degree from source endpoints
4. seed diagonal entries from the degree term or normalized identity term
5. subtract edge contributions with the requested normalization
6. aggregate repeated coordinates into a single COO tensor
7. convert to CSR or CSC when requested

This keeps the feature local to the existing sparse-query layer and avoids mutating graph state or sparse caches.

## Non-Goals

- eigenvalue or eigenvector utilities
- positional-encoding layers or preprocessing pipelines
- bipartite or rectangular relation Laplacians
- multi-dimensional edge payload Laplacians
- a separate normalized adjacency API

## Testing Strategy

Add coverage for:

- unnormalized Laplacian on a weighted homogeneous graph
- random-walk and symmetric normalization
- square heterogeneous relation support
- rejection on bipartite relations and invalid normalization names
- `Graph.laplacian(...)` bridge behavior
- `vgl.ops` namespace export
- featureless storage-backed graphs preserving declared shape

Then run focused query/core/package tests followed by the full repository suite.
