# Graph Sparse Format State Design

## Context

VGL now has a usable sparse-structure stack:

- `Graph.adjacency(...)` caches internal `SparseTensor` adjacency views
- `Graph.adj(...)` exposes weighted sparse adjacency
- `Graph.adj_external(...)` exports torch / SciPy sparse matrices
- `Graph.adj_tensors(...)` exports raw structure tuples

What is still missing is the DGL-style graph format-state surface:

- `formats(...)`
- `create_formats_()`

This is not a new sparse algorithm. It is a graph-level bookkeeping API that lets users ask which sparse formats are already materialized, clone a graph with a restricted allowed-format set, and eagerly build the allowed formats before a hot path such as multi-process loading.

## Scope Choice

There were two plausible approaches:

1. Implement graph-level sparse format state only.
2. Tie the API directly to per-edge-store cache presence without explicit state.

Option 1 is the right slice. Directly deriving everything from `adjacency_cache` breaks DGL-like clone isolation: a `formats(...)` clone must be able to have its own allowed / created status even when node and edge feature tensors are shared with the base graph. That requires an explicit graph-level state model.

## Recommended API

Add one public op pair:

- `formats(graph, formats=None)`
- `create_formats_(graph)`

Add matching `Graph` methods:

- `graph.formats(...)`
- `graph.create_formats_()`

Return rules:

- `formats(None)` returns `{"created": [...], "not created": [...]}`
- `formats("csr")` or `formats(["coo", "csr"])` returns a cloned graph
- `create_formats_()` mutates the current graph's sparse-format state and returns `None`

## Semantics

Supported formats:

- `"coo"`
- `"csr"`
- `"csc"`

Initial state on a fresh graph:

- allowed formats: `("coo", "csr", "csc")`
- created formats: `("coo",)` because `edge_index` is already the native COO-like structure

`formats(selected)` should:

1. validate the requested format names
2. preserve only the requested formats as the allowed set
3. retain the intersection between requested formats and current created formats
4. if that intersection is empty, mark the highest-priority requested format as created using canonical DGL order `coo -> csr -> csc`

`create_formats_()` should:

- eagerly create all allowed but not-yet-created formats
- update graph state so `formats()` reports them as created
- return `None`

## Clone And Sharing Rules

Observed local DGL behavior shows that `formats(...)` clones do not share sparse-format state with the base graph, but node feature tensors still share storage. VGL should mirror that tradeoff:

- clone graph objects and edge stores with independent sparse-format state and adjacency caches
- reuse node and edge data mappings and underlying tensors

This keeps the API cheap while preventing one clone's eager format creation from mutating another graph's reported status.

## Implementation Approach

Add lightweight graph-level fields to `Graph`:

- allowed sparse formats
- created sparse formats

Then:

- update `Graph.adjacency(...)` so any requested layout marks that format created
- implement `formats(...)` as a graph clone with shared store data but isolated format state and isolated `adjacency_cache`
- implement `create_formats_()` by materializing each allowed format through `graph.adjacency(...)`

The clone operation should preserve graph-level metadata such as `feature_store`, `graph_store`, and schema.

## Testing Strategy

Extend coverage in:

- `tests/core/test_graph_sparse_cache.py`
- `tests/core/test_graph_ops_api.py`
- `tests/core/test_feature_backed_graph.py`
- `tests/test_package_layout.py`

Key regressions:

- fresh graphs report `coo` created and `csr` / `csc` not created
- `formats("csr")` returns a clone with isolated state
- `formats(["coo", "csr"])` retains only the requested set and keeps created intersection
- empty or invalid format selections fail clearly
- `create_formats_()` marks all allowed formats created and returns `None`
- storage-backed graphs preserve format status and eager creation
- adjacency caches on clones stay isolated from the base graph

## Non-Goals

This batch should not include:

- changing the visible structure or values of adjacency exports
- adding new sparse layouts beyond COO / CSR / CSC
- persistent on-disk sparse cache serialization
- DGL-exact exception classes or C++-style error text
