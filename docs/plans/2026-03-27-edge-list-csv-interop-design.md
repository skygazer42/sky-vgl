# Edge-List CSV Interoperability Design

## Context

VGL now has an in-memory homogeneous edge-list bridge through `Graph.from_edge_list(...)`, `graph.to_edge_list()`, and the matching `vgl.compat` helpers. That closes the lowest-level structural interoperability gap inside one Python process, but it still does not give users a durable file format handoff.

This is still a practical gap in the data ecosystem layer. Small graph imports, preprocessing outputs, shell-generated datasets, and quick experiments often start as CSV or delimiter-separated edge tables before they graduate into richer dataset containers. Without a file-backed edge-list path, VGL has an in-memory substrate but no minimal persistence story on top of it.

## Scope Choice

Three slices were considered:

1. Full pandas/DataFrame-based tabular import/export immediately.
2. Standard-library CSV/tabular file I/O built on the new in-memory edge-list adapter.
3. Multi-file graph-table ingestion with separate node and edge tables.

Option 2 is the right slice.

It keeps dependencies unchanged, composes directly with the existing edge-list adapter, and delivers one useful persistence format now. Option 1 would add a new optional dependency surface before the core semantics are proven. Option 3 is valuable, but it is a broader dataset-ingestion problem rather than the next minimal extension of the new edge-list substrate.

## Recommended API

Add:

- `vgl.compat.from_edge_list_csv(path, *, src_column="src", dst_column="dst", edge_columns=None, delimiter=",", num_nodes=None)`
- `vgl.compat.to_edge_list_csv(graph, path, *, src_column="src", dst_column="dst", edge_columns=None, delimiter=",")`
- `Graph.from_edge_list_csv(...)`
- `graph.to_edge_list_csv(...)`

Keep the scope intentionally narrow:

- homogeneous graphs only
- path-like file input/output only in this batch
- integer endpoint ids only
- optional edge feature columns are numeric and become tensors
- explicit `num_nodes` still preserves isolated nodes on import
- export writes one header row plus one row per edge in stored edge order

## Semantics

### Import

`from_edge_list_csv(...)` should:

- read a delimiter-separated file with a header row
- require `src_column` and `dst_column`
- parse endpoint ids as integers
- optionally collect edge-feature columns into tensors
- infer edge feature columns automatically when `edge_columns is None`, excluding the source and destination columns
- preserve explicit `num_nodes` by delegating the final graph construction to `from_edge_list(...)`
- return an empty graph cleanly when the file only contains the header

Column parsing should stay conservative:

- integer-looking columns become `torch.long`
- otherwise numeric columns become floating tensors
- mixed or non-numeric feature columns fail clearly

### Export

`to_edge_list_csv(...)` should:

- require a homogeneous graph
- write one row per visible edge in stored edge order
- write `src` / `dst` columns first, followed by any selected edge feature columns
- infer exportable edge feature columns when `edge_columns is None` by selecting tensor edge features aligned to edge count, excluding `edge_index`
- serialize scalar integer and floating values without inventing nested table formats

Multi-dimensional edge features are out of scope for this batch. Export should fail clearly when a selected edge feature is not rank-1 or not edge-aligned.

## Non-Goals

- pandas or DataFrame adapters
- separate node-table CSV import/export
- arbitrary string node labels
- heterogeneous graph CSV conventions
- nested / JSON-encoded edge feature payloads
- compression, remote storage, or on-disk dataset manifests

## Testing Strategy

Add focused coverage for:

- round-tripping a homogeneous graph through CSV with numeric edge features
- custom column names and custom delimiters
- explicit `num_nodes` preserving isolated nodes when importing CSV
- empty header-only CSV import
- rejection of mixed or non-numeric feature columns
- rejection of heterogeneous export
- compat and package export wiring

Then run focused compat/package tests and the full suite.
