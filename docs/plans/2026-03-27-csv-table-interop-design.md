# CSV Table Interoperability Design

## Context

VGL now supports homogeneous in-memory edge-list interoperability plus CSV edge-list file I/O. That gives the project a minimal structural handoff, but it still leaves a practical data-ecosystem gap: there is no file-backed path that carries node features and isolated nodes through explicit node records.

Many preprocessing pipelines do not start from one edge-only file. They use a pair of small tables:

- `nodes.csv` for node ids and node attributes
- `edges.csv` for source, destination, and edge attributes

Without that paired-table surface, VGL can persist edge structure and edge features, but not the common tabular shape that preserves node attributes and isolated nodes directly.

## Scope Choice

Three slices were considered:

1. Add pandas/DataFrame interoperability now.
2. Add homogeneous paired CSV table import/export using only the standard library.
3. Design a larger directory-backed tabular dataset format.

Option 2 is the right slice.

It keeps dependencies unchanged, composes with the new edge-list CSV substrate, and delivers one practical interoperability format immediately. Pandas would widen the dependency and optional-feature story too early, while a directory-backed dataset format is a larger product decision than this next additive substrate step.

## Recommended API

Add:

- `vgl.compat.from_csv_tables(nodes_path, edges_path, *, node_id_column="node_id", src_column="src", dst_column="dst", node_columns=None, edge_columns=None, delimiter=",")`
- `vgl.compat.to_csv_tables(graph, nodes_path, edges_path, *, node_id_column="node_id", src_column="src", dst_column="dst", node_columns=None, edge_columns=None, delimiter=",")`
- `Graph.from_csv_tables(...)`
- `graph.to_csv_tables(...)`

Keep the scope intentionally narrow:

- homogeneous graphs only
- paired node and edge CSV files only
- integer public node ids only
- numeric scalar node and edge feature columns only
- `node_id` rows define the full public node space, so isolated nodes survive naturally

## Semantics

### Import

`from_csv_tables(...)` should:

- require a header row in both files
- require one node id column in `nodes.csv`
- require `src` / `dst` columns in `edges.csv`
- parse node ids, sources, and destinations as integers
- require node ids to be unique
- require every edge endpoint to appear in the node table
- infer node and edge feature columns automatically when explicit column lists are omitted
- parse feature columns conservatively as numeric tensors
- map public node ids into internal contiguous positions for `edge_index`
- store the public ids as `n_id`

This means import can preserve non-contiguous external node ids such as `10, 20, 30` while still building a standard internal `edge_index`.

### Export

`to_csv_tables(...)` should:

- require a homogeneous graph
- write `nodes.csv` with one row per node
- write `edges.csv` with one row per edge in stored edge order
- prefer public node ids from `graph.ndata["n_id"]` when present and rank-1 aligned to node count
- otherwise fall back to `0..num_nodes-1`
- infer exportable node and edge feature columns when explicit column lists are omitted
- only export scalar numeric feature tensors aligned to node or edge count

Multi-dimensional features remain out of scope in this batch. Export should fail clearly instead of inventing column-expansion or JSON encoding rules.

## Non-Goals

- heterogeneous graph table conventions
- string node ids
- pandas / DataFrame adapters
- compressed or remote files
- multi-dimensional feature flattening
- dataset manifests or directory-scoped dataset loaders

## Testing Strategy

Add focused coverage for:

- round-tripping a homogeneous graph through paired node/edge CSV tables
- preserving non-contiguous public node ids through `n_id`
- custom column names and custom delimiters
- preserving isolated nodes through explicit node rows
- rejection when an edge references an unknown node id
- rejection of non-numeric feature columns
- rejection of heterogeneous export
- compat and package export wiring

Then run focused compat/package tests and the full suite.
