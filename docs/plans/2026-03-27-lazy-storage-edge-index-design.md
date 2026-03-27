# Lazy Storage-Backed Edge Index Design

## Context

`Graph.from_storage(...)` is already lazy for node and non-structural edge features, but it still eagerly resolves `edge_index` inside `EdgeStore.from_storage(...)`. That makes storage-backed graphs partially eager at construction time, even when the underlying `graph_store` could have served structure lazily or from metadata-backed stores.

This is now the main blocker for turning partition-backed graphs and future on-disk graph stores into truly lazy large-graph substrates. We already made distributed store metadata paths lazy for feature shapes, edge types, and node counts; `edge_index` materialization is the next structural hotspot.

## Goals

- Make `Graph.from_storage(...)` avoid `graph_store.edge_index(...)` during construction
- Load `edge_index` only on first structural access and cache it afterward
- Keep the public `Graph`, `EdgeStore`, and `edata` APIs unchanged
- Preserve current behavior for adjacency, graph transforms, and batch paths once structure is touched

## Approach

Keep `EdgeStore` on top of `LazyFeatureMap`, but stop treating `edge_index` as an eagerly populated value. Instead, register it as another cached loader.

That means:

- construction stays metadata-only
- the first `graph.edge_index`, `graph.edges[etype].edge_index`, or adjacency build triggers one `graph_store.edge_index(...)`
- later accesses reuse the cached tensor

For non-structural edge features, defer the `graph_store.edge_count(...)` lookup into each loader closure as well. That way graph construction performs zero graph-store structure calls, not “zero edge-index calls but still one edge-count call”.

## Non-Goals

- No `LocalGraphShard` refactor in this batch
- No new graph-store backend type in this batch
- No partition payload format rewrite in this batch

## Verification

- Regression proving `Graph.from_storage(...)` does not touch `graph_store.edge_index(...)` during construction
- Regression proving first `edge_index` access loads once and caches
- Regression proving adjacency materialization still works through the lazy structure path
- Fresh focused and full `pytest` verification before merge
