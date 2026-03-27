# HeteroBlock DGL Interoperability Design

## Context

VGL now has:

- relation-local `Block` with `Block.to_dgl()` / `Block.from_dgl(...)`
- multi-relation `HeteroBlock`
- sampler paths that can now emit `HeteroBlock` for heterogeneous node/link workloads

The remaining compatibility gap is that the DGL adapter still stops at single-relation blocks. `vgl.compat.block_from_dgl(...)` rejects multi-relation DGL blocks, `Block.from_dgl(...)` cannot import them, and `HeteroBlock` has no DGL bridge at all.

That leaves an obvious parity hole relative to DGL’s hetero block workflows: VGL can now produce multi-relation block layers internally, but callers cannot round-trip those layers through DGL.

## Scope Choice

Three API shapes were considered:

1. Widen `block_from_dgl(...)` / `block_to_dgl(...)` to accept and return either `Block` or `HeteroBlock`.
2. Keep `Block` compatibility APIs strict and add explicit `hetero_block_from_dgl(...)` / `hetero_block_to_dgl(...)`.
3. Skip dedicated block APIs and force callers through `Graph.from_dgl(...)` / `Graph.to_dgl()`.

Option 2 is the right slice.

Option 1 would silently change the return type of existing `Block.from_dgl(...)` call sites, which is a bad compatibility surprise. Option 3 throws away block-specific source/destination frontier metadata, which is the whole point of the abstraction. Separate explicit APIs keep the old `Block` surface stable while letting `HeteroBlock` participate fully in DGL interop.

## Recommended Public Surface

Add:

- `HeteroBlock.from_dgl(dgl_block)`
- `HeteroBlock.to_dgl()`
- `vgl.compat.hetero_block_from_dgl(dgl_block)`
- `vgl.compat.hetero_block_to_dgl(block)`

Keep unchanged:

- `Block.from_dgl(...)` stays single-relation only
- `vgl.compat.block_from_dgl(...)` stays single-relation only

This keeps old single-relation code deterministic while making the multi-relation path explicit.

## Conversion Semantics

### Export

`hetero_block_to_dgl(...)` should:

1. Export one DGL block relation per `edge_type`.
2. Use the original public node types, not the internal `__src` / `__dst` store names.
3. Size DGL source/destination node sets from `src_n_id` / `dst_n_id`.
4. Copy all visible node/edge features except `edge_index`.
5. Preserve public `n_id` / `e_id` metadata in the DGL frames.
6. Preserve temporal `time_attr` metadata when present.

### Import

`hetero_block_from_dgl(...)` should:

1. Require `dgl_block.is_block`.
2. Allow multiple canonical edge types.
3. Normalize public `n_id` / `e_id` metadata, falling back to contiguous ids when external metadata is missing.
4. Reconstruct internal `src_store_types` / `dst_store_types` exactly the way `to_hetero_block(...)` does, so same-type relations get stable `__src` / `__dst` stores.
5. Rebuild a wrapped VGL graph plus `src_n_id` / `dst_n_id` dictionaries keyed by original node types.

## Stability Requirements

The adapter must stay schema-stable for:

- bipartite plus same-type relations in the same block
- empty edge sets for one selected relation
- node types that appear on only one side of the block

Those cases are common in sampled frontier layers and are exactly where store-name bookkeeping gets easy to break.

## Testing Strategy

Add focused regressions for:

- `HeteroBlock.to_dgl()` / `HeteroBlock.from_dgl(...)` round-trip on a multi-relation block
- importing an external multi-relation DGL block with explicit `n_id` / `e_id`
- same-type relation store renaming surviving the round-trip
- temporal `time_attr` propagation through multi-relation block export/import
- `Block.from_dgl(...)` still rejecting multi-relation DGL blocks
- `vgl.compat` exports exposing the new helper functions

Then rerun the focused DGL adapter suite and the full repository suite before merge.
