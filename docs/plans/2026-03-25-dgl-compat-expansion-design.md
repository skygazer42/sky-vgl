# DGL Compatibility Expansion Design

## Problem

`vgl.compat.dgl` currently only handles the narrowest homogeneous `DGLGraph` path. That means VGL can model heterogeneous and temporal graphs internally, but the compatibility boundary still drops canonical edge types and temporal metadata as soon as a user leaves the VGL runtime. Relative to DGL, that is a visible adapter gap rather than a modeling gap.

## Goals

- Preserve the current homogeneous DGL adapter behavior for the simple one-node-type / one-default-edge-type case
- Round-trip heterogeneous graphs with canonical edge types and typed node/edge features intact
- Round-trip temporal graphs without losing VGL's `time_attr`
- Keep storage-backed and sparse-cache-backed graphs safe to export without leaking internal cache mechanics into the adapter surface

## Design

### 1. Keep plain homogeneous graphs on `dgl.graph(...)`

For the exact plain homogeneous case, the adapter should keep the existing lightweight `dgl.graph(...)` path. That preserves the simplest migration story for users moving between VGL and ordinary DGL homogeneous workflows.

### 2. Export typed or temporal graphs through `dgl.heterograph(...)`

When a graph has typed node spaces, non-default relation names, multiple edge types, or temporal metadata, `to_dgl(...)` should export through `dgl.heterograph(...)`. This is the only stable way to preserve canonical edge types on the DGL side.

### 3. Preserve temporal metadata through adapter-owned attributes

DGL has edge tensors but no native VGL `time_attr` concept. The adapter should therefore attach a lightweight graph-level metadata field such as `vgl_time_attr` on exported DGL graphs. `from_dgl(...)` can read that field back and reconstruct `Graph.temporal(...)` when appropriate.

### 4. Make heterograph import tolerant but explicit

`from_dgl(...)` should detect heterograph-style typed node and edge spaces, import per-type node/edge payloads, and reconstruct a VGL `Graph.hetero(...)` or `Graph.temporal(...)`. This detection should stay adapter-local so the core `Graph` model does not need to know anything about DGL internals.

## Non-goals

- No attempt to mirror every DGL graph-level attribute or metagraph API
- No distributed DGL / DistDGL support
- No requirement that PyG and DGL adapters share identical implementation structure

## Verification

- Focused DGL adapter regressions for hetero, temporal, and storage-backed graphs
- Fresh full repository regression before merge
