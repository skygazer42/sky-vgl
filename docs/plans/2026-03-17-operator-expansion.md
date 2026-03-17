# Operator Expansion Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Expand the operator zoo with missing landmark graph convolutions and graph-transformer encoders while preserving the library's lightweight graph abstraction and testability.

**Architecture:** Keep the current split between low-level message passing ops in `vgl.nn.conv` and higher-level neural modules in `vgl.nn`. Use the existing homogeneous `graph_or_x, edge_index=None` convention for homo operators, and add explicit hetero graph modules for relation-aware operators instead of mutating the core `Graph` API first. Favor stable, trainable landmark operators over shallow paper-name stubs.

**Tech Stack:** Python, PyTorch, pytest

---

## Yearly Roadmap

- `2015-2016`: spectral foundations. Already covered by `ChebConv`, `GCNConv`.
- `2017`: inductive and relation-aware operators. `SAGEConv`, `GATConv` exist; `RGCNConv` is still missing.
- `2018`: expressive neighborhood aggregation. `GINConv`, `AGNNConv`, `ARMAConv` already exist.
- `2019`: propagation and scalable sampling. `APPNPConv`, `ClusterGCNConv` already exist.
- `2020`: deeper and stronger aggregators. `GCN2Conv`, `GENConv`, `PNAConv`, `DAGNNConv` already exist; `HGTConv` is still missing.
- `2021`: graph transformers become mainstream. `TransformerConv` exists; `Graphormer`-style encoder is still missing.
- `2022`: hybrid local-global transformer layers. `GPSLayer` is still missing.
- `2023`: hop-token or scalable sparse graph transformers. `NAGphormer`-style encoder is still missing.
- `2024-2026`: fast-moving graph-transformer variants do not yet have one clear cross-library standard. This phase should treat them as an "emerging encoder bucket" and prefer stable reusable encoder scaffolds over paper-chasing one-offs.

## Phase 1 Scope

- Add `RGCNConv` for relation-aware heterogeneous message passing.
- Add `HGTConv` for heterogeneous transformer-style attention.
- Add `GraphTransformerEncoderLayer` and `GraphTransformerEncoder`.
- Add `GraphormerEncoderLayer` and `GraphormerEncoder`.
- Add `GPSLayer`.
- Add `NAGphormerEncoder`.
- Complete root exports for existing `GATConv` and `SAGEConv` while touching the operator surface.

### Task 1: Lock The New Operator Surface With Tests

**Files:**
- Modify: `tests/nn/test_convs.py`
- Create: `tests/nn/test_transformer_encoders.py`
- Modify: `tests/integration/test_homo_conv_zoo.py`
- Modify: `tests/test_package_exports.py`

**Step 1: Write the failing tests**

- Add hetero forward-shape tests for `RGCNConv` and `HGTConv`.
- Add homogeneous forward-shape tests for `GraphTransformerEncoder`, `GraphormerEncoder`, `GPSLayer`, and `NAGphormerEncoder`.
- Add at least one integration test proving the new encoders plug into the training loop.
- Extend package export tests so the new operators are visible from `vgl.nn` and the broad root package.

**Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/nn/test_convs.py tests/nn/test_transformer_encoders.py tests/integration/test_homo_conv_zoo.py tests/test_package_exports.py -q
```

Expected: FAIL because the new operators and exports do not exist yet.

### Task 2: Implement Heterogeneous Landmark Operators

**Files:**
- Create: `vgl/nn/conv/rgcn.py`
- Create: `vgl/nn/conv/hgt.py`
- Modify: `vgl/nn/conv/__init__.py`
- Modify: `vgl/nn/__init__.py`
- Modify: `vgl/__init__.py`

**Step 1: Add `RGCNConv`**

- Use explicit `node_types` and `relation_types` at construction time.
- Return a `dict[node_type, tensor]`.
- Apply relation-specific source projections and mean aggregation into each destination node type.

**Step 2: Add `HGTConv`**

- Use per-node-type query/key/value projections and relation-specific transforms.
- Support `heads` and return per-node-type outputs with destination-type residual projections.

### Task 3: Implement Graph Transformer Encoders

**Files:**
- Create: `vgl/nn/encoders.py`
- Modify: `vgl/nn/__init__.py`
- Modify: `vgl/__init__.py`

**Step 1: Add `GraphTransformerEncoderLayer` and `GraphTransformerEncoder`**

- Wrap sparse `TransformerConv` with residual, normalization, and feed-forward blocks.

**Step 2: Add `GraphormerEncoderLayer` and `GraphormerEncoder`**

- Implement homogeneous dense self-attention with structural bias derived from graph connectivity or shortest-path buckets.

**Step 3: Add `GPSLayer`**

- Combine one local graph operator with one global self-attention block and an MLP.

**Step 4: Add `NAGphormerEncoder`**

- Build hop-token sequences from repeated propagation, then run a transformer encoder over hop tokens per node.

### Task 4: Export And Verify

**Files:**
- Modify: `vgl/nn/conv/__init__.py`
- Modify: `vgl/nn/__init__.py`
- Modify: `vgl/__init__.py`
- Modify: `README.md`

**Step 1: Export the new operators**

- Keep `vgl.nn` as the canonical neural entry point.
- Preserve the root package as the broad convenience surface.

**Step 2: Run targeted verification**

```bash
python -m pytest tests/nn/test_convs.py tests/nn/test_transformer_encoders.py tests/integration/test_homo_conv_zoo.py tests/test_package_exports.py -q
python -m ruff check vgl tests examples
```

Expected: PASS

### Task 5: Full Verification

**Files:**
- No code changes expected

**Step 1: Run the full suite**

```bash
python -m pytest -q
python -m ruff check vgl tests examples
```

Expected: PASS
