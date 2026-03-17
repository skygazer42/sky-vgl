# Point Geometric Operators Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add one compact geometric operator batch by covering two missing point-cloud-style graph convolutions that fit naturally on top of VGL's homogeneous graph abstraction.

**Architecture:** Keep both operators in `vgl.nn.conv` and preserve the existing homogeneous `graph_or_x, edge_index=None` calling convention. Resolve node positions from `graph.pos` / `graph.ndata["pos"]` when a graph object is passed, and expose explicit `pos=` arguments for direct tensor calls to stay consistent with the rest of the library's optional-runtime-input pattern.

**Tech Stack:** Python, PyTorch, pytest

---

## Phase 6 Scope

- Add `PointNetConv` as a point-geometric neighborhood MLP operator.
- Add `PointTransformerConv` as a position-aware local attention operator.
- Export both operators from `vgl.nn.conv`, `vgl.nn`, and `vgl`.
- Extend tests, README, and `examples/homo/conv_zoo.py` with position-aware graph coverage.

## Source Alignment

- `PointNetConv` should stay close to the public PyG family by aggregating MLP-transformed relative point offsets and source features.
- `PointTransformerConv` should stay close to the public PyG family by combining local attention with learned positional encodings.

This phase should remain VGL-sized and deliberately avoid full parity with optional kwargs such as bipartite inputs or separate local/global MLP customization.

### Task 1: Lock The New Surface With Tests

**Files:**
- Modify: `tests/nn/test_convs.py`
- Modify: `tests/integration/test_homo_conv_zoo.py`
- Modify: `tests/test_package_exports.py`

**Step 1: Write the failing tests**

- Add forward-shape coverage for `PointNetConv(graph)` using `pos` from node data.
- Add forward-shape coverage for `PointTransformerConv(x, edge_index, pos=...)`.
- Extend the homogeneous training-loop integration zoo so both new operators run end-to-end.
- Extend root package export tests so both classes are visible from `vgl`.

**Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/nn/test_convs.py tests/integration/test_homo_conv_zoo.py tests/test_package_exports.py -q
```

Expected: FAIL because `PointNetConv`, `PointTransformerConv`, and their exports do not exist yet.

### Task 2: Implement `PointNetConv`

**Files:**
- Create: `vgl/nn/conv/pointnet.py`
- Modify: `vgl/nn/conv/__init__.py`
- Modify: `vgl/nn/__init__.py`
- Modify: `vgl/__init__.py`

**Step 1: Add a point-geometric homogeneous operator**

- Accept `in_channels` and `out_channels`.
- Resolve `pos` from the graph or an explicit argument.
- Concatenate source-node features with relative coordinates and aggregate transformed neighborhood messages.
- Keep the operator node-output only and homogeneous-only.

### Task 3: Implement `PointTransformerConv`

**Files:**
- Create: `vgl/nn/conv/pointtransformer.py`
- Modify: `vgl/nn/conv/__init__.py`
- Modify: `vgl/nn/__init__.py`
- Modify: `vgl/__init__.py`

**Step 1: Add a position-aware local attention operator**

- Accept `in_channels`, `out_channels`, and `pos_channels`.
- Resolve `pos` from the graph or an explicit argument.
- Build learned positional encodings from relative offsets and use them in attention weights and value updates.
- Preserve homogeneous-only support and current tensor-call ergonomics.

### Task 4: Integration Surface

**Files:**
- Modify: `tests/integration/test_homo_conv_zoo.py`
- Modify: `examples/homo/conv_zoo.py`
- Modify: `README.md`

**Step 1: Update integration and public docs**

- Ensure the training zoo and example graph include `pos` so the new operators can run without special wrappers.
- Add the new operators to the README public operator list.

### Task 5: Verify

**Files:**
- No code changes expected

**Step 1: Run targeted verification**

```bash
python -m pytest tests/nn/test_convs.py tests/integration/test_homo_conv_zoo.py tests/test_package_exports.py -q
python -m ruff check vgl tests examples
```

Expected: PASS

**Step 2: Run the full suite**

```bash
python -m pytest -q
python -m ruff check vgl tests examples
```

Expected: PASS
