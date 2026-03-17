# Edge-Gated Operators Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add another compact operator batch by covering two missing edge-aware graph convolutions that fit naturally on top of VGL's stabilized `edge_data` contract.

**Architecture:** Keep both operators in `vgl.nn.conv` and preserve the existing homogeneous `graph_or_x, edge_index=None` calling convention. Resolve `edge_attr` from `graph.edata` when a graph object is passed, and expose explicit `edge_attr=` arguments for direct tensor calls to stay consistent with the rest of the edge-aware operator family.

**Tech Stack:** Python, PyTorch, pytest

---

## Phase 5 Scope

- Add `GatedGCNConv` as a gated edge-conditioned homogeneous convolution.
- Add `PDNConv` as an edge-conditioned propagation operator.
- Export both operators from `vgl.nn.conv`, `vgl.nn`, and `vgl`.
- Extend the homogeneous training zoo, package-export tests, README, and `examples/homo/conv_zoo.py`.

## Source Alignment

- `GatedGCNConv` should stay close to the public gated-GCN family by using edge-conditioned gates to modulate incoming messages.
- `PDNConv` should stay close to the public pathfinder-discovery family by generating edge-dependent propagation weights from `edge_attr`.

This phase should remain VGL-sized and deliberately avoid heavyweight parity features such as tuple inputs, returned edge states, or full paper-level option surfaces.

### Task 1: Lock The New Surface With Tests

**Files:**
- Modify: `tests/nn/test_convs.py`
- Modify: `tests/integration/test_homo_conv_zoo.py`
- Modify: `tests/test_package_exports.py`

**Step 1: Write the failing tests**

- Add forward-shape coverage for `GatedGCNConv(graph)` with `edge_attr` resolved from `graph.edata`.
- Add forward-shape coverage for `PDNConv(x, edge_index, edge_attr=...)`.
- Extend the homogeneous training-loop integration zoo so both new operators run end-to-end.
- Extend root package export tests so both classes are visible from `vgl`.

**Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/nn/test_convs.py tests/integration/test_homo_conv_zoo.py tests/test_package_exports.py -q
```

Expected: FAIL because `GatedGCNConv`, `PDNConv`, and their exports do not exist yet.

### Task 2: Implement `GatedGCNConv`

**Files:**
- Create: `vgl/nn/conv/gatedgcn.py`
- Modify: `vgl/nn/conv/__init__.py`
- Modify: `vgl/nn/__init__.py`
- Modify: `vgl/__init__.py`

**Step 1: Add a gated edge-aware homogeneous operator**

- Accept `in_channels`, `out_channels`, and `edge_channels`.
- Resolve edge features from `graph.edata["edge_attr"]` or explicit `edge_attr`.
- Use edge-conditioned sigmoid gates to modulate source-node messages into each destination node.
- Return node features only, keeping the current VGL operator family contract stable.

### Task 3: Implement `PDNConv`

**Files:**
- Create: `vgl/nn/conv/pdn.py`
- Modify: `vgl/nn/conv/__init__.py`
- Modify: `vgl/nn/__init__.py`
- Modify: `vgl/__init__.py`

**Step 1: Add an edge-conditioned propagation operator**

- Accept `in_channels`, `out_channels`, and `edge_channels`.
- Build edge-dependent propagation weights from `edge_attr`.
- Support optional self-loops and degree-style normalization in a lightweight VGL-native way.
- Preserve homogeneous-only support and current tensor-call ergonomics.

### Task 4: Integration Surface

**Files:**
- Modify: `tests/integration/test_homo_conv_zoo.py`
- Modify: `examples/homo/conv_zoo.py`
- Modify: `README.md`

**Step 1: Update integration and public docs**

- Ensure the training zoo and example graph include `edge_data` so the new operators can run without special wrappers.
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
