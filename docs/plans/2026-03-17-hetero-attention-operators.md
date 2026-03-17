# Hetero Attention Operators Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add one compact heterogeneous operator batch by covering two missing relation-aware attention convolutions that fit the current VGL heterogeneous graph contract.

**Architecture:** Keep both operators in `vgl.nn.conv` and preserve the current heterogeneous convention of accepting an explicit `graph` object and returning `dict[node_type, tensor]`. Resolve per-relation `edge_attr` directly from each hetero edge store and avoid introducing a second "typed homogeneous tensor" runtime just to mirror external library APIs.

**Tech Stack:** Python, PyTorch, pytest

---

## Phase 7 Scope

- Add `HEATConv` as a heterogeneous edge-enhanced attention operator.
- Add `RGATConv` as a relation-aware graph attention operator.
- Export both operators from `vgl.nn.conv`, `vgl.nn`, and `vgl`.
- Extend heterogeneous contract tests, end-to-end hetero training tests, and README operator lists.

## Source Alignment

- `HEATConv` should stay close to the public PyG family by using node-type / relation-aware attention with edge attributes folded into the attention/message path.
- `RGATConv` should stay close to the public PyG family by using relation-specific attention and relation-specific message transforms.

This phase should remain VGL-sized and deliberately avoid full parity with typed-homogeneous tensor APIs, basis-decomposition options, or other heavyweight flags.

### Task 1: Lock The New Surface With Tests

**Files:**
- Modify: `tests/nn/test_convs.py`
- Modify: `tests/integration/test_end_to_end_hetero.py`
- Modify: `tests/test_package_exports.py`

**Step 1: Write the failing tests**

- Extend the hetero graph fixture to include per-relation `edge_attr`.
- Add forward-shape coverage for `HEATConv(graph)`.
- Add forward-shape coverage for `RGATConv(graph)`.
- Extend the hetero end-to-end training coverage so both new operators run through `Trainer`.
- Extend root package export tests so both classes are visible from `vgl`.

**Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/nn/test_convs.py tests/integration/test_end_to_end_hetero.py tests/test_package_exports.py -q
```

Expected: FAIL because `HEATConv`, `RGATConv`, and their exports do not exist yet.

### Task 2: Implement `HEATConv`

**Files:**
- Create: `vgl/nn/conv/heat.py`
- Modify: `vgl/nn/conv/__init__.py`
- Modify: `vgl/nn/__init__.py`
- Modify: `vgl/__init__.py`

**Step 1: Add an edge-enhanced hetero attention operator**

- Accept `in_channels`, `out_channels`, `node_types`, `relation_types`, `edge_channels`, and `heads`.
- Require a heterogeneous `graph` input and return one output tensor per node type.
- Use node-type and relation-aware attention terms with edge-attribute projections in the message path.

### Task 3: Implement `RGATConv`

**Files:**
- Create: `vgl/nn/conv/rgat.py`
- Modify: `vgl/nn/conv/__init__.py`
- Modify: `vgl/nn/__init__.py`
- Modify: `vgl/__init__.py`

**Step 1: Add a relation-aware hetero graph attention operator**

- Accept `in_channels`, `out_channels`, `node_types`, `relation_types`, optional `edge_channels`, and `heads`.
- Use relation-specific key/value transforms and destination-normalized attention.
- Support edge attributes when `edge_channels` is provided.

### Task 4: Integration Surface

**Files:**
- Modify: `tests/integration/test_end_to_end_hetero.py`
- Modify: `README.md`

**Step 1: Update integration and public docs**

- Keep the training coverage focused on node classification for one destination type.
- Add both operators to the README heterogeneous / relation-aware operator lists.

### Task 5: Verify

**Files:**
- No code changes expected

**Step 1: Run targeted verification**

```bash
python -m pytest tests/nn/test_convs.py tests/integration/test_end_to_end_hetero.py tests/test_package_exports.py -q
python -m ruff check vgl tests examples
```

Expected: PASS

**Step 2: Run the full suite**

```bash
python -m pytest -q
python -m ruff check vgl tests examples
```

Expected: PASS
