# Landmark Operator Phase 3 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add another batch of landmark graph operators by covering missing edge-aware/geometric convolutions and one more modern graph-transformer encoder family.

**Architecture:** Keep homogeneous operators in `vgl.nn.conv` with the existing `graph_or_x, edge_index=None` calling convention and resolve required edge-aligned tensors from `graph.edata` when present. Add the new encoder family to `vgl.nn.encoders` as lightweight VGL-native modules that combine sparse/local propagation with dense global attention rather than paper-for-paper heavyweight reproductions.

**Tech Stack:** Python, PyTorch, pytest

---

## Phase 3 Scope

- Add `CGConv` as a gated edge-conditioned convolution for attributed homogeneous graphs.
- Add `SplineConv` as a pseudo-coordinate geometric convolution with continuous kernel interpolation.
- Add `FAConv` as a frequency-adaptive residual attention operator aligned with the broader `FAGCN` family.
- Add `SGFormerEncoderLayer` and `SGFormerEncoder` as a lightweight local-global transformer encoder family.
- Export the new operators through `vgl.nn` and the root `vgl` surface.

## Year Coverage Rationale

- `2018`: `CGConv` and `SplineConv` are still common landmark edge-aware / geometric operators.
- `2021`: `FAConv` remains a recognizable frequency-adaptive residual graph convolution missing from the current public surface.
- `2023`: `SGFormer`-style local-global encoders are a good fit for the current abstraction without forcing an unstable `2024-2026` paper chase into the stable API.

### Task 1: Lock The New Surface With Tests

**Files:**
- Modify: `tests/nn/test_convs.py`
- Modify: `tests/nn/test_transformer_encoders.py`
- Modify: `tests/integration/test_homo_conv_zoo.py`
- Modify: `tests/test_package_exports.py`

**Step 1: Write the failing tests**

- Add forward-shape tests for `CGConv`, `SplineConv`, and `FAConv`.
- Cover both graph-input resolution of `edge_attr` / `pseudo` and explicit-argument calling paths where appropriate.
- Add forward-shape tests for `SGFormerEncoderLayer` and `SGFormerEncoder`.
- Extend the training-loop integration zoo so the new operators are exercised end-to-end.
- Extend package export tests so the new classes are visible from `vgl` and `vgl.nn`.

**Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/nn/test_convs.py tests/nn/test_transformer_encoders.py tests/integration/test_homo_conv_zoo.py tests/test_package_exports.py -q
```

Expected: FAIL because the new operators and exports do not exist yet.

### Task 2: Implement Edge-Aware And Geometric Convolutions

**Files:**
- Create: `vgl/nn/conv/cg.py`
- Create: `vgl/nn/conv/spline.py`
- Create: `vgl/nn/conv/fa.py`
- Modify: `vgl/nn/conv/__init__.py`
- Modify: `vgl/nn/__init__.py`
- Modify: `vgl/__init__.py`

**Step 1: Add `CGConv`**

- Use a gated message function over concatenated source, destination, and edge features.
- Support `sum`, `mean`, and `max` aggregation.
- Keep the operator channel-preserving and residual by default.

**Step 2: Add `SplineConv`**

- Resolve pseudo-coordinates from `graph.edata["pseudo"]` or an explicit argument.
- Build lightweight continuous kernel interpolation over a regular pseudo-coordinate grid.
- Support optional root weighting to preserve current VGL conv conventions.

**Step 3: Add `FAConv`**

- Accept the current node features plus initial/reference features `x0`.
- Compute attention-like edge weights and mix propagated features with the initial signal using `eps`.
- Support graph input plus explicit `x0` for compatibility with the rest of the library.

### Task 3: Implement A New Transformer Encoder Family

**Files:**
- Modify: `vgl/nn/encoders.py`
- Modify: `vgl/nn/__init__.py`
- Modify: `vgl/__init__.py`

**Step 1: Add `SGFormerEncoderLayer`**

- Combine one local graph propagation branch with one dense multi-head self-attention branch.
- Blend the branches with a simple learnable or configurable mixing coefficient.
- Wrap the block with residual connections, normalization, and a feed-forward network.

**Step 2: Add `SGFormerEncoder`**

- Stack `SGFormerEncoderLayer` modules with the same `graph_or_x, edge_index=None` convention as the other encoders.
- Keep output channels equal to input channels to stay composable with the current training zoo.

### Task 4: Export And Document

**Files:**
- Modify: `vgl/nn/conv/__init__.py`
- Modify: `vgl/nn/__init__.py`
- Modify: `vgl/__init__.py`
- Modify: `README.md`

**Step 1: Export the new classes**

- Keep `vgl.nn` as the preferred neural API surface.
- Preserve root-level convenience imports for parity with the existing package style.

**Step 2: Update docs where needed**

- Touch operator lists only where the new public API is already enumerated.

### Task 5: Verify

**Files:**
- No code changes expected

**Step 1: Run targeted verification**

```bash
python -m pytest tests/nn/test_convs.py tests/nn/test_transformer_encoders.py tests/integration/test_homo_conv_zoo.py tests/test_package_exports.py -q
python -m ruff check vgl tests examples
```

Expected: PASS

**Step 2: Run the full suite**

```bash
python -m pytest -q
python -m ruff check vgl tests examples
```

Expected: PASS
