# History And Propagation Operators Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add one compact operator batch that covers a history-aware attention convolution and a deeper iterative propagation convolution missing from the current VGL operator surface.

**Architecture:** Keep both operators in `vgl.nn.conv` and preserve the current homogeneous `graph_or_x, edge_index=None` conventions. `DNAConv` should accept node-history tensors without forcing a wider `Graph` abstraction change, while `TWIRLSConv` should package pre-MLP, propagation, and post-MLP blocks into one trainable homogeneous operator that fits the existing training zoo.

**Tech Stack:** Python, PyTorch, pytest

---

## Phase 4 Scope

- Add `DNAConv` as a dynamic neighborhood aggregation layer over per-node feature histories.
- Add `TWIRLSConv` as a deeper iterative propagation operator with lightweight pre/post MLPs.
- Export both layers through `vgl.nn.conv`, `vgl.nn`, and the root `vgl` surface.
- Extend docs and integration coverage to include the new batch.

## Source Alignment

- `DNAConv` should stay close to the public PyG operator family shape: equal-width channels, multi-head attention, and history-aware input rather than a plain `x` tensor.
- `TWIRLSConv` should stay close to the public DGL operator family shape: MLP blocks wrapped around iterative propagation with explicit propagation-step control.

This phase should remain VGL-sized and deliberately avoid full parity with every optional flag in those libraries.

### Task 1: Lock The New Surface With Tests

**Files:**
- Modify: `tests/nn/test_convs.py`
- Modify: `tests/integration/test_homo_conv_zoo.py`
- Modify: `tests/test_package_exports.py`

**Step 1: Write the failing tests**

- Add `DNAConv` forward-shape tests for graph input plus explicit history tensors.
- Add `DNAConv` forward-shape tests for direct `(history, edge_index)` usage.
- Add `TWIRLSConv` forward-shape tests for graph input and direct `(x, edge_index)` usage.
- Extend the homogeneous training-loop integration zoo to include both operators.
- Extend package export tests so both classes are visible from `vgl`.

**Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/nn/test_convs.py tests/integration/test_homo_conv_zoo.py tests/test_package_exports.py -q
```

Expected: FAIL because `DNAConv`, `TWIRLSConv`, and their exports do not exist yet.

### Task 2: Implement `DNAConv`

**Files:**
- Create: `vgl/nn/conv/dna.py`
- Modify: `vgl/nn/conv/__init__.py`
- Modify: `vgl/nn/__init__.py`
- Modify: `vgl/__init__.py`

**Step 1: Add a history-aware homogeneous operator**

- Accept a per-node history tensor with shape `[num_nodes, num_layers, channels]`.
- Support `conv(graph, history=...)` and `conv(history, edge_index)`.
- Use the latest node state as the query and attend over source-node history tokens.
- Return one equal-width output tensor `[num_nodes, channels]`.

### Task 3: Implement `TWIRLSConv`

**Files:**
- Create: `vgl/nn/conv/twirls.py`
- Modify: `vgl/nn/conv/__init__.py`
- Modify: `vgl/nn/__init__.py`
- Modify: `vgl/__init__.py`

**Step 1: Add a deeper propagation operator**

- Support `conv(graph)` and `conv(x, edge_index)`.
- Apply one lightweight MLP before propagation, iterative smoothing for `steps`, and one lightweight MLP after propagation.
- Mix each propagation step with the initial hidden state using `alpha` to avoid collapse.
- Preserve dtype, device, and current homogeneous-only contract.

### Task 4: Integration Surface

**Files:**
- Modify: `tests/integration/test_homo_conv_zoo.py`
- Modify: `README.md`

**Step 1: Update integration and public docs**

- Ensure the training zoo can construct the extra runtime inputs needed by `DNAConv`.
- Add both operators to the README public operator list and headline counts where appropriate.

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
