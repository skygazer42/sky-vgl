# Edge-Aware Operator Expansion Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add first-class homogeneous edge features and use them to implement high-value edge-aware graph operators plus one semantic heterogeneous attention operator.

**Architecture:** Extend `Graph.homo()` with explicit `edge_data` so homogeneous graphs can carry edge-level tensors without pretending they are node features. Keep the current lightweight graph abstraction, update PyG/DGL adapters to preserve common edge tensors, and implement edge-aware operators in `vgl.nn.conv` using the same `graph_or_x, edge_index=None` runtime style with optional edge feature arguments when needed.

**Tech Stack:** Python, PyTorch, pytest

---

### Task 1: Lock Homogeneous Edge Features With Tests

**Files:**
- Modify: `tests/core/test_graph_homo.py`
- Modify: `tests/core/test_graph_view.py`
- Modify: `tests/compat/test_pyg_adapter.py`
- Modify: `tests/compat/test_dgl_adapter.py`

**Step 1: Write the failing tests**

- Verify `Graph.homo(edge_data=...)` stores edge tensors under `edata` and schema metadata.
- Verify `snapshot()` / `window()` filter edge-aligned tensors together with `edge_index`.
- Verify PyG and DGL adapters preserve `edge_attr` on round trip.

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/core/test_graph_homo.py tests/core/test_graph_view.py tests/compat/test_pyg_adapter.py tests/compat/test_dgl_adapter.py -q
```

Expected: FAIL because `Graph.homo()` does not accept `edge_data` and adapters ignore edge features.

### Task 2: Lock The New Operator Surface With Tests

**Files:**
- Modify: `tests/nn/test_convs.py`
- Modify: `tests/integration/test_homo_conv_zoo.py`
- Modify: `tests/integration/test_end_to_end_hetero.py`
- Modify: `tests/test_package_exports.py`

**Step 1: Write the failing tests**

- Add homogeneous forward-shape tests for `NNConv`, `ECConv`, `GINEConv`, and `GMMConv`.
- Add heterogeneous forward-shape and training-loop tests for `HANConv`.
- Extend export coverage for the new operators.

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/nn/test_convs.py tests/integration/test_homo_conv_zoo.py tests/integration/test_end_to_end_hetero.py tests/test_package_exports.py -q
```

Expected: FAIL because the operators and exports do not exist yet.

### Task 3: Implement Homogeneous Edge Features

**Files:**
- Modify: `vgl/graph/graph.py`
- Modify: `vgl/graph/view.py`
- Modify: `vgl/compat/pyg.py`
- Modify: `vgl/compat/dgl.py`

**Step 1: Extend `Graph.homo()`**

- Add `edge_data=None` keyword support.
- Store edge tensors in the homogeneous `EdgeStore`.
- Keep `edge_index` under edge data and update schema edge feature names accordingly.

**Step 2: Keep view semantics correct**

- Ensure `snapshot()` and `window()` filter all edge-aligned tensors, not only `edge_index` and timestamp.

**Step 3: Preserve edge tensors in adapters**

- Round-trip `edge_attr` through PyG.
- Round-trip DGL `edata` tensors when present.

### Task 4: Implement Edge-Aware And Semantic Operators

**Files:**
- Create: `vgl/nn/conv/nnconv.py`
- Create: `vgl/nn/conv/gine.py`
- Create: `vgl/nn/conv/gmm.py`
- Create: `vgl/nn/conv/han.py`
- Modify: `vgl/nn/conv/__init__.py`
- Modify: `vgl/nn/__init__.py`
- Modify: `vgl/__init__.py`

**Step 1: Add `NNConv` and `ECConv`**

- Support `edge_attr` through either graph edge data or an explicit runtime argument.
- Use an edge MLP that produces per-edge weight matrices.
- Keep `ECConv` as a stable alias/wrapper around `NNConv`.

**Step 2: Add `GINEConv`**

- Combine neighborhood messages with edge features before MLP update.
- Keep the API small: `in_channels`, `out_channels`, optional `eps`.

**Step 3: Add `GMMConv`**

- Support pseudo-coordinates via edge data or explicit runtime argument.
- Use Gaussian kernels with learned means and variances.

**Step 4: Add `HANConv`**

- Use per-relation projections and semantic attention over relation outputs for each destination node type.
- Return a `dict[node_type, tensor]`.

### Task 5: Verify And Document

**Files:**
- Modify: `README.md`
- Modify: `docs/core-concepts.md`

**Step 1: Run targeted verification**

```bash
python -m pytest tests/core/test_graph_homo.py tests/core/test_graph_view.py tests/compat/test_pyg_adapter.py tests/compat/test_dgl_adapter.py tests/nn/test_convs.py tests/integration/test_homo_conv_zoo.py tests/integration/test_end_to_end_hetero.py tests/test_package_exports.py -q
python -m ruff check vgl tests examples
```

Expected: PASS

**Step 2: Run full verification**

```bash
python -m pytest -q
python -m ruff check vgl tests examples
```

Expected: PASS
