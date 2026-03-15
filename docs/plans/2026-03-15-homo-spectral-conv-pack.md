# Homogeneous Spectral Convolution Pack Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `TAGConv`, `SGConv`, and `ChebConv` as stable homogeneous operators and wire them into the existing export, test, and example surface.

**Architecture:** Add one file per operator under `vgl.nn.conv`, keep each layer compatible with the current homogeneous `Graph` and `Trainer` path, and extend the existing compact conv-zoo test and example instead of creating parallel integration surfaces. Use a tiny shared helper only if it reduces obvious duplication across the three operators.

**Tech Stack:** Python 3.11+, PyTorch, pytest, ruff, mypy

**Execution Rules:** Use `@test-driven-development` for every code task, keep commits small, and use `@verification-before-completion` before claiming the phase is done.

---

### Task 1: Add Failing Tests for the New Operator Batch

**Files:**
- Modify: `tests/nn/test_convs.py`
- Modify: `tests/integration/test_homo_conv_zoo.py`
- Modify: `tests/test_package_exports.py`

**Step 1: Write the failing contract tests**

Extend `tests/nn/test_convs.py` with:

- imports for `TAGConv`, `SGConv`, `ChebConv`
- shape tests for graph input
- one shape test for `TAGConv(x, edge_index)`
- heterogeneous graph rejection tests for the new operators

Use small tensors and assert only stable contract behavior:

```python
def test_tag_conv_accepts_graph_input():
    conv = TAGConv(in_channels=4, out_channels=3, k=2)

    out = conv(_homo_graph())

    assert out.shape == (3, 3)


def test_tag_conv_accepts_x_and_edge_index():
    x = torch.randn(3, 4)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    conv = TAGConv(in_channels=4, out_channels=3, k=2)

    out = conv(x, edge_index)

    assert out.shape == (3, 3)


def test_sg_conv_accepts_graph_input():
    conv = SGConv(in_channels=4, out_channels=3, k=2)

    out = conv(_homo_graph())

    assert out.shape == (3, 3)


def test_cheb_conv_accepts_graph_input():
    conv = ChebConv(in_channels=4, out_channels=3, k=3)

    out = conv(_homo_graph())

    assert out.shape == (3, 3)
```

Extend the heterogeneous rejection parametrization with:

```python
(TAGConv, {"in_channels": 4, "out_channels": 3}),
(SGConv, {"in_channels": 4, "out_channels": 3}),
(ChebConv, {"in_channels": 4, "out_channels": 3}),
```

**Step 2: Write the failing integration extension**

Extend `tests/integration/test_homo_conv_zoo.py` so `convs` also includes:

```python
TAGConv(in_channels=4, out_channels=4, k=2),
SGConv(in_channels=4, out_channels=4, k=2),
ChebConv(in_channels=4, out_channels=4, k=3),
```

**Step 3: Write the failing export assertions**

Update `tests/test_package_exports.py` so the root package also exposes:

```python
from vgl import TAGConv, SGConv, ChebConv

assert TAGConv.__name__ == "TAGConv"
assert SGConv.__name__ == "SGConv"
assert ChebConv.__name__ == "ChebConv"
```

**Step 4: Run the tests to verify RED**

Run: `python -m pytest tests/nn/test_convs.py tests/integration/test_homo_conv_zoo.py tests/test_package_exports.py -v`

Expected: `FAIL` because the new operator classes and exports do not exist yet.

**Step 5: Commit nothing yet**

Do not commit until the operator implementations exist and the tests are green.

### Task 2: Implement `TAGConv`, `SGConv`, and `ChebConv` with Public Exports

**Files:**
- Create: `vgl/nn/conv/tag.py`
- Create: `vgl/nn/conv/sg.py`
- Create: `vgl/nn/conv/cheb.py`
- Modify: `vgl/nn/conv/__init__.py`
- Modify: `vgl/nn/__init__.py`
- Modify: `vgl/__init__.py`

**Step 1: Add minimal operator implementations**

Implement:

- `TAGConv(in_channels, out_channels, k=2)`
- `SGConv(in_channels, out_channels, k=1)`
- `ChebConv(in_channels, out_channels, k=2)`

All three should:

- accept `conv(graph)` and `conv(x, edge_index)`
- fail early on heterogeneous graphs with `"homogeneous"` in the error
- preserve tensor device and dtype

Use a shared normalized propagation helper only if it stays internal and tiny. A valid helper shape is:

```python
def _propagate_mean(x, edge_index):
    row, col = edge_index
    out = torch.zeros_like(x)
    out.index_add_(0, col, x[row])
    degree = torch.zeros(x.size(0), dtype=x.dtype, device=x.device)
    degree.index_add_(0, col, torch.ones(col.size(0), dtype=x.dtype, device=x.device))
    return out / degree.clamp_min(1).unsqueeze(-1)
```

`SGConv` should propagate `k` times, then apply one linear layer.

`TAGConv` should stack `[x, hop1, ..., hopk]` and project once.

`ChebConv` should build recurrence terms:

```python
t0 = x
t1 = propagate(x)
tn = 2 * propagate(tn_minus_1) - tn_minus_2
```

then concatenate `[t0, ..., tk]` and project once.

**Step 2: Export the operators**

Update:

- `vgl/nn/conv/__init__.py`
- `vgl/nn/__init__.py`
- `vgl/__init__.py`

so the new operators are available from the package root.

**Step 3: Run the focused tests**

Run: `python -m pytest tests/nn/test_convs.py tests/integration/test_homo_conv_zoo.py tests/test_package_exports.py -v`

Expected: `PASS`

**Step 4: Make only the smallest follow-up fixes**

If failures remain, limit fixes to:

- homogeneous graph input handling
- propagation output shapes
- export omissions
- integration hidden-size mismatches

Do not widen this batch with extra options.

**Step 5: Re-run the focused tests**

Run: `python -m pytest tests/nn/test_convs.py tests/integration/test_homo_conv_zoo.py tests/test_package_exports.py -v`

Expected: `PASS`

**Step 6: Commit**

```bash
git add vgl/nn/conv/tag.py vgl/nn/conv/sg.py vgl/nn/conv/cheb.py vgl/nn/conv/__init__.py vgl/nn/__init__.py vgl/__init__.py tests/nn/test_convs.py tests/integration/test_homo_conv_zoo.py tests/test_package_exports.py
git commit -m "feat: add spectral homogeneous conv pack"
```

### Task 3: Extend the Example and Documentation Surface

**Files:**
- Modify: `examples/homo/conv_zoo.py`
- Modify: `README.md`
- Modify: `docs/quickstart.md`
- Modify: `docs/core-concepts.md`

**Step 1: Extend the example surface**

Update `examples/homo/conv_zoo.py` so it imports the new operators and runs:

```python
run_one("tag", TAGConv(in_channels=4, out_channels=4, k=2), 4)
run_one("sg", SGConv(in_channels=4, out_channels=4, k=2), 4)
run_one("cheb", ChebConv(in_channels=4, out_channels=4, k=3), 4)
```

Keep the example compact and continue printing one list of result dictionaries.

**Step 2: Update docs minimally**

Update:

- `README.md` to list the new built-in operators and keep `conv_zoo.py` as the main demonstration entrypoint
- `docs/quickstart.md` to mention `TAGConv`, `SGConv`, and `ChebConv` as swappable homogeneous backbones
- `docs/core-concepts.md` to reflect the expanded built-in conv operator set

**Step 3: Run the example**

Run: `python examples/homo/conv_zoo.py`

Expected: prints result dictionaries for `gin`, `gatv2`, `appnp`, `tag`, `sg`, and `cheb`

**Step 4: Make only the smallest fixes**

If the example fails, fix only import, hidden-size, or operator shape issues.

**Step 5: Commit**

```bash
git add examples/homo/conv_zoo.py README.md docs/quickstart.md docs/core-concepts.md
git commit -m "feat: extend homo conv zoo"
```

### Task 4: Run Full Regression Verification

**Files:**
- Verify only

**Step 1: Run the focused batch tests**

Run: `python -m pytest tests/nn/test_convs.py tests/integration/test_homo_conv_zoo.py tests/test_package_exports.py -v`

Expected: `PASS`

**Step 2: Run the full test suite**

Run: `python -m pytest -v`

Expected: `PASS`

**Step 3: Run lint and type checks**

Run: `python -m ruff check .`

Expected: `All checks passed`

Run: `python -m mypy vgl`

Expected: `Success: no issues found`

**Step 4: Run the example smoke suite**

Run:

- `python examples/homo/node_classification.py`
- `python examples/homo/graph_classification.py`
- `python examples/homo/link_prediction.py`
- `python examples/homo/conv_zoo.py`
- `python examples/hetero/node_classification.py`
- `python examples/hetero/graph_classification.py`
- `python examples/temporal/event_prediction.py`

Expected: all commands exit `0` and print result summaries.

**Step 5: Commit follow-up fixes only if verification required edits**

```bash
git add README.md docs/quickstart.md docs/core-concepts.md vgl tests examples
git commit -m "chore: verify spectral conv pack"
```

Do not create an empty commit if no follow-up edits were needed.
