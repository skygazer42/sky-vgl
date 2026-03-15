# Homogeneous Spectral Convolution Pack Design

**Date:** 2026-03-15
**Status:** Approved for planning

## Goal

Expand `vgl` with a second batch of stable homogeneous operators by adding `TAGConv`, `SGConv`, and `ChebConv` without changing the current `Graph` and `Trainer` integration model.

## Scope Decisions

- Focus on homogeneous graph operators only
- Add one coherent operator batch:
  - `TAGConv`
  - `SGConv`
  - `ChebConv`
- Keep the public API compatible with the current conv surface
- Continue supporting both `conv(graph)` and `conv(x, edge_index)`
- Reuse the current training loop and example structure
- Prefer compact implementations over a broad message-passing rewrite

## Chosen Direction

Three directions were considered:

1. Add more attention-style layers first
2. Add one classic spectral-style homogeneous batch next
3. Pause algorithm work and redesign `MessagePassing` before adding more operators

The chosen direction is:

> Add one classic spectral-style homogeneous batch next, with lightweight implementations that preserve the existing stable user entrypoints.

This gives `vgl` broader algorithm coverage while keeping the core abstraction under our control. A larger attention batch would overlap too much with the existing `GAT` family, and a framework rewrite would slow delivery without improving user-facing coverage enough.

## Architecture

Phase 7 should extend `vgl.nn.conv` with:

- `TAGConv`
- `SGConv`
- `ChebConv`

These layers should live under:

- `vgl/nn/conv/tag.py`
- `vgl/nn/conv/sg.py`
- `vgl/nn/conv/cheb.py`

They should be exported from:

- `vgl.nn.conv`
- `vgl.nn`
- `vgl`

No additional namespace such as `spectral`, `advanced`, or `experimental` should be introduced. The built-in operator surface should remain flat and easy to discover.

## Public API Shape

The public import path should be:

```python
from vgl.nn.conv import TAGConv, SGConv, ChebConv
```

and also:

```python
from vgl import TAGConv, SGConv, ChebConv
```

The user-facing constructors should remain simple and close to PyG/DGL habits:

```python
conv = TAGConv(in_channels=64, out_channels=32, k=2)
out = conv(graph)

conv = SGConv(in_channels=64, out_channels=32, k=2)
out = conv(graph)

conv = ChebConv(in_channels=64, out_channels=32, k=3)
out = conv(graph)
```

The parameter surface should stay intentionally small for now. The goal is to expose stable, usable operators, not every optional flag from larger frameworks.

## Operator Semantics

### Shared propagation rule

This phase should keep propagation simple and numerically stable enough for the tiny training examples:

- support homogeneous graphs only
- use repeated neighborhood aggregation over `edge_index`
- normalize by destination degree to avoid uncontrolled scale growth across hops
- preserve input dtype and device

This is a pragmatic implementation choice for `vgl`, not a promise of full framework parity.

### SGConv

`SGConv` should expose:

- `in_channels`
- `out_channels`
- `k=1`

The semantic should be:

- repeatedly propagate node features for `k` hops
- apply one linear projection after propagation

Output shape:

- `[num_nodes, out_channels]`

### TAGConv

`TAGConv` should expose:

- `in_channels`
- `out_channels`
- `k=2`

The semantic should be:

- keep the original node features
- compute hop features from `A x` up to `A^k x`
- concatenate `[x, hop1, ..., hopk]`
- project once to `out_channels`

Output shape:

- `[num_nodes, out_channels]`

### ChebConv

`ChebConv` should expose:

- `in_channels`
- `out_channels`
- `k=2`

The semantic should be:

- build a lightweight Chebyshev-style basis over repeated normalized propagation
- use recurrence terms up to order `k`
- concatenate the basis terms
- project once to `out_channels`

Output shape:

- `[num_nodes, out_channels]`

This keeps the implementation small while still preserving the key difference from `TAGConv`: the basis is recurrence-driven rather than plain hop stacking.

## Message Passing Boundary

Phase 7 should not force these layers through a large `MessagePassing` redesign.

Rules:

- allow each layer to own its propagation loop
- extract only tiny shared helpers if duplication becomes noisy
- do not rework the existing conv stack just to generalize the new operators

The package still benefits more from operator breadth than from an internal abstraction reset at this stage.

## Graph and Input Constraints

Phase 7 should enforce:

- homogeneous graph input only
- support for both `conv(graph)` and `conv(x, edge_index)`
- clear early failure on heterogeneous graphs with `"homogeneous"` in the error
- output tensors following the input dtype and device

## Testing Strategy

Phase 7 tests should cover:

### Operator contract tests

In `tests/nn/test_convs.py` add coverage for:

- `TAGConv` on `Graph`
- `SGConv` on `Graph`
- `ChebConv` on `Graph`
- `TAGConv` on `(x, edge_index)`
- `SGConv` and `ChebConv` order behavior through stable output shapes
- early failure on heterogeneous graph input

### Public export tests

`tests/test_package_exports.py` should assert that:

- `TAGConv`
- `SGConv`
- `ChebConv`

are exposed from the root package.

### Integration path

Extend the compact homogeneous conv integration path so the new operators also plug into the current training loop.

### Example surface

Extend `examples/homo/conv_zoo.py` so it runs:

- `gin`
- `gatv2`
- `appnp`
- `tag`
- `sg`
- `cheb`

This keeps the example entrypoint compact and makes operator growth visible in one place.

## Deliverables

Phase 7 should ship:

- `TAGConv`
- `SGConv`
- `ChebConv`
- exports from `vgl.nn.conv`, `vgl.nn`, and `vgl`
- contract tests and export tests
- training-loop integration coverage
- an updated `conv_zoo` example
- minimal documentation updates

## Explicit Non-Goals

Do not include:

- heterogeneous operator work
- temporal operator work
- edge-aware spectral variants
- caching controls
- normalization option matrices
- dropout or residual toggles
- a `MessagePassing` rewrite

## Repository Touchpoints

Phase 7 will mostly affect:

- `vgl/nn/conv/`
- `vgl/nn/__init__.py`
- `vgl/__init__.py`
- `tests/nn/`
- `tests/integration/`
- `tests/test_package_exports.py`
- `examples/homo/conv_zoo.py`
- `README.md`
- `docs/`

## Acceptance Criteria

Phase 7 is complete when:

1. `TAGConv`, `SGConv`, and `ChebConv` are publicly exported
2. all three operators work on homogeneous graphs and support both call styles
3. the compact integration test covers the new operators
4. `examples/homo/conv_zoo.py` demonstrates the expanded operator set
5. existing tests and examples continue to pass

## Next Step

The next step is to write an implementation plan with exact file edits, tests, commands, and commit checkpoints.
