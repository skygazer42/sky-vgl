# Gradient Value Clipping Plan

## Goal

Add a callback-based training strategy that clips dense gradient values before optimizer steps.

## Scope

- Add `GradientValueClipping` callback in `vgl.engine.callbacks`
- Support callback export from `vgl.engine`, `vgl.train`, `vgl.train.callbacks`, and `vgl`
- Add tests for:
  - callback configuration validation
  - dense gradient clipping behavior
  - sparse gradient passthrough
  - package exports

## Design

- Run clipping during `on_before_optimizer_step`
- Apply elementwise clamp to dense gradients using `[-clip_value, clip_value]`
- Skip sparse gradients and parameters without gradients
- Keep the callback stateless, since clipping itself has no resumable state

## Verification

- `python -m pytest tests/train/test_callbacks.py tests/test_package_exports.py tests/test_package_layout.py -q`
- `python -m pytest -q`
- `python -m ruff check vgl tests examples`
