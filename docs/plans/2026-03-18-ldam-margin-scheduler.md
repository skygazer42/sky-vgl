# LDAM Margin Scheduler Plan

## Goal

Add a callback-based training strategy that schedules task-level `ldam_max_margin` over epochs for LDAM losses.

## Scope

- Add `LdamMarginScheduler` callback in `vgl.engine.callbacks`
- Support callback export from `vgl.engine`, `vgl.train`, `vgl.train.callbacks`, and `vgl`
- Add tests for:
  - callback configuration validation
  - linear scheduling behavior and original-value restore
  - callback state restoration / checkpoint resume
  - package exports

## Design

- Use a linear epoch schedule from `start_value` to `end_value`
- Apply the scheduled value before each epoch by updating `task.ldam_max_margin`
- Support wrapped tasks such as `RDropTask` by resolving `base_task`
- Restore the original task `ldam_max_margin` after `fit()`

## Verification

- `python -m pytest tests/train/test_callbacks.py tests/train/test_trainer_evaluation.py tests/test_package_exports.py tests/test_package_layout.py -q`
- `python -m pytest -q`
- `python -m ruff check vgl tests examples`
