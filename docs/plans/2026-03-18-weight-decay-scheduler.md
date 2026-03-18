# Weight Decay Scheduler Plan

## Goal

Add a callback-based training strategy that schedules optimizer `weight_decay` over epochs without changing trainer core logic.

## Scope

- Add `WeightDecayScheduler` callback in `vgl.engine.callbacks`
- Support callback export from `vgl.engine`, `vgl.train`, `vgl.train.callbacks`, and `vgl`
- Add tests for:
  - callback configuration validation
  - linear factor scheduling behavior and original-value restore
  - callback state restoration / checkpoint resume
  - package exports

## Design

- Use a linear epoch schedule from `start_factor` to `end_factor`
- Apply the scheduled factor to each optimizer param group's original `weight_decay`
- Preserve param-group-relative structure instead of overwriting all groups with one absolute value
- Restore the original param-group `weight_decay` values after `fit()`

## Verification

- `python -m pytest tests/train/test_callbacks.py tests/train/test_trainer_evaluation.py tests/test_package_exports.py tests/test_package_layout.py -q`
- `python -m pytest -q`
- `python -m ruff check vgl tests examples`
