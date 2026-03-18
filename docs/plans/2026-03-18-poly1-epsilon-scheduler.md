# Poly1 Epsilon Scheduler Plan

## Goal

Add a callback that linearly schedules `Poly1CrossEntropyTask.epsilon` across epochs without modifying trainer core behavior.

## Scope

- Add `Poly1EpsilonScheduler` callback in `vgl.engine.callbacks`
- Support export from `vgl.engine`, `vgl.train.callbacks`, `vgl.train`, and `vgl`
- Add tests for:
  - configuration validation
  - linear scheduling and original-value restoration
  - callback state round-trip
  - trainer checkpoint resume
  - package exports

## Design

- Resolve the first task in the wrapper chain that exposes `epsilon`
- Schedule `epsilon` linearly from `start_value` to `end_value` between `start_epoch` and `end_epoch`
- Apply the scheduled value on fit start and after each epoch
- Restore the original `epsilon` value on fit end
- Persist only the current scheduled value in callback state

## Verification

- `python -m pytest tests/train/test_callbacks.py tests/train/test_trainer_evaluation.py tests/test_package_exports.py tests/test_package_layout.py -q`
- `python -m pytest -q`
- `python -m ruff check vgl tests examples`
