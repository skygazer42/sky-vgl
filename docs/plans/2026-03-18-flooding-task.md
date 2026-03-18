# Flooding Task Plan

## Goal

Add a task-wrapper training strategy that applies flooding during training without modifying trainer core behavior.

## Scope

- Add `FloodingTask` wrapper in `vgl.tasks`
- Support task export from `vgl.tasks`, `vgl.train.tasks`, `vgl.train`, and `vgl`
- Add tests for:
  - training-only flooding behavior
  - composition with paired-loss tasks such as `RDropTask`
  - trainer behavior for flooded paired-loss training vs evaluation
  - package exports

## Design

- Wrap any base task and delegate attributes, targets, and metric predictions
- Replace training loss with `abs(loss - level) + level`
- Leave evaluation and test losses unchanged
- If the base task supports `paired_loss`, expose a flooded paired-loss path so trainer still uses two forwards

## Verification

- `python -m pytest tests/train/test_tasks.py tests/train/test_training_strategies.py tests/test_package_exports.py tests/test_package_layout.py -q`
- `python -m pytest -q`
- `python -m ruff check vgl tests examples`
