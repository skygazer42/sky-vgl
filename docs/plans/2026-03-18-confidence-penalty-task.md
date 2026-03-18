# Confidence Penalty Task Plan

## Goal

Add a task-wrapper training strategy that penalizes overconfident predictions during training without changing trainer core logic.

## Scope

- Add `ConfidencePenaltyTask` wrapper in `vgl.tasks`
- Support task export from `vgl.tasks`, `vgl.train.tasks`, `vgl.train`, and `vgl`
- Add tests for:
  - training-only confidence penalty behavior
  - binary and multiclass entropy regularization
  - composition with paired-loss tasks such as `RDropTask`
  - package exports

## Design

- Wrap any base task and delegate attributes, targets, and metric predictions
- Add `- coefficient * entropy(predictions)` during training
- Leave evaluation and test losses unchanged
- If the base task supports `paired_loss`, regularize the average entropy of both prediction sets so trainer still uses two forwards

## Verification

- `python -m pytest tests/train/test_tasks.py tests/train/test_training_strategies.py tests/test_package_exports.py tests/test_package_layout.py -q`
- `python -m pytest -q`
- `python -m ruff check vgl tests examples`
