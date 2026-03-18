# Poly1 Cross Entropy Task Plan

## Goal

Add a task-wrapper training strategy that swaps the training loss to Poly-1 cross entropy without modifying trainer core behavior.

## Scope

- Add `Poly1CrossEntropyTask` wrapper in `vgl.tasks`
- Support task export from `vgl.tasks`, `vgl.train.tasks`, `vgl.train`, and `vgl`
- Add tests for:
  - training-only Poly-1 behavior
  - binary and multiclass support
  - composition with paired-loss tasks such as `RDropTask`
  - trainer behavior for paired-loss training vs evaluation
  - package exports

## Design

- Wrap any base task and delegate attributes, targets, and metric predictions
- During training, compute `base_loss + epsilon * mean(1 - p_true)` from the base task's metric-ready logits
- Leave validation and test losses unchanged by delegating to the base task
- If the base task supports `paired_loss`, preserve any extra paired regularization term and replace only the averaged supervised component with the Poly-1 objective

## Verification

- `python -m pytest tests/train/test_tasks.py tests/train/test_training_strategies.py tests/test_package_exports.py tests/test_package_layout.py -q`
- `python -m pytest -q`
- `python -m ruff check vgl tests examples`
