# Bootstrap Task Plan

## Goal

Add a task-wrapper training strategy that swaps the training loss to a soft or hard bootstrapped objective without modifying trainer core behavior.

## Scope

- Add `BootstrapTask` wrapper in `vgl.tasks`
- Support both `soft` and `hard` bootstrap modes
- Support task export from `vgl.tasks`, `vgl.train.tasks`, `vgl.train`, and `vgl`
- Add tests for:
  - training-only bootstrap behavior
  - binary and multiclass support
  - composition with paired-loss tasks such as `RDropTask`
  - trainer behavior for paired-loss training vs evaluation
  - package exports

## Design

- Wrap any base task and delegate attributes, targets, and metric predictions
- During training, blend observed targets with model predictions using `beta`
- Use detached model probabilities as soft pseudo-labels in `soft` mode
- Use argmax or threshold pseudo-labels in `hard` mode
- Leave validation and test losses unchanged by delegating to the base task
- If the base task supports `paired_loss`, preserve any extra paired regularization term and replace only the averaged supervised component with the bootstrapped objective

## Verification

- `python -m pytest tests/train/test_tasks.py tests/train/test_training_strategies.py tests/test_package_exports.py tests/test_package_layout.py -q`
- `python -m pytest -q`
- `python -m ruff check vgl tests examples`
