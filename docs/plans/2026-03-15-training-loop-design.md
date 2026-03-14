# Training Loop Phase 5 Design

**Date:** 2026-03-15
**Status:** Approved for planning

## Goal

Extend `vgl` with a stable training-loop surface that supports epoch-level metrics, validation and test evaluation, best-model selection, and optional best-checkpoint persistence without introducing a callback framework or a second trainer family.

## Scope Decisions

- Focus on the training loop, not on operator breadth
- Keep one generic `Trainer` for all current task types
- Add validation and test evaluation to the existing training path
- Add epoch-level metric aggregation
- Add optional best-checkpoint saving through `save_best_path`
- Keep checkpoint scope limited to model weights only
- Prioritize `accuracy`, `loss`, and best-model selection before a larger metric zoo

## Chosen Direction

Three directions were considered:

1. Add a separate `Evaluator` and checkpoint manager next to the current `Trainer`
2. Extend the existing `Trainer` with evaluation, metrics, monitoring, and best-state handling
3. Jump directly to a callback-driven training platform with checkpoint/resume support

The chosen direction is:

> Extend the existing `Trainer` with evaluation, metrics, monitoring, and best-state handling while keeping `Task` and `Metric` contracts explicit and small.

This keeps the public training surface coherent. A separate evaluator object would add ceremony without solving the core contract problem, and a callback-driven system would overfit phase 5.

## Architecture

Phase 5 should keep `Trainer` as the single public lifecycle entrypoint:

```python
trainer = Trainer(
    model=model,
    task=task,
    optimizer=torch.optim.Adam,
    lr=1e-3,
    max_epochs=20,
    metrics=["accuracy"],
    monitor="val_loss",
    monitor_mode="min",
    save_best_path="artifacts/best.pt",
)

history = trainer.fit(train_data, val_data=val_data)
val_result = trainer.evaluate(val_data, stage="val")
test_result = trainer.test(test_data)
```

The core execution path should become:

`data -> model(batch) -> task.loss(...) -> task metric view -> metric aggregation -> epoch summary -> monitor selection`

No second trainer family should be introduced. Node, graph, link, and temporal tasks must continue to share one trainer lifecycle.

## Public API Shape

The `Trainer` constructor should grow only a small number of new knobs:

- `metrics`
- `monitor`
- `monitor_mode`
- `save_best_path`

The public methods should be:

- `fit(train_data, val_data=None)`
- `evaluate(data, stage="val")`
- `test(data)`

`fit(...)` should return structured epoch history instead of only `{"epochs": N}`.

The first stable history shape should include:

```python
{
    "epochs": 20,
    "train": [{"loss": ...}, ...],
    "val": [{"loss": ..., "accuracy": ...}, ...],
    "best_epoch": 7,
    "best_metric": 0.42,
    "monitor": "val_loss",
}
```

This makes examples, tests, and user code stable without forcing users to parse logs.

## Task and Metric Contracts

Phase 5 should keep `Task` responsible for supervision semantics and keep `Metric` responsible for numeric aggregation.

### Task

`Task` should support:

- `loss(batch, predictions, stage)`
- `targets(batch, stage)`
- `predictions_for_metrics(batch, predictions, stage)`

`targets(...)` should return supervision targets aligned to the metric view for that stage.

`predictions_for_metrics(...)` should return predictions in the same supervised shape. This is required because node classification uses stage masks and therefore cannot reuse raw model outputs directly for metric aggregation.

### Metric

`Metric` should become a real streaming contract:

- `reset()`
- `update(predictions, targets)`
- `compute()`

Phase 5 should ship one stable built-in metric first:

- `Accuracy`

This is enough to prove the contract across the current task family without prematurely adding a broad metrics layer.

## Data Flow and Metric Semantics

Trainer metric aggregation should follow one explicit pattern:

1. Run the model on a batch
2. Compute loss through the task
3. Ask the task for metric-aligned predictions and targets
4. Feed those tensors to each metric
5. Aggregate at epoch end

The trainer should not guess task targets or supervision masks.

This keeps the responsibility boundary clear:

- `Task` understands supervision shape
- `Metric` understands metric math
- `Trainer` only orchestrates stages and aggregation

Phase 5 should favor epoch-level aggregation, not batch-level monitoring noise. Monitor keys should refer to aggregated values such as:

- `train_loss`
- `val_loss`
- `val_accuracy`
- `test_accuracy`

## Best Model Selection

Best-model selection should be explicit and deterministic.

Rules:

- If `val_data` is provided and `monitor` is omitted, default to `val_loss`
- If `val_data` is absent and `monitor` is omitted, default to `train_loss`
- If the user requests a `val_*` monitor without `val_data`, fail early
- `monitor_mode` must be `min` or `max` when provided
- If `monitor_mode` is omitted, infer `min` for `*_loss` and `max` otherwise

The trainer should always keep the best state in memory through:

- `best_epoch`
- `best_metric`
- `best_state_dict`

If `save_best_path` is provided, the trainer should also persist the best model weights to that path whenever the monitored value improves.

## Checkpoint Semantics

`save_best_path` should remain narrowly scoped.

Phase 5 checkpoints should:

- store model weights only
- create missing parent directories automatically
- fail early if the provided path is an existing directory

Phase 5 checkpoints should not:

- save optimizer state
- support resume
- introduce callback hooks
- become a generalized artifact system

At the end of `fit(...)`, if a best model was selected, the trainer should restore the in-memory model to `best_state_dict` so that a subsequent `trainer.test(...)` uses best weights by default.

## Error Handling

Phase 5 should fail early when:

- `monitor` points to an unknown stage key or unknown metric key
- a `val_*` monitor is requested without `val_data`
- `monitor_mode` is not `min` or `max`
- metric names are unknown
- a metric returns a non-scalar value
- `task.targets(...)` and `task.predictions_for_metrics(...)` disagree in batch size
- `evaluate()` or `test()` receives no usable batches
- `save_best_path` points to an existing directory

Error messages should make it clear whether the failure comes from task semantics, metric semantics, or trainer configuration.

## Testing Strategy

Phase 5 tests should cover four layers:

### Metric tests

- `Accuracy` handles multiclass logits
- `Accuracy` handles binary logits
- unknown metrics fail early
- shape mismatches fail clearly

### Task contract tests

- node classification exposes masked predictions and masked targets
- graph classification exposes graph-level targets
- link prediction exposes one-logit-per-edge predictions and labels
- temporal event prediction exposes event-level targets

### Trainer behavior tests

- `fit(train, val)` returns structured epoch history
- `evaluate(data, stage)` and `test(data)` do not step the optimizer
- best-model selection works for `val_loss` and `val_accuracy`
- `fit(...)` restores best in-memory weights
- `save_best_path` writes a checkpoint

### Integration and example tests

- a homogeneous node classification example demonstrates the full train/val/test loop
- existing graph, link, and temporal tests remain green
- the updated example surface remains runnable

## Example Surface

Phase 5 should upgrade one example into a full training-loop example first:

- `examples/homo/node_classification.py`

This example should demonstrate:

- train + val + test stages
- metric selection
- best checkpoint saving
- stable `history` usage

Other examples should remain runnable but do not need to become full training-loop showcases in this phase.

## Phase 5 Deliverables

Phase 5 should ship:

- a real metric contract
- built-in `Accuracy`
- `Task` hooks for metric-aligned predictions and targets
- `Trainer.evaluate(...)`
- `Trainer.test(...)`
- epoch-level history with monitor metadata
- in-memory best-state restoration
- optional best-checkpoint saving
- one full node-classification training-loop example
- documentation updates

## Explicit Non-Goals

Do not include:

- callback systems
- early stopping
- learning-rate schedulers
- checkpoint resume
- optimizer-state checkpoints
- a large metric zoo
- experiment tracking integrations

These belong to later phases and should not distort the first stable training-loop contract.

## Repository Touchpoints

Phase 5 will mostly affect:

- `vgl/train/`
- `examples/homo/`
- `tests/train/`
- `tests/integration/`
- `docs/`

## Stability Constraints

Phase 5 should follow four rules:

1. Keep one generic `Trainer`
2. Keep supervision interpretation in `Task`, not in `Trainer`
3. Keep metric aggregation explicit and epoch-level
4. Keep checkpoint semantics narrow and predictable

These constraints preserve room for later callback and scheduler work without forcing a rewrite.

## Acceptance Criteria

Phase 5 is complete when:

1. `Trainer.fit(train, val)` produces structured epoch history
2. `Trainer.evaluate(...)` and `Trainer.test(...)` work on the current task family
3. best-model selection restores the best in-memory weights
4. `save_best_path` persists best weights when requested
5. at least one example demonstrates the full training loop
6. existing node, graph, link, and temporal tests continue to pass

## Next Step

The next step is to create a detailed implementation plan with exact files, tests, commands, and commit checkpoints.
