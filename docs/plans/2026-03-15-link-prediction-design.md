# Link Prediction Phase 4 Design

**Date:** 2026-03-15
**Status:** Approved for planning

## Goal

Extend `vgl` with a stable homogeneous link prediction training path that uses explicit candidate edges as the supervision contract while preserving the existing `Graph` abstraction and generic `Trainer`.

## Scope Decisions

- Focus on `link prediction`, not ranking or retrieval
- Focus on homogeneous / single-edge-type graphs only
- Use explicit candidate edges `(src, dst, label)` as the dataset contract
- Keep one canonical `Graph` abstraction
- Keep `Trainer` unchanged
- Use binary `0/1` supervision only in the first version
- Support batches from a single source graph only

## Chosen Direction

Three directions were considered:

1. Reuse `GraphBatch` and place link labels inside `metadata`
2. Add dedicated link prediction sample, batch, and task contracts while keeping `Trainer` unchanged
3. Reframe link prediction as graph classification over synthetic per-edge graphs

The chosen direction is:

> Add `LinkPredictionRecord`, `LinkPredictionBatch`, and `LinkPredictionTask` while preserving the existing `Graph` abstraction and generic `Trainer`.

This keeps the supervision contract explicit and stable. Hiding link semantics inside `metadata` would make the API fragile, and pretending link prediction is graph classification would distort both concepts.

## Architecture

Phase 4 should extend the current pipeline as follows:

`Dataset / Sampler -> LinkPredictionRecord -> Loader -> LinkPredictionBatch -> Model -> LinkPredictionTask -> Trainer`

The graph core remains unchanged. Message passing still runs on one canonical `Graph`, and supervision lives in the batch as candidate edges rather than in a second graph family.

## Public API Shape

The public API should remain structurally aligned with the existing task surface:

```python
task = LinkPredictionTask(
    target="label",
    loss="binary_cross_entropy",
)

trainer = Trainer(
    model=model,
    task=task,
    optimizer=torch.optim.Adam,
    lr=1e-3,
    max_epochs=5,
)

trainer.fit(loader)
```

Models should consume one normalized batch object:

```python
logits = model(batch)
loss = task.loss(batch, logits, stage="train")
```

`LinkPredictionTask` should only interpret supervision. It should not own negative sampling, graph rewriting, or decoder policy.

## Core Additions

### LinkPredictionRecord

`LinkPredictionRecord` should be the smallest structured unit that flows into link prediction collation.

Each record should carry at least:

- `graph`
- `src_index`
- `dst_index`
- `label`
- optional `metadata`
- optional `sample_id`

This keeps supervision explicit and avoids hiding core training fields inside loose dictionaries.

### LinkPredictionBatch

`LinkPredictionBatch` should be the single object passed to models and tasks for link prediction.

The batch should expose at least:

- `graph`
- `src_index`
- `dst_index`
- `labels`
- optional `metadata`

The first implementation should support the common case where all samples in a batch come from one source graph only.

### LinkPredictionTask

`LinkPredictionTask` should be the supervision contract for binary link prediction over explicit candidate edges.

The task should:

- accept explicit labeled edge batches
- compute binary classification loss from logits
- reject unsupported loss modes early

Ranking losses, automatic negative sampling, and multi-class edge objectives are out of scope for this phase.

## Data Flow

Phase 4 should use explicit candidate-edge samples.

Each dataset item or sampler output should describe one candidate edge:

- source node index
- destination node index
- binary label
- optional metadata

Loader collation should preserve these fields in `LinkPredictionBatch` without mutating graph structure or generating additional supervision.

## Batch Semantics

The first batch design should favor explicit semantics and API stability over hidden automation.

For a batch of explicit candidate edges:

- retain the shared context graph reference
- store vectorized `src_index`, `dst_index`, and `labels`
- optionally store `metadata`
- require every record in a batch to share the same `graph`

The meaning of `graph` is important:

- `graph` is the context graph used for message passing
- `(src_index, dst_index)` is the candidate edge to score
- the candidate edge may or may not exist inside `graph`

Phase 4 should not perform automatic negative sampling, automatic edge removal, or per-edge subgraph extraction.

## Leakage Boundary

`LinkPredictionBatch.graph` should be treated as the context graph, not as a guaranteed full-truth graph.

For many training setups, positive supervision edges should already be removed from the context graph to avoid label leakage. Phase 4 should document this expectation clearly, but the framework should not try to guess or automatically repair leakage-prone inputs.

This keeps the first stable version explicit and predictable.

## Error Handling

Validation should be split across the data and task layers.

### Batch Construction Validation

`LinkPredictionBatch.from_records(...)` should fail early when:

- the record list is empty
- the batch mixes multiple source graphs
- `src_index` or `dst_index` falls outside the graph node range
- field lengths disagree
- labels are not binary `0/1`
- metadata length does not align with batch size

### Loader Validation

`Loader` should only route by type:

- if items are `LinkPredictionRecord`, collate them into `LinkPredictionBatch`
- do not repair malformed records
- do not add negative samples
- do not mutate graph structure

### Task Validation

`LinkPredictionTask` should fail early when:

- an unsupported loss mode is requested
- model output shape does not match batch size
- model output is not interpretable as one logit per candidate edge

The first version should use a binary classification path based on logits plus BCE-style loss, not a ranking objective.

## Testing Strategy

Phase 4 tests should cover four layers:

### Core contract tests

- `LinkPredictionRecord` and `LinkPredictionBatch` shape and validation guarantees
- empty record failures
- mixed-graph failures
- out-of-range index failures
- non-binary label failures

### Data tests

- loader dispatch for `LinkPredictionRecord -> LinkPredictionBatch`
- no accidental fallback to `GraphBatch`

### Task and trainer tests

- `LinkPredictionTask` computes loss for valid batches
- invalid loss modes fail early
- model output shape mismatches fail clearly
- `Trainer.fit(loader)` runs one link prediction epoch end to end

### Integration and example tests

- a real homogeneous link prediction example runs successfully
- existing node classification, graph classification, and temporal event prediction paths continue to pass

## Example Surface

Phase 4 should add one concrete example first:

- `examples/homo/link_prediction.py`

The example should:

- build a small homogeneous graph
- create explicit candidate edges with binary labels
- use `Loader(ListDataset(...), FullGraphSampler(), batch_size=...)`
- define a small model that encodes `batch.graph`, gathers `src/dst` node features, and emits one logit per edge
- train for one epoch with `Trainer.fit(...)`

Phase 4 should not add heterogenous, temporal, or ranking-based link prediction examples.

## Phase 4 Deliverables

Phase 4 should ship:

- `LinkPredictionRecord`
- `LinkPredictionBatch`
- `LinkPredictionTask`
- loader support for explicit link prediction samples
- one real homogeneous link prediction example
- matching unit and integration tests
- package export updates
- documentation updates

## Explicit Non-Goals

Do not include:

- heterogeneous link prediction
- automatic negative sampling
- ranking losses or ranking metrics
- multi-graph link prediction collation
- per-edge subgraph extraction
- automatic leakage detection or edge removal

These belong to later phases and should not distort the first stable training contract.

## Repository Touchpoints

Phase 4 will mostly affect:

- `vgl/core/`
- `vgl/data/`
- `vgl/train/`
- `examples/homo/`
- `tests/core/`
- `tests/data/`
- `tests/train/`
- `tests/integration/`
- `docs/`

## Stability Constraints

Phase 4 should follow four rules:

1. Keep one canonical `Graph` abstraction
2. Keep supervision fields explicit rather than burying them in `metadata`
3. Keep `Trainer` generic and unchanged
4. Keep leakage handling explicit in data construction rather than hidden inside framework logic

These constraints preserve API coherence as later phases add more edge prediction objectives.

## Acceptance Criteria

Phase 4 is complete when:

1. explicit candidate-edge samples train through `Trainer` end to end
2. link prediction batches expose stable public fields
3. binary link prediction loss works on one-logit-per-edge outputs
4. existing node, graph, and temporal training tests still pass
5. the homogeneous link prediction example runs as a real training path

## Next Step

The next step is to create a detailed implementation plan with exact files, tests, commands, and commit checkpoints.
