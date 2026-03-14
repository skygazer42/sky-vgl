# Graph Classification Phase 2 Design

**Date:** 2026-03-15
**Status:** Approved for planning

## Goal

Extend the current GNN framework MVP with a stable graph classification training path that supports both many-small-graph datasets and subgraph classification sampled from a single large graph.

## Scope Decisions

- Focus on `training capability`, not on general operator breadth
- Prioritize `graph classification` before link prediction or temporal prediction
- Support both:
  - many-small-graph datasets
  - subgraph classification sampled from a single large graph
- Support labels from:
  - graph objects
  - sample metadata
- Preserve the existing node classification path without regressions

## Chosen Direction

Three directions were considered:

1. A minimal patch that adds graph classification without changing batch semantics
2. A batch-first graph classification path that unifies both data sources
3. Two parallel input paths with separate batch types

The chosen direction is:

> Use a batch-first graph classification path and unify many-small-graph and sampled-subgraph inputs behind one enhanced `GraphBatch`.

This is the only direction that matches the existing project goals of API stability and unified abstraction. Separate batch paths would move complexity outward into tasks, trainers, and models, and would be difficult to unwind later.

## Architecture

Phase 2 should extend the existing pipeline as follows:

`Dataset/Sampler -> SampleRecord -> Loader -> GraphBatch -> Encoder -> Readout -> Head -> GraphClassificationTask`

The important addition is `SampleRecord`, which captures sample identity, metadata, and source information without forcing all supervision into the graph object itself.

## Core Additions

### SampleRecord

`SampleRecord` should be the smallest structured unit that flows into graph classification collation.

Each record should carry:

- `graph`
- `metadata`
- `sample_id`
- optional `source_graph_id`
- optional `subgraph_seed`

This makes it possible to keep graph-level labels inside the graph for small-graph datasets while also supporting labels that belong to the sample metadata for subgraph sampling.

### Enhanced GraphBatch

The current `GraphBatch` only tracks:

- `graphs`
- `graph_index`

For graph classification, the batch must become more explicit. It should carry at least:

- `graphs`
- `graph_index`
- `graph_ptr`
- `labels`
- `metadata`
- optional `source`

The batch should remain the single object passed to models and tasks.

### GraphClassificationTask

Phase 2 should introduce `GraphClassificationTask` with support for:

- `label_source="graph"`
- `label_source="metadata"`
- optional `label_source="auto"` for convenience

The stable path should be explicit source selection, while `auto` remains a convenience mode rather than the main contract.

### Readout

The framework should add simple graph-level pooling:

- `global_mean_pool`
- `global_sum_pool`
- `global_max_pool`

These should consume node representations plus `graph_index` or equivalent batch membership information.

## Data Paths

### Many-Small-Graph Datasets

- Each dataset item is already a graph sample
- `Loader` collates many graphs into one `GraphBatch`
- Labels usually come from `graph.y`

### Single Large Graph -> Subgraph Samples

- A sampler produces subgraph samples from the large source graph
- Each sampled result becomes a `SampleRecord`
- `Loader` collates sampled records into the same `GraphBatch`
- Labels can come from the subgraph graph object or from sample metadata

The trainer should not distinguish between these origins once batching is complete.

## Public API Shape

The new training API should look like:

```python
task = GraphClassificationTask(
    target="y",
    label_source="graph",
    metrics=["accuracy"],
)
```

or:

```python
task = GraphClassificationTask(
    target="label",
    label_source="metadata",
    metrics=["accuracy"],
)
```

Models should consume a unified batch:

```python
node_repr = encoder(batch)
graph_repr = readout(node_repr, batch.graph_index)
logits = head(graph_repr)
```

## Error Handling

Graph classification should fail early when:

- graph-level labels are missing
- metadata labels are missing when requested
- batch graph counts and label counts disagree
- readout is called without graph membership information
- sampled subgraph records omit identity or source information

These checks are part of the stable API contract and should not be treated as optional convenience features.

## Testing Strategy

Phase 2 tests should cover four layers:

### Contract tests

- `GraphClassificationTask` supports both graph and metadata label sources
- `GraphBatch` exposes graph-level labels and membership information

### Behavior tests

- pooling operators return correct per-graph outputs
- many-small-graph batching keeps membership and labels aligned
- sampled-subgraph batching keeps metadata and labels aligned

### Integration tests

- many-small-graph graph classification runs end to end
- large-graph subgraph classification runs end to end

### Regression tests

- current node classification tests remain green
- current examples remain runnable

## Phase 2 Deliverables

Phase 2 should ship:

- `GraphClassificationTask`
- enhanced `GraphBatch`
- `SampleRecord`
- minimal subgraph sampling entrypoint for graph classification samples
- graph-level pooling utilities
- one many-small-graph end-to-end example
- one large-graph sampled-subgraph end-to-end example
- matching tests and docs

## Explicit Non-Goals for Phase 2

Do not include:

- link prediction
- temporal prediction
- large pooling module libraries
- graph-level multi-task learning
- performance-focused sampling optimization
- broad trainer callback/checkpoint expansion

These belong to later phases and should not distort the graph classification API.

## Repository Touchpoints

Phase 2 will mostly affect:

- `src/gnn/core/batch.py`
- `src/gnn/data/loader.py`
- `src/gnn/data/sampler.py`
- `src/gnn/nn/`
- `src/gnn/train/tasks.py`
- `src/gnn/train/trainer.py`
- `tests/data/`
- `tests/train/`
- `tests/integration/`
- `examples/`

## Stability Constraints

Phase 2 should follow three rules:

1. `Trainer` accepts standardized batch semantics only
2. `GraphClassificationTask` consumes normalized labels and metadata, not raw sampling details
3. graph-level pooling depends on explicit batch membership, never implicit guessing

These constraints keep the training layer coherent as additional task types are added in later phases.

## Acceptance Criteria

Phase 2 is complete when:

1. many-small-graph graph classification trains end to end
2. large-graph sampled-subgraph graph classification trains end to end
3. both graph and metadata label sources work
4. existing node classification tests and examples still pass

## Next Step

The next step is to create a detailed implementation plan with exact files, tests, commands, and commit checkpoints.
