# Core Concepts

## Graph

`Graph` is the canonical graph object in this package. Homogeneous graphs are a special case of the same abstraction used for heterogeneous and temporal graphs.

## GraphView

`GraphView` is a lightweight projection over an existing graph, used for operations such as `snapshot()` and `window()`.

## GraphBatch

`GraphBatch` groups multiple graphs into one training input and tracks node-to-graph membership.

For graph classification it also carries:

- `graph_ptr`
- `labels`
- `metadata`

## MessagePassing

`MessagePassing` is the low-level neural primitive for graph convolutions. `GCNConv`, `SAGEConv`, and `GATConv` build on top of it.

## SampleRecord

`SampleRecord` is the structured pre-collation unit for graph classification. It lets the framework carry:

- a `graph`
- sample metadata
- sample identity
- optional source graph information

This is what makes many-small-graph and sampled-subgraph inputs converge on the same batch contract.

## LinkPredictionRecord and LinkPredictionBatch

`LinkPredictionRecord` is the explicit candidate-edge unit for link prediction training. Each record carries:

- a homogeneous context `graph`
- `src_index`
- `dst_index`
- `label`

`LinkPredictionBatch` collates these records into one model input while keeping link supervision explicit through:

- `graph`
- `src_index`
- `dst_index`
- `labels`

## TemporalEventRecord and TemporalEventBatch

`TemporalEventRecord` is the explicit candidate-event unit for temporal training. Each record carries:

- a temporal `graph`
- `src_index`
- `dst_index`
- `timestamp`
- `label`

`TemporalEventBatch` collates these records into one model input while keeping temporal supervision explicit through:

- `graph`
- `src_index`
- `dst_index`
- `timestamp`
- `labels`

## Readout

Graph classification uses graph-level pooling to convert node representations into graph representations.

The package currently exposes:

- `global_mean_pool`
- `global_sum_pool`
- `global_max_pool`

## Trainer and Task

`Task` defines the supervision contract. It owns:

- `loss(...)`
- `targets(...)`
- `predictions_for_metrics(...)`

`Metric` owns streaming aggregation such as `reset()`, `update(...)`, and `compute()`.

`Trainer` runs the optimization loop without taking ownership of the core graph abstraction. It owns:

- `fit(train, val=None)`
- `evaluate(data, stage="val")`
- `test(data)`
- monitor-based best-model selection
- optional best-checkpoint saving

The current training layer supports:

- node classification
- graph classification with labels from graph objects
- graph classification with labels from sample metadata
- link prediction from explicit candidate-edge samples
- temporal event prediction from explicit candidate-event samples
