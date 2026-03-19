# Core Concepts

## Graph

`Graph` is the canonical graph object in this package. Homogeneous graphs are a special case of the same abstraction used for heterogeneous and temporal graphs.

Homogeneous graphs can carry edge-level tensors through `Graph.homo(edge_data={...})`. These tensors are exposed through `graph.edata` and are what edge-aware operators consume.

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

Built-in homogeneous convolution layers live under `vgl.nn.conv`. The current set includes `GCNConv`, `SAGEConv`, `GATConv`, `GINConv`, `GATv2Conv`, `APPNPConv`, `TAGConv`, `SGConv`, `ChebConv`, `AGNNConv`, `LightGCNConv`, `LGConv`, `FAGCNConv`, `ARMAConv`, `GPRGNNConv`, `MixHopConv`, `BernConv`, `SSGConv`, `DAGNNConv`, `GCN2Conv`, `GraphConv`, `H2GCNConv`, `EGConv`, `LEConv`, `ResGatedGraphConv`, `GatedGraphConv`, `ClusterGCNConv`, `GENConv`, `FiLMConv`, `SimpleConv`, `EdgeConv`, `FeaStConv`, `MFConv`, `PNAConv`, `GeneralConv`, `AntiSymmetricConv`, `TransformerConv`, `WLConvContinuous`, `SuperGATConv`, and `DirGNNConv`.

For heterogeneous graphs, the package also exposes relation-aware operators such as `RGCNConv`, `HGTConv`, and `HANConv`.

For homogeneous graphs with edge features, the package also exposes edge-aware operators such as `NNConv`, `ECConv`, `GINEConv`, and `GMMConv`.

`GroupRevRes` lives under `vgl.nn` and wraps equal-width homogeneous operators into a grouped reversible residual block.

## Graph Transformer Encoders

Beyond message-passing layers, `vgl.nn` also exposes graph transformer encoder blocks:

- `GraphTransformerEncoderLayer` and `GraphTransformerEncoder`
- `GraphormerEncoderLayer` and `GraphormerEncoder`
- `GPSLayer`
- `NAGphormerEncoder`

These modules keep the same lightweight graph contract and can be dropped into node-level models without changing the training pipeline.

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
- `TrainingHistory` as the structured return type from `fit(...)`
- `evaluate(data, stage="val")`
- `test(data)`
- `load_checkpoint(path, map_location=None, weights_only=True)`
- `restore_checkpoint(model, path, map_location=None, strict=True, weights_only=True)`
- optional `loggers=[...]` with built-in `ConsoleLogger`, `JSONLinesLogger`, `CSVLogger`, and `TensorBoardLogger`
- `log_every_n_steps` for training-step event frequency control
- console controls such as `console_mode`, `console_theme`, `console_metric_names`, `console_show_learning_rate`, `console_show_events`, and `console_show_timestamp`
- optional `on_stage_start(stage_record)` logger hook before each train / val / test phase
- detailed console mode with a fit-start run summary card, explicit stage-start markers, `tqdm`-style training-step progress and ETA, fit-level progress / ETA in epoch summaries, and final average speed fields
- structured logger controls such as `events`, `metric_names`, and `show_learning_rate`, plus `include_context` on JSON/CSV loggers
- structured lifecycle events such as `monitor_improved` and `checkpoint_saved`, with monitor-improvement deltas plus checkpoint size/save-time metadata available in event records
- optional `callbacks=[...]` with `on_fit_start`, `on_epoch_end`, and `on_fit_end` hooks
- `StopTraining` for callback-driven early stopping
- built-in `EarlyStopping` and `HistoryLogger` callbacks
- monitor-based best-model selection
- optional best-checkpoint saving

The `vgl.engine` package also exposes module-level checkpoint helpers and format constants:

- `ConsoleLogger`
- `CSVLogger`
- `JSONLinesLogger`
- `TensorBoardLogger`
- `Logger`
- `save_checkpoint(...)`
- `load_checkpoint(...)`
- `restore_checkpoint(...)`
- `TrainingHistory`
- `CHECKPOINT_FORMAT`
- `CHECKPOINT_FORMAT_VERSION`

The current training layer supports:

- node classification
- graph classification with labels from graph objects
- graph classification with labels from sample metadata
- link prediction from explicit candidate-edge samples
- temporal event prediction from explicit candidate-event samples
