# Quickstart

`vgl` is a PyTorch-first graph learning package with one core `Graph` abstraction.

Preferred imports follow the domain layout:

```python
from vgl.dataloading import DataLoader, FullGraphSampler, ListDataset, SampleRecord
from vgl.engine import (
    CHECKPOINT_FORMAT,
    Callback,
    CSVLogger,
    EarlyStopping,
    HistoryLogger,
    JSONLinesLogger,
    StopTraining,
    TensorBoardLogger,
    TrainingHistory,
    Trainer,
    load_checkpoint,
    restore_checkpoint,
)
from vgl.graph import Graph
from vgl.tasks import GraphClassificationTask, NodeClassificationTask
```

Legacy `vgl.data` and `vgl.train` paths still work, but new code should prefer the package layout above.

The smallest workflow is:

1. Build a `Graph`
2. Define a `Task`
3. Build a PyTorch model
4. Train it with `Trainer`

For a homogeneous graph:

```python
graph = Graph.homo(
    edge_index=edge_index,
    x=x,
    y=y,
    train_mask=train_mask,
    val_mask=val_mask,
    test_mask=test_mask,
)
task = NodeClassificationTask(
    target="y",
    split=("train_mask", "val_mask", "test_mask"),
    metrics=["accuracy"],
)
trainer = Trainer(
    model=model,
    task=task,
    optimizer=torch.optim.Adam,
    lr=1e-3,
    max_epochs=10,
    monitor="val_accuracy",
    save_best_path="artifacts/best.pt",
    loggers=[JSONLinesLogger("artifacts/train.jsonl", flush=True)],
    log_every_n_steps=10,
)
history = trainer.fit(graph, val_data=graph)
test_result = trainer.test(graph)
best_state = load_checkpoint("artifacts/best.pt")
restored = restore_checkpoint(model, "artifacts/best.pt")
```

`Trainer` enables console logging by default, emitting step progress plus epoch/final summaries. Add `loggers=[JSONLinesLogger(...)]` when you also want structured event logs on disk, `loggers=[CSVLogger(...)]` when you want one CSV row per epoch, `loggers=[TensorBoardLogger(...)]` when you want TensorBoard scalars, set `log_every_n_steps` to control training-step emission frequency, or disable terminal output with `enable_console_logging=False`.

For quieter terminal output, configure the default console logger through `Trainer`:

```python
trainer = Trainer(
    ...,
    enable_progress_bar=False,
    console_mode="compact",
    console_theme="cat",
    console_metric_names={"loss", "train_loss", "val_loss"},
    console_show_learning_rate=False,
    console_show_events=False,
)
```

Console logs include `HH:MM:SS` timestamps by default. Detailed mode starts with a run summary card for model/task/optimizer metadata and parameter counts, emits stage-start lines when training / validation / testing begin, shows `tqdm`-style batch progress, percentage, throughput, and ETA during training steps, adds fit-level progress such as `fit=3/10 (30.0%)` plus `fit_eta=...` in epoch summaries, and finishes with aggregate speed fields such as `avg_epoch_time=...` and `avg_steps_per_second=...`. Set `console_theme="cat"` when you want an ASCII status mascot for phases such as `starting`, `waiting`, `training`, `validating`, `testing`, `tracking`, `saving`, and `done`, with distinct cat faces per phase plus a small ASCII progress bar during training steps, or set `console_show_timestamp=False` when you want to suppress the time prefix.

`JSONLinesLogger` can filter events when you only want coarse-grained summaries:

```python
epoch_logger = JSONLinesLogger(
    "artifacts/epochs.jsonl",
    events={"epoch_end", "fit_end"},
    flush=True,
)
```

Training-step and epoch-end records include current optimizer learning-rate fields such as `lr`.

The structured logger stream also includes lifecycle events such as `monitor_improved` and `checkpoint_saved`. `monitor_improved` records include `previous_best`, `current_value`, and `improvement_delta` so you can track how much the monitored score moved, while `checkpoint_saved` records include `size_bytes` and `save_seconds`. The initial `fit_start` record carries run metadata including model name, task name, optimizer name, callback/logger names, and parameter counts.

To keep file logs small, both `JSONLinesLogger` and `CSVLogger` support metric filtering and reduced context:

```python
minimal_logger = JSONLinesLogger(
    "artifacts/minimal.jsonl",
    events={"epoch_end", "fit_end"},
    metric_names={"train_loss", "val_loss"},
    include_context=False,
    show_learning_rate=False,
    flush=True,
)
```

`show_learning_rate=False` hides `lr` / `lr/group_*` fields, while `include_context=False` keeps only the core event coordinates plus the filtered `metrics` payload.

For TensorBoard:

```python
tb_logger = TensorBoardLogger(
    "artifacts/tensorboard",
    events={"train_step", "epoch_end", "fit_end"},
    show_learning_rate=False,
    flush=True,
)
```

Launch TensorBoard with `tensorboard --logdir artifacts/tensorboard`. `TensorBoardLogger` requires the optional `tensorboard` package.

`trainer.fit(...)` returns a `TrainingHistory` object with epoch summaries, monitor metadata, elapsed time fields, and early-stop state.

For alternative homogeneous backbones, the same training path can swap in `GINConv`, `GATv2Conv`, `APPNPConv`, `TAGConv`, `SGConv`, `ChebConv`, `AGNNConv`, `LightGCNConv`, `LGConv`, `FAGCNConv`, `ARMAConv`, `GPRGNNConv`, `MixHopConv`, `BernConv`, `SSGConv`, `DAGNNConv`, `GCN2Conv`, `GraphConv`, `H2GCNConv`, `EGConv`, `LEConv`, `ResGatedGraphConv`, `GatedGraphConv`, `ClusterGCNConv`, `GENConv`, `FiLMConv`, `SimpleConv`, `EdgeConv`, `FeaStConv`, `MFConv`, `PNAConv`, `GeneralConv`, `AntiSymmetricConv`, `TransformerConv`, `WLConvContinuous`, `SuperGATConv`, or `DirGNNConv` inside the model definition.

For deeper equal-width stacks, you can also wrap operators such as `LGConv` with `GroupRevRes`.

For training-time hooks, pass callback objects to `Trainer(callbacks=[...])`. Implement `on_epoch_end(...)` for checkpoint policy or control-flow changes, inspect the shared `TrainingHistory` object inside callbacks, raise `StopTraining` when you want early stopping, or use the built-in `EarlyStopping` and `HistoryLogger` callbacks directly. Prefer `loggers=[...]` over callbacks for new logging/reporting integrations.

For graph classification over many small graphs:

```python
samples = [
    SampleRecord(graph=graph_a, metadata={}, sample_id="a"),
    SampleRecord(graph=graph_b, metadata={}, sample_id="b"),
]
loader = DataLoader(
    dataset=ListDataset(samples),
    sampler=FullGraphSampler(),
    batch_size=2,
    label_source="graph",
    label_key="y",
)
task = GraphClassificationTask(target="y", label_source="graph")
trainer = Trainer(model=model, task=task, optimizer=torch.optim.Adam, lr=1e-3, max_epochs=10)
trainer.fit(loader)
```

For graph classification from sampled subgraphs of a larger graph:

```python
dataset = ListDataset([
    (source_graph, {"seed": 1, "label": 1, "sample_id": "s1"}),
    (source_graph, {"seed": 2, "label": 0, "sample_id": "s2"}),
])
loader = DataLoader(
    dataset=dataset,
    sampler=NodeSeedSubgraphSampler(),
    batch_size=2,
    label_source="metadata",
    label_key="label",
)
task = GraphClassificationTask(target="label", label_source="metadata")
trainer.fit(loader)
```

For homogeneous link prediction from explicit candidate edges:

```python
graph = Graph.homo(edge_index=edge_index, x=x)
samples = [
    LinkPredictionRecord(graph=graph, src_index=0, dst_index=1, label=1),
    LinkPredictionRecord(graph=graph, src_index=2, dst_index=0, label=0),
]
loader = DataLoader(
    dataset=ListDataset(samples),
    sampler=FullGraphSampler(),
    batch_size=2,
)
task = LinkPredictionTask(target="label")
trainer = Trainer(model=model, task=task, optimizer=torch.optim.Adam, lr=1e-3, max_epochs=10)
trainer.fit(loader)
```

For temporal event prediction from explicit candidate-event samples:

```python
graph = Graph.temporal(nodes=nodes, edges=edges, time_attr="timestamp")
samples = [
    TemporalEventRecord(graph=graph, src_index=0, dst_index=1, timestamp=3, label=1),
    TemporalEventRecord(graph=graph, src_index=2, dst_index=0, timestamp=5, label=0),
]
loader = DataLoader(
    dataset=ListDataset(samples),
    sampler=FullGraphSampler(),
    batch_size=2,
)
task = TemporalEventPredictionTask(target="label")
trainer = Trainer(model=model, task=task, optimizer=torch.optim.Adam, lr=1e-3, max_epochs=10)
trainer.fit(loader)
```
