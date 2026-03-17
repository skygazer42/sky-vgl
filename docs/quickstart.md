# Quickstart

`vgl` is a PyTorch-first graph learning package with one core `Graph` abstraction.

Preferred imports follow the domain layout:

```python
from vgl.dataloading import DataLoader, FullGraphSampler, ListDataset, SampleRecord
from vgl.engine import (
    CHECKPOINT_FORMAT,
    Callback,
    EarlyStopping,
    HistoryLogger,
    StopTraining,
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
)
history = trainer.fit(graph, val_data=graph)
test_result = trainer.test(graph)
best_state = load_checkpoint("artifacts/best.pt")
restored = restore_checkpoint(model, "artifacts/best.pt")
```

`trainer.fit(...)` returns a `TrainingHistory` object with epoch summaries, monitor metadata, and early-stop state.

For alternative homogeneous backbones, the same training path can swap in `GINConv`, `GATv2Conv`, `APPNPConv`, `TAGConv`, `SGConv`, `ChebConv`, `AGNNConv`, `LightGCNConv`, `LGConv`, `FAGCNConv`, `ARMAConv`, `GPRGNNConv`, `MixHopConv`, `BernConv`, `SSGConv`, `DAGNNConv`, `GCN2Conv`, `GraphConv`, `H2GCNConv`, `EGConv`, `LEConv`, `ResGatedGraphConv`, `GatedGraphConv`, `ClusterGCNConv`, `GENConv`, `FiLMConv`, `SimpleConv`, `EdgeConv`, `FeaStConv`, `MFConv`, `PNAConv`, `GeneralConv`, `AntiSymmetricConv`, `TransformerConv`, `WLConvContinuous`, `SuperGATConv`, or `DirGNNConv` inside the model definition.

For deeper equal-width stacks, you can also wrap operators such as `LGConv` with `GroupRevRes`.

For training-time hooks, pass callback objects to `Trainer(callbacks=[...])`. Implement `on_epoch_end(...)` for logging or checkpoint policy, inspect the shared `TrainingHistory` object inside callbacks, raise `StopTraining` when you want early stopping, or use the built-in `EarlyStopping` and `HistoryLogger` callbacks directly.

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
