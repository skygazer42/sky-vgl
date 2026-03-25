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

For advanced systems work, the new foundation layers sit underneath the same surface API:

- `vgl.sparse` for cached COO/CSR/CSC adjacency layouts, transpose/reduction helpers, and sparse operators
- `vgl.storage` for feature / graph stores, mmap-backed feature tensors, and `Graph.from_storage(...)` with retained feature-source context
- `vgl.ops` for reusable graph transforms, homogeneous/heterogeneous relation-local subgraph extraction, and compaction
- `vgl.data` for dataset manifests, cache helpers, built-in datasets, and manifest-backed homo/hetero/temporal on-disk datasets with lazy per-item payloads and split views
- `vgl.distributed` for partition metadata, local shard loading, typed node routing, relation-scoped edge routing, edge feature fetches, partition graph queries, sampling coordination contracts, and routed plan feature sources across homogeneous, temporal homogeneous, single-node-type multi-relation, and multi-node-type heterogeneous graphs

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

For debug loops and experiment bookkeeping, `Trainer` also accepts:

```python
trainer = Trainer(
    ...,
    default_root_dir="artifacts/debug-run",
    run_name="sanity-check",
    fast_dev_run=True,
    num_sanity_val_steps=2,
    val_check_interval=0.5,
    profiler="simple",
)
```

`default_root_dir` becomes the base for relative checkpoint and logger paths, `run_name` is carried into structured records plus `TrainingHistory`, `fast_dev_run` trims every stage to a tiny sample, forces one epoch, and suppresses automatic checkpoint writes, `num_sanity_val_steps` runs validation before training begins, `val_check_interval` can insert mid-epoch validation during `fit(...)`, and `profiler="simple"` attaches coarse timing totals to fit/epoch summaries. When you need deterministic stage caps without enabling fast-dev mode, `limit_train_batches`, `limit_val_batches`, and `limit_test_batches` accept either absolute batch counts or fractions.

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

For heterogeneous graph classification, keep the same `batch.graphs` loop but pool with per-node-type membership such as `batch.graph_index_by_type["paper"]` or `batch.graph_index_by_type["author"]`.

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


## Advanced Foundation Workflows

### Storage-backed Graphs

```python
import torch
from vgl.graph import GraphSchema, Graph
from vgl.storage import FeatureStore, InMemoryGraphStore, MmapTensorStore

edge_type = ("node", "to", "node")
schema = GraphSchema(
    node_types=("node",),
    edge_types=(edge_type,),
    node_features={"node": ("x",)},
    edge_features={edge_type: ("edge_index",)},
)
MmapTensorStore.save("artifacts/features/x.bin", torch.randn(4, 16))
feature_store = FeatureStore({
    ("node", "node", "x"): MmapTensorStore("artifacts/features/x.bin"),
})
graph_store = InMemoryGraphStore(
    edges={edge_type: torch.tensor([[0, 1, 2], [1, 2, 3]])},
    num_nodes={"node": 4},
)
graph = Graph.from_storage(schema=schema, feature_store=feature_store, graph_store=graph_store)

# edge structure is ready immediately; x is resolved from the feature store on first access
# graph.feature_store retains the originating source for later plan execution
node_features = graph.x
adjacency = graph.adjacency(layout="coo")
```

When a later `SamplingPlan` includes feature-fetch stages, `PlanExecutor.execute(..., graph=graph)` and `Loader(..., sampler=...)` will reuse `graph.feature_store` automatically unless you pass an explicit `feature_store=` override. For sampled node, link, and temporal workloads, `NodeNeighborSampler(node_feature_names=..., edge_feature_names=...)`, `LinkNeighborSampler(...)`, and `TemporalNeighborSampler(...)` can append those fetch stages opt-in and materialize the fetched slices back into each sampled subgraph. Use dictionaries keyed by node type / edge type when the sampled graph is heterogeneous.

### On-disk Datasets

```python
from vgl.data.catalog import DatasetManifest, DatasetSplit
from vgl.data.ondisk import OnDiskGraphDataset

manifest = DatasetManifest(
    name="toy-graph",
    version="1.0",
    splits=(DatasetSplit("train", size=len(graphs)),),
)
OnDiskGraphDataset.write("artifacts/toy", manifest, graphs)
dataset = OnDiskGraphDataset("artifacts/toy")
train_dataset = dataset.split("train")

# new writes store one payload per graph under artifacts/toy/graphs/graph-*.pt
# graphs may contain Graph.homo(...), Graph.hetero(...), or Graph.temporal(...)
# older artifacts/toy/graphs.pt datasets remain readable
first_graph = train_dataset[0]
```

### Local Partition and Shard Flows

```python
from vgl.distributed import LocalGraphShard, LocalSamplingCoordinator, write_partitioned_graph

# graph can be Graph.homo(...), Graph.temporal(...), or Graph.hetero(...) with one or many node types
manifest = write_partitioned_graph(graph, "artifacts/partitions", num_partitions=2)
shard = LocalGraphShard.from_partition_dir("artifacts/partitions", partition_id=0)
coordinator = LocalSamplingCoordinator({0: shard})

local_graph = shard.graph
global_edge_index = shard.global_edge_index(edge_type=("node", "follows", "node"))
partition_node_ids = coordinator.partition_node_ids(0, node_type="paper")
partition_edge_ids = coordinator.partition_edge_ids(0, edge_type=("author", "writes", "paper"))
edge_weights = coordinator.fetch_edge_features(
    ("edge", ("author", "writes", "paper"), "weight"),
    partition_edge_ids,
).values
partition_adjacency = coordinator.fetch_partition_adjacency(0, edge_type=("node", "follows", "node"), layout="csr")
```

Plan-backed feature fetch stages can also use the same routed source directly through `PlanExecutor.execute(..., feature_store=coordinator)` or `Loader(..., feature_store=coordinator)` when you want executor-driven feature access instead of direct store access. Those explicit arguments remain the highest-priority override; otherwise, storage-backed graphs can supply the same context through their retained `graph.feature_store`. Once node sampling opts into those stages, the fetched node and edge tensors are aligned to the sampled subgraph and exposed through `batch.graph` like ordinary in-graph features.

These advanced paths are still designed to terminate in the same public training contracts: `Graph`, batch objects from `Loader`, and `Trainer.fit/evaluate/test`.
