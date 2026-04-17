# Public Surface Contract

This internal contract document records the public install, import, and runtime guarantees that packaging scans, docs scans, and smoke tests expect from a release.

Install the published distribution with `pip install sky-vgl`. Add extras such as `pip install "sky-vgl[networkx]"`, `pip install "sky-vgl[scipy]"`, `pip install "sky-vgl[tensorboard]"`, `pip install "sky-vgl[dgl]"`, `pip install "sky-vgl[pyg]"`, or `pip install "sky-vgl[full]"` when you need those optional integrations. Public releases target Python 3.10+, while imports remain under `vgl`.

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

`GraphBatch` is the canonical batched graph container for graph-level training inputs. `GraphView` is the canonical read-only graph projection for snapshot/window-style access. `NodeStore` and `EdgeStore` are lower-level storage-facing graph internals; prefer `Graph`, `GraphView`, and `GraphBatch` in application code.

## Artifact Metadata Contract

Serialized graph and trainer-checkpoint artifacts carry explicit format metadata so releases can detect incompatible payload changes instead of guessing from file layout.

- `Graph.artifact_metadata()` emits `format="vgl.graph"` and `format_version=1`.
- `save_checkpoint(...)` emits `format="vgl.trainer_checkpoint"` and `format_version=1`.
- Legacy raw `state_dict` checkpoints still normalize through `load_checkpoint(...)` / `restore_checkpoint(...)` as `format="legacy.state_dict"` with `format_version=0`.
- Format metadata keys are reserved; callers may add extra metadata, but they must not overwrite `format` or `format_version`.

Minimal checkpoint round-trip:

```python
from vgl.engine import (
    CHECKPOINT_FORMAT,
    CHECKPOINT_FORMAT_VERSION,
    load_checkpoint,
    save_checkpoint,
)

save_checkpoint("artifacts/best.pt", model.state_dict(), metadata={"epoch": 3})
payload = load_checkpoint("artifacts/best.pt")
assert payload["format"] == CHECKPOINT_FORMAT
assert payload["format_version"] == CHECKPOINT_FORMAT_VERSION
```

Minimal graph artifact metadata:

```python
from vgl import Graph

graph = Graph.homo(edge_index=edge_index, x=x)
metadata = graph.artifact_metadata()
assert metadata["format"] == "vgl.graph"
assert metadata["format_version"] == 1
```

## Root Export Tiers

`vgl root exports follow stable, compatibility, and internal tiers`.

VGL treats `from vgl import ...` as a tiered contract so root exports stay intentional instead of growing by accident.

- Stable root exports: the canonical quickstart names kept at the front of `vgl.__all__`. This tier is the public root-level contract for docs, smoke tests, and examples.
- Compatibility-only root exports: additional root symbols kept for existing callers, but new code should prefer the owning domain module directly. Examples include `Loader`, `SampleRecord`, and `RandomWalkSampler`.
- Internal APIs live outside the root namespace by design.
- Internal modules are intentionally not re-exported from `vgl`. Import those APIs from their owning packages such as `vgl.ops`, `vgl.storage`, `vgl.distributed`, `vgl.transforms`, `vgl.metrics`, and `vgl.nn`.

The stable root tier is:

```python
from vgl import (
    DataLoader,
    DatasetRegistry,
    Graph,
    GraphBatch,
    GraphClassificationTask,
    KarateClubDataset,
    LinkPredictionTask,
    MessagePassing,
    NodeClassificationTask,
    PlanetoidDataset,
    TUDataset,
    TemporalEventPredictionTask,
    Trainer,
    __version__,
)
```

For public datasets and preprocessing, VGL now exposes:

```python
from vgl.data import KarateClubDataset, PlanetoidDataset, TUDataset
from vgl.transforms import Compose, NormalizeFeatures, RandomNodeSplit

cora = PlanetoidDataset(root="data", name="Cora", transform=Compose([NormalizeFeatures()]))
karate = KarateClubDataset(
    root="data",
    transform=Compose([RandomNodeSplit(num_train_per_class=2, num_val=4, num_test=4, seed=0)]),
)
mutag = TUDataset(root="data", name="MUTAG")
```

`PlanetoidDataset` supports `Cora`, `Citeseer`, and `PubMed`; `TUDataset` supports standard TU graph-classification collections such as `MUTAG`, `PROTEINS`, and `ENZYMES`; and `KarateClubDataset` provides a zero-download starter graph. TU optional node labels, edge labels, and edge attributes are preserved as `graph.ndata["node_label"]`, `graph.edata["edge_label"]`, and `graph.edata["edge_attr"]`. `Compose`, `RandomNodeSplit`, `RandomGraphSplit`, `NormalizeFeatures`, `ToUndirected`, `AddSelfLoops`, `RemoveSelfLoops`, `LargestConnectedComponents`, `FeatureStandardize`, and `TrainOnlyFeatureNormalizer` cover the common preprocessing path.

Split semantics stay explicit: `PlanetoidDataset` keeps the original citation masks, `KarateClubDataset` is typically paired with `RandomNodeSplit(...)`, and graph collections such as `TUDataset` should be split with `RandomGraphSplit(...)` when you need train/val/test partitions.

For string-driven dataset creation, `DatasetRegistry.default()` also exposes aliases such as `cora`, `citeseer`, `pubmed`, `mutag`, `proteins`, and `enzymes`, accepts common hyphen/underscore variants such as `karate_club` or `imdb_binary`, now also accepts compact aliases such as `karateclub` or `imdbbinary`, and accepts family-prefixed forms such as `planetoid:cora`, `planetoid/cora`, `planetoid.cora`, `planetoid_pubmed`, `tu:proteins`, `tu/proteins`, `tu.proteins`, `tu.imdbbinary`, or `tu-imdbbinary`.

For additional subgraph sampling workflows, `vgl.dataloading` now also exports `RandomWalkSampler`, `Node2VecWalkSampler`, `GraphSAINTNodeSampler`, `GraphSAINTEdgeSampler`, `GraphSAINTRandomWalkSampler`, `ClusterData`, `ClusterLoader`, and `ShaDowKHopSampler`.

`RandomWalkSampler(..., num_walks=k)` and `Node2VecWalkSampler(..., num_walks=k)` keep the single-walk path behavior by default, but can now emit multiple walks for one scalar `metadata["seed"]` or use an explicit seed collection such as `metadata["seed"] = [0, 3, 8]`. They also expose `walk_starts`, `walk_start_positions`, `walk_nodes`, `walk_lengths`, `walk_ended_early`, `num_walks_ended_early`, `sampled_num_walks`, and `walk_edge_pairs` metadata alongside `walks`, so walk-oriented pipelines can inspect exact starts, where those starts landed inside the sampled subgraph, the unique visited node set, each walk's effective non-padding length, whether each walk terminated early at a dead end, the aggregate count of early-terminated walks, the actual number of traced walks without inferring it from nested list shapes, and the traversed edge sequence in original node-id space. `GraphSAINTRandomWalkSampler` follows the same `walk_lengths`, `walk_ended_early`, `num_walks_ended_early`, `sampled_num_walks`, `walk_start_positions`, and `walk_edge_pairs` conventions for its traced walks. When you want those explicit multi-seed inputs to materialize a `NodeBatch` instead of one graph sample, pass `expand_seeds=True`; VGL will then expand them into multiple `SampleRecord`s that share one sampled walk subgraph.

The GraphSAINT samplers also accept explicit control metadata: `GraphSAINTNodeSampler` can force one or more seed nodes into the sampled subgraph, `GraphSAINTRandomWalkSampler` can take explicit walk starts through `metadata["seed"]`, and `GraphSAINTEdgeSampler` can force specific edge ids through `metadata["edge_id"]` or `metadata["edge_ids"]`. When `GraphSAINTNodeSampler` or `GraphSAINTRandomWalkSampler` receives multiple explicit seed nodes, VGL expands them into multiple `SampleRecord`s that share one sampled subgraph, so `Loader` can still build a `NodeBatch` with one `seed_index` entry per supervised node. `ShaDowKHopSampler` now follows the same pattern for explicit multi-seed inputs.
For node-centered induced-subgraph sampling, `GraphSAINTNodeSampler`, `GraphSAINTEdgeSampler`, and `ShaDowKHopSampler` also expose `seed_positions`, which maps each `seed_ids` entry back to its local position inside `sampled_node_ids`.

Across these advanced samplers, metadata now also records effective subgraph sizes through `sampled_num_nodes` and `sampled_num_edges`; induced-subgraph samplers additionally expose `subgraph_edge_ids` for the concrete original edges that landed in the sampled subgraph, along with a compatible `sampled_edge_ids` alias when the sampled edges correspond directly to original graph edges.
For partition-style mini-batching, `ClusterData` samples also keep `cluster_id` plus a partition-oriented `partition_id` alias, and surface `num_parts` both at the top level and inside `sampling_config`.

For advanced systems work, the new foundation layers sit underneath the same surface API:

- `vgl.sparse` for cached COO/CSR/CSC adjacency layouts, multi-value edge payloads, transpose/reduction helpers, and sparse operators such as payload-aware `spmm(...)`, `sddmm(...)`, and `edge_softmax(...)`
- `vgl.storage` for feature / graph stores, mmap-backed feature tensors, and `Graph.from_storage(...)` with retained feature-source context
- `vgl.ops` for reusable graph transforms, edge queries, adjacency and Laplacian sparse queries, `to_simple(...)`, `reverse(...)`, frontier subgraphs, relation-local `to_block(...)`, multi-relation `to_hetero_block(...)`, line graphs, random walks, metapath random walks, metapath reachability, homogeneous/heterogeneous relation-local subgraph extraction, relation-local k-hop expansion, and compaction
- `vgl.data` for dataset manifests, cache helpers, built-in datasets, and manifest-backed homo/hetero/temporal on-disk datasets with lazy per-item payloads and split views
- `vgl.distributed` for partition metadata, local shard loading, typed node routing, relation-scoped edge routing, edge feature fetches, owned-local plus boundary/incident partition queries, stitched homogeneous node/link/temporal sampling, stitched typed heterogeneous temporal sampling, plus non-temporal heterogeneous node/link sampling across shard boundaries, sampling coordination contracts, and routed plan feature sources across homogeneous, temporal homogeneous, single-node-type multi-relation, and multi-node-type heterogeneous graphs

Use `graph.num_nodes(...)` / `graph.num_edges(...)` or the `number_of_*` aliases for cardinality, `graph.all_edges(...)` for ordered full-edge enumeration, `graph.formats()` / `graph.create_formats_()` for graph sparse-format status and eager creation, `graph.adj(...)` for DGL-style weighted adjacency sparse views, `graph.laplacian(...)` for square-relation Laplacian sparse views, `graph.adj_external(...)` for torch / SciPy sparse export, `graph.adj_tensors(...)` for raw COO/CSR/CSC adjacency export, `graph.inc(...)` for incidence-matrix sparse views, `graph.in_subgraph(...)` / `graph.out_subgraph(...)` for frontier-preserving filters, `graph.find_edges(...)`, `graph.edge_ids(...)`, and `graph.has_edges_between(...)` for edge lookups, `graph.in_edges(...)`, `graph.out_edges(...)`, `graph.predecessors(...)`, and `graph.successors(...)` for one-hop adjacency, `graph.in_degrees(...)` / `graph.out_degrees(...)` for degree inspection, `graph.to_simple(...)` for parallel-edge deduplication, and `graph.reverse(...)` for reversed structure. Count helpers return `int`; degree queries return `int` for one node and tensors for many or all nodes; `graph.formats()` reports `created` versus `not created` sparse layouts; `graph.create_formats_()` eagerly materializes the allowed set and returns `None`; `graph.adj(..., eweight_name="weight")` overlays edge features as sparse values; `graph.laplacian(normalization="rw")` or `"sym"` adds normalized Laplacian sparse views on square relations while preserving declared storage-backed node space; `graph.adj_external()` returns `torch.sparse_coo_tensor` by default, `graph.adj_external(torch_fmt="csr")` or `"csc"` returns native compressed torch layouts, and `graph.adj_external(scipy_fmt="coo")` exports SciPy matrices; `graph.adj_tensors("coo")` returns `(src, dst)` while `"csr"` / `"csc"` return compressed pointers, coordinates, and aligned public edge ids; `graph.inc(...)` returns a `SparseTensor`; `graph.to_simple(count_attr="count")` preserves one visible edge per endpoint pair, stores multiplicity counts, and drops `e_id` on the simplified relation because original edge identity is no longer one-to-one.

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

When you need a graph-op-level heterogeneous message-flow layer, use `graph.to_hetero_block({...})` or `vgl.ops.to_hetero_block(...)`. This builds one multi-relation bipartite layer from destination frontiers keyed by node type and returns a `HeteroBlock` object with per-type `src_n_id` / `dst_n_id` plus relation-scoped edge views. The sampler `output_blocks=True` paths can now emit either relation-local `Block` or multi-relation `HeteroBlock` layers, depending on whether one relation-local schema is sufficient for that sampled workload. Multi-relation layers can now also round-trip through DGL with `HeteroBlock.to_dgl()`, `HeteroBlock.from_dgl(...)`, `vgl.compat.hetero_block_to_dgl(...)`, or `vgl.compat.hetero_block_from_dgl(...)`.

When you want DGL-style message-flow blocks for a node workload, switch the loader to `NodeNeighborSampler([15, 10], output_blocks=True)`. The resulting `NodeBatch` still exposes `batch.graph` and `batch.seed_index`, and now also exposes `batch.blocks[0]`, `batch.blocks[1]`, ... ordered from the outer frontier back to the seed frontier. Homogeneous paths emit `Block`. On heterogeneous graphs, one-inbound-relation workloads keep relation-local `Block`, while zero- or multi-inbound-relation workloads now emit `HeteroBlock`, covering both local full-graph sampling and stitched shard-local sampling when the loader uses a coordinator-backed feature source. Each emitted relation-local `Block` can still round-trip through DGL with `batch.blocks[0].to_dgl()` or `Block.from_dgl(dgl_block)` when the external block is single-relation, while each emitted `HeteroBlock` can now do the same for multi-relation DGL blocks through `batch.blocks[0].to_dgl()`, `HeteroBlock.from_dgl(dgl_block)`, or the `vgl.compat.hetero_block_*` helpers.

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

When migrating existing link datasets, `HardNegativeLinkSampler(..., hard_negative_dst_metadata_key="hard_pool")` can read hard-negative destinations from a custom metadata key instead of requiring `hard_negative_dst`, and `CandidateLinkSampler(..., candidate_dst_metadata_key="candidate_pool")` does the same for ranking candidates in full-graph evaluation flows. Expanded negatives and candidates also keep `metadata["query_id"]` as the sampler-side query id when the record field is unset, and query grouping across sampled or direct full-graph batches falls back through `query_id`, `metadata["query_id"]`, `sample_id`, and `metadata["sample_id"]`. `CandidateLinkSampler` skips already-negative seed records by default, which lets ranking loaders consume split datasets that already include pre-sampled negatives, while `CandidateLinkSampler(..., skip_negative_seed_records=False)` restores strict positive-only validation when you want accidental negatives to fail fast. `UniformNegativeLinkSampler(..., skip_negative_seed_records=True)` and `HardNegativeLinkSampler(..., skip_negative_seed_records=True)` provide the same opt-in behavior for mixed-label training datasets. Resolved `query_id` values are normalized back onto emitted record metadata, metadata-backed `sample_id` values are also normalized back onto emitted positive and generated negative records, including suffixed negative sample ids, resolved seed-edge exclusion is mirrored onto positive-record metadata as `exclude_seed_edges=True`, and resolved relation types are mirrored back onto emitted metadata as `edge_type` / `reverse_edge_type`. `RandomLinkSplit` keeps stable `sample_id` values for every split record, preserves `query_id == sample_id` for positive supervision edges, reuses the owning positive `query_id` and source node for generated negatives so direct full-graph batches still group as query-local ranking queries, avoids duplicate negative destinations when that query still has enough unique negatives available, emits splits with generated negatives in query-grouped order so those ranking groups stay contiguous in direct batches, and marks positive split records with `exclude_seed_edge` / `metadata["exclude_seed_edges"]` so later batching or neighbor sampling can drop the supervision edge automatically. Resolved candidate / hard-negative pools are normalized back onto both emitted record fields and the standard metadata keys.

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

When `graph` has multiple relations or node types, pass `edge_type=` on each `TemporalEventRecord`, for example `TemporalEventRecord(..., edge_type=('author', 'writes', 'paper'))`. In sampled loaders, `TemporalNeighborSampler` keeps strict-history extraction relation-local and `TemporalEventBatch` exposes `edge_type`, `edge_types`, `edge_type_index`, `src_node_type`, and `dst_node_type` for typed temporal models. When the source graph is shard-local and sampled through `LocalSamplingCoordinator`, that relation-local history can also stitch earlier cross-partition events into one typed temporal subgraph.

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
format_status = graph.formats()
graph.create_formats_()
weighted_adjacency = graph.adj(eweight_name="weight", layout="csr")
laplacian = graph.laplacian(normalization="sym", layout="csr")
external_adjacency = graph.adj_external(torch_fmt="csr")
external_adjacency_scipy = graph.adj_external(scipy_fmt="coo")
src, dst = graph.adj_tensors("coo")
crow_indices, col_indices, edge_ids = graph.adj_tensors("csr")
```

When a later `SamplingPlan` includes feature-fetch stages, `PlanExecutor.execute(..., graph=graph)` and `Loader(..., sampler=...)` will reuse `graph.feature_store` automatically unless you pass an explicit `feature_store=` override. For sampled node, link, and temporal workloads, `NodeNeighborSampler(node_feature_names=..., edge_feature_names=...)`, `LinkNeighborSampler(...)`, and `TemporalNeighborSampler(...)` can append those fetch stages opt-in and materialize the fetched slices back into each sampled subgraph. Use dictionaries keyed by node type / edge type when the sampled graph is heterogeneous.

For node sampling specifically, each dataset item may carry `metadata["seed"]` as one integer or a rank-1 seed collection. `NodeNeighborSampler` will sample one union subgraph for that item and materialize one flat `seed_index` entry per requested seed, so the downstream `NodeBatch` contract stays unchanged. With `output_blocks=True`, the same node path now emits `NodeBatch.blocks` for homogeneous graphs and for heterogeneous graphs: one-inbound-relation workloads keep `Block`, while zero- or multi-inbound-relation workloads emit `HeteroBlock`.

`LinkNeighborSampler(..., output_blocks=True)` keeps the existing `LinkPredictionBatch.graph`, `src_index`, `dst_index`, and `labels` contract and additionally materializes `LinkPredictionBatch.blocks` in outer-to-inner order. This now covers local and stitched homogeneous link sampling plus heterogeneous link sampling through a coordinator-backed feature source. Those blocks are derived from the sampled message-passing graph, so positive seed edges removed by `exclude_seed_edge` / `exclude_seed_edges` stay absent from the block layers. On heterogeneous graphs, single-relation supervision keeps `Block`, while mixed-edge-type supervision emits `HeteroBlock`, and those multi-relation block layers now have the same explicit DGL import/export path through `HeteroBlock.to_dgl()` and `HeteroBlock.from_dgl(...)`.

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
boundary_edge_index = shard.boundary_edge_index(edge_type=("node", "follows", "node"))
partition_node_ids = coordinator.partition_node_ids(0, node_type="paper")
partition_edge_ids = coordinator.partition_edge_ids(0, edge_type=("author", "writes", "paper"))
partition_boundary_edge_ids = coordinator.partition_boundary_edge_ids(0, edge_type=("author", "writes", "paper"))
edge_weights = coordinator.fetch_edge_features(
    ("edge", ("author", "writes", "paper"), "weight"),
    partition_edge_ids,
).values
incident_edge_index = coordinator.fetch_partition_incident_edge_index(0, edge_type=("node", "follows", "node"))
partition_adjacency = coordinator.fetch_partition_adjacency(0, edge_type=("node", "follows", "node"), layout="csr")
```

Plan-backed feature fetch stages can also use the same routed source directly through `PlanExecutor.execute(..., feature_store=coordinator)` or `Loader(..., feature_store=coordinator)` when you want executor-driven feature access instead of direct store access. Those explicit arguments remain the highest-priority override; otherwise, storage-backed graphs can supply the same context through their retained `graph.feature_store`. When the source graph is a shard-local `shard.graph`, `NodeNeighborSampler` and `LinkNeighborSampler` can now use coordinator incident-edge queries to stitch remote frontier nodes and edges into the sampled subgraph for homogeneous workloads and for non-temporal heterogeneous node/link workloads, while keeping node and edge tensors aligned through global `n_id` / `e_id`. `TemporalNeighborSampler` can do the same for earlier cross-partition history in homogeneous workloads and can stitch earlier cross-partition relation-local history for typed heterogeneous temporal workloads.

```python
loader = DataLoader(
    dataset=ListDataset([(shard.graph, {"seed": 1, "sample_id": "part0"})]),
    sampler=NodeNeighborSampler(num_neighbors=[-1, -1]),
    batch_size=1,
    feature_store=coordinator,
)
batch = next(iter(loader))
# batch.graph.n_id now contains both local and remote frontier nodes
```

These advanced paths are still designed to terminate in the same public training contracts: `Graph`, batch objects from `Loader`, and `Trainer.fit/evaluate/test`.
