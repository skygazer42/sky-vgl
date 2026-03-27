# Core Concepts

## Graph

`Graph` is the canonical graph object in this package. Homogeneous graphs are a special case of the same abstraction used for heterogeneous and temporal graphs.

Homogeneous graphs can carry edge-level tensors through `Graph.homo(edge_data={...})`. These tensors are exposed through `graph.edata` and are what edge-aware operators consume.

`Graph` also has lightweight interoperability entry points for common external graph shapes. Homogeneous graphs can round-trip through PyG, NetworkX, in-memory edge lists, CSV edge-list files, and paired node/edge CSV tables via `Graph.from_*` / `graph.to_*` helpers under `vgl.compat`, while DGL compatibility continues to cover homogeneous, heterogeneous, temporal, and block-oriented flows.

`Graph` also has a storage-backed construction path. `Graph.from_storage(schema=..., feature_store=..., graph_store=...)` builds one graph view from feature / graph stores without changing the public graph contract. Structural data such as `edge_index` is available immediately, while node and edge features resolve lazily on first access and then stay cached on the store. Feature tensors can live in lightweight in-memory stores or in `MmapTensorStore` files backed by raw tensor buffers plus metadata sidecars for large feature tables. Storage-backed graphs also retain the originating `feature_store` and `graph_store` on the graph object, so later plan-backed execution can reuse those sources when no explicit override is supplied. That retained graph-store metadata is what lets featureless storage-backed graphs preserve declared node counts for adjacency views and other count-driven structure ops without inventing placeholder node features.

## SparseTensor and Adjacency Caches

`vgl.sparse` is the low-level sparse execution layer. It provides `SparseTensor`, COO/CSR/CSC conversion helpers, bidirectional interop with native `torch.sparse_*` COO/CSR/CSC tensors, transpose, additive reductions, row/column structural selection, sparse edge payloads shaped `(nnz, ...)`, and sparse ops such as sparse-dense matmul that preserves trailing sparse payload dimensions, sampled dense-dense matmul via `sddmm(...)`, and edge-wise normalization via `edge_softmax(...)`.

`Graph.adjacency(layout=...)` is the main bridge back into user code for adjacency-oriented structure. It builds sparse adjacency views through `vgl.sparse`, including CSC layouts for column-oriented traversals, and caches them on each edge store so repeated structural operations do not need to rebuild the same layout. `Graph.formats()` reports which graph sparse formats are currently created versus merely allowed, while `Graph.create_formats_()` eagerly materializes the allowed set. `Graph.adj(...)` adds a DGL-style weighted adjacency entry point on top of the same `SparseTensor` substrate, with optional `eweight_name` selection and layout control. `Graph.adj_external(...)` adds an external sparse export path that emits `torch.sparse_coo_tensor` by default, can emit native torch COO / CSR / CSC layouts through `torch_fmt=`, and still exports SciPy `coo` / `csr` matrices through `scipy_fmt=` for downstream interop. `Graph.adj_tensors(...)` complements that with raw structural tensor export: `("coo")` returns `(src, dst)`, `("csr")` returns `(crow_indices, col_indices, edge_ids)`, and `("csc")` returns `(ccol_indices, row_indices, edge_ids)`. `Graph.inc(...)` adds incidence-matrix views whose columns are edge-aligned instead of destination-aligned.

`vgl.ops` sits one layer above that and now supports both homogeneous structure rewrites and relation-local heterogeneous `node_subgraph(...)`, `edge_subgraph(...)`, `khop_nodes(...)`, `khop_subgraph(...)`, and `compact_nodes(...)` flows when an `edge_type` is selected. It also exposes `in_subgraph(...)` and `out_subgraph(...)` as frontier-preserving edge filters that keep the original node space intact while extracting inbound or outbound edges and surfacing selected `e_id`, graph cardinality helpers through `num_nodes(...)`, `number_of_nodes(...)`, `num_edges(...)`, and `number_of_edges(...)`, full-edge enumeration through `all_edges(...)`, graph format-state helpers through `formats(...)` and `create_formats_(...)`, weighted adjacency sparse views through `adj(...)`, external adjacency export through `adj_external(...)`, raw adjacency export through `adj_tensors(...)`, incidence sparse views through `inc(...)`, edge-query helpers through `find_edges(...)`, `edge_ids(...)`, `has_edges_between(...)`, `in_edges(...)`, `out_edges(...)`, `predecessors(...)`, `successors(...)`, `in_degrees(...)`, and `out_degrees(...)`, non-mutating `reverse(...)`, relation-local `to_block(...)` plus multi-relation `to_hetero_block(...)` for message-flow rewrites, `line_graph(...)` for edge-centric homogeneous topology transforms, `random_walk(...)` for repeated relation-local path sampling, and `metapath_random_walk(...)` for typed path sampling, and `metapath_reachable_graph(...)` for composing heterogeneous metapaths into one derived reachable relation. When a derived graph already carries `edata["e_id"]`, the query helpers resolve against that stable public edge-id space instead of falling back to local edge positions. `formats()` reports `created` versus `not created` sparse formats in canonical `coo/csr/csc` order, `create_formats_()` eagerly builds every allowed format, and `all_edges(order="eid")`, `adj(...)`, `adj_external(...)`, `adj_tensors(...)`, and `inc(...)` all align to that same public edge-id ordering model. `all_edges(order="srcdst")` sorts lexicographically by endpoints. `adj(..., eweight_name="weight")` overlays edge features as sparse values while preserving public edge order within visible COO rows / cols and compressed CSR / CSC buckets. `adj_external(...)` exports the same visible structure into either `torch.sparse_coo_tensor` or SciPy `coo` / `csr` matrices, and `transpose=True` swaps source/destination orientation before export. `adj_tensors("coo")` returns reordered endpoints directly, while `adj_tensors("csr")` and `adj_tensors("csc")` expose compressed pointers plus aligned public edge ids and preserve public edge order within each row or column bucket. `inc(typestr="in")` and `inc(typestr="out")` place `+1` at destination or source rows respectively, while `inc(typestr="both")` uses `-1/+1` source/destination entries and leaves self-loop columns structurally empty, matching DGL’s observed behavior. `in_edges(...)` / `out_edges(...)` can return endpoints, edge ids, or all three together, while `predecessors(...)` / `successors(...)` intentionally preserve duplicates for parallel edges. `in_degrees(...)` / `out_degrees(...)` return Python `int` for scalar queries and tensors for vector or full-node-space queries, and the count helpers, adjacency sparse views, external adjacency exports, adjacency-tensor pointers, and incidence views use retained graph node counts so featureless storage-backed isolated nodes still contribute to total node counts, exported matrix shapes, CSR pointer lengths, row dimensions, and zero-degree outputs. `reverse(...)` keeps edge order stable while swapping endpoints and retains graph-level node-count context, which matters for featureless storage-backed graphs. `to_block(...)` returns a lightweight `Block` container whose wrapped graph carries compacted source/destination frontiers plus `n_id` / `e_id` metadata in stable relation-local order, while `to_hetero_block(...)` returns a `HeteroBlock` container for one multi-relation heterogeneous message-flow layer built from per-type destination frontiers. For bipartite relations, `khop_nodes(...)` consumes seeds keyed by node type and returns per-type node ids so the resulting `khop_subgraph(...)` can preserve the hetero schema, while `metapath_random_walk(...)` returns one node-id trace column per metapath step.

## GraphView

`GraphView` is a lightweight projection over an existing graph, used for operations such as `snapshot()` and `window()`. Views continue to reference graph-level runtime context from the base graph, including retained storage-backed feature and graph sources.

## Block

`Block` is the compact relation-local message-flow container returned by `to_block(...)` or `Graph.to_block(...)`. It wraps one bipartite `graph` plus explicit source/destination node id metadata through `src_n_id`, `dst_n_id`, and edge metadata through `edata["e_id"]`. This gives VGL a first-class bridge to DGL-style block workflows without forcing every sampler to change contracts. `NodeNeighborSampler(..., output_blocks=True)` still uses this container for homogeneous node sampling and for heterogeneous node workloads whose supervised `node_type` has exactly one inbound relation, including the stitched coordinator-backed path. `LinkNeighborSampler(..., output_blocks=True)` does the same for homogeneous link sampling and for heterogeneous link workloads whose sampled records stay on one supervised `edge_type`. DGL interop stays relation-local here: single-relation DGL blocks round-trip through `Block.to_dgl()`, `Block.from_dgl(...)`, `vgl.compat.block_to_dgl(...)`, and `vgl.compat.block_from_dgl(...)`.

For same-type relations, the wrapped graph uses deterministic internal source/destination node stores while `Block` keeps the original `src_type` / `dst_type` visible to callers. For bipartite relations, the wrapped graph keeps the original endpoint node types. `Block.to(...)` and `Block.pin_memory()` mirror the existing graph/batch transfer behavior so block-aware code can participate in the same device pipeline.

`HeteroBlock` is the additive multi-relation counterpart used by `to_hetero_block(...)` or `Graph.to_hetero_block(...)`. It wraps one heterogeneous bipartite layer, keeps `src_n_id` and `dst_n_id` keyed by the original node types, and resolves internal store renaming through `src_store_types` and `dst_store_types` so same-type relations can still participate safely in one layer. Samplers now use it directly when one relation-local `Block` is no longer expressive enough: heterogeneous node block output with zero or multiple inbound relations materializes `HeteroBlock` layers, and heterogeneous link block output does the same when one sampled item mixes multiple `edge_type` values. As a result, `NodeBatch.blocks` and `LinkPredictionBatch.blocks` may now contain either `Block` or `HeteroBlock`. Unlike `Block`, this container now has its own dedicated DGL bridge through `HeteroBlock.to_dgl()`, `HeteroBlock.from_dgl(...)`, `vgl.compat.hetero_block_to_dgl(...)`, and `vgl.compat.hetero_block_from_dgl(...)`, so multi-relation sampled frontiers can round-trip through DGL while preserving canonical edge types, `n_id` / `e_id`, and temporal `time_attr` metadata.

## GraphBatch

`GraphBatch` groups multiple graphs into one training input and tracks node-to-graph membership.

For homogeneous graph batches, membership lives in `graph_index` and `graph_ptr`. For heterogeneous or temporal graph batches, membership is tracked per node type via `graph_index_by_type` and `graph_ptr_by_type`.

For graph classification it also carries:

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

This is what makes many-small-graph and sampled-subgraph inputs converge on the same batch contract. For homogeneous node sampling, `SampleRecord` can now also carry per-hop `blocks` derived from the sampled subgraph, including stitched homogeneous shard-local samples after coordinator-backed materialization.

## Sampling Plans and Materialization

Neighbor sampling now routes through explicit `SamplingPlan` stages inside `vgl.dataloading`. The public samplers still look like `NodeNeighborSampler`, `LinkNeighborSampler`, and `TemporalNeighborSampler`, but internally they can build plans, execute expansion / feature-fetch stages, and materialize the result back into the same batch contracts. Feature-fetch stages can resolve against an explicit feature source passed into `Loader` or `PlanExecutor`, fall back to a storage-backed graph's retained `feature_store`, or use a coordinator-backed routed source such as `LocalSamplingCoordinator` via `fetch_node_features(...)` / `fetch_edge_features(...)`, so the executor stays agnostic to whether tensors come from one local store or a partitioned runtime. For shard-local homogeneous graphs, `NodeNeighborSampler` and `LinkNeighborSampler` can now use the same coordinator to stitch cross-partition frontier nodes and edges into one sampled subgraph, and homogeneous `TemporalNeighborSampler` can stitch earlier cross-partition history into one sampled temporal subgraph, while keeping node and edge tensors aligned by global `n_id` / `e_id`. For shard-local non-temporal heterogeneous graphs, `NodeNeighborSampler` and `LinkNeighborSampler` can now stitch typed cross-partition frontier structure into one sampled hetero subgraph while keeping per-type node and edge tensors aligned by global `n_id` / `e_id`. For shard-local typed heterogeneous temporal graphs, `TemporalNeighborSampler` can now stitch earlier cross-partition relation-local history into one sampled temporal subgraph while keeping per-type node tensors aligned by global `n_id` and sampled relation edges aligned by global `e_id`. During node, link, and temporal sample materialization, fetched node and edge slices are aligned to each sampled subgraph's `n_id` / `e_id` order and overlaid onto the resulting graph. `NodeNeighborSampler(node_feature_names=..., edge_feature_names=...)`, `LinkNeighborSampler(...)`, and `TemporalNeighborSampler(...)` are the opt-in public shortcuts for appending those fetch stages automatically.

For node sampling, one plan context can now carry one seed or a rank-1 seed collection. Materialization keeps the sampled graph shared per context, then expands the supervision side into the existing flat `NodeBatch.seed_index` contract so models and tasks do not need a separate multi-seed batch type. `NodeNeighborSampler(..., output_blocks=True)` also derives one block per hop from that sampled subgraph, ordered outer-to-inner, so block edges stay restricted to sampled fanout structure. Homogeneous paths still emit `Block`. On heterogeneous graphs, one-inbound-relation workloads keep relation-local `Block`, while zero- or multi-inbound-relation workloads now emit `HeteroBlock` layers for both local full-graph sampling and stitched shard-local sampling through a coordinator-backed feature source.

For link sampling, materialization keeps the existing `LinkPredictionBatch.graph`, `src_index`, `dst_index`, and `labels` contract. `LinkNeighborSampler(..., output_blocks=True)` additionally derives one block per hop from the sampled message-passing graph, ordered outer-to-inner; this now covers local and stitched homogeneous link sampling plus heterogeneous link sampling through a coordinator-backed feature source, and if positive supervision edges are marked for exclusion those edges are removed from the block message-passing view as well. On heterogeneous graphs, single-relation supervision keeps relation-local `Block`, while mixed-edge-type supervision now emits `HeteroBlock`.

This keeps the user-facing API stable while opening a path toward larger-graph runtimes, feature stores, and shard-aware coordination.

## LinkPredictionRecord and LinkPredictionBatch

`LinkPredictionRecord` is the explicit candidate-edge unit for link prediction training. Each record carries:

- a context `graph`
- `src_index`
- `dst_index`
- `label`
- optional `edge_type` / `reverse_edge_type` metadata for heterogeneous or multi-relation supervision
- optional per-hop `blocks` for local homogeneous block-aware link sampling

`LinkPredictionBatch` collates these records into one model input while keeping link supervision explicit through:

- `graph`
- `src_index`
- `dst_index`
- `labels`
- optional `blocks` for local homogeneous block-aware link mini-batches

## TemporalEventRecord and TemporalEventBatch

`TemporalEventRecord` is the explicit candidate-event unit for temporal training. Each record carries:

- a temporal `graph`
- `src_index`
- `dst_index`
- `timestamp`
- `label`
- optional `edge_type` when the temporal graph has multiple relations or node types

`TemporalEventBatch` collates these records into one model input while keeping temporal supervision explicit through:

- `graph`
- `src_index`
- `dst_index`
- `timestamp`
- `labels`
- `edge_types` and `edge_type_index` for per-event typed temporal supervision
- `edge_type`, `src_node_type`, and `dst_node_type` when all records share one relation

`TemporalNeighborSampler` preserves the same contract for sampled batches. On typed heterogeneous temporal graphs it extracts strict-history subgraphs relation-locally, keeps the sampled records' `edge_type`, and aligns feature materialization through per-type node and edge ids before overlaying those tensors back onto the sampled graph. When the source graph is shard-local and routed through `LocalSamplingCoordinator`, that relation-local history can also stitch earlier cross-partition events into one typed temporal subgraph.

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


## Dataset and Distributed Foundations

`vgl.data` now includes `DatasetManifest`, `DatasetSplit`, `DatasetCatalog`, `DataCache`, fixture-backed built-in datasets, and an `OnDiskGraphDataset` format. These pieces let dataset metadata, cache paths, and serialized graph collections share one vocabulary. Newly written `OnDiskGraphDataset` artifacts store one serialized graph payload per file under `graphs/`, deserialize items lazily on access, expose contiguous manifest-backed `split(name)` views, and still keep legacy monolithic `graphs.pt` datasets readable.

`vgl.distributed` builds on that with `PartitionManifest`, deterministic local partition writing, `LocalGraphShard`, local passthrough store adapters, and a `LocalSamplingCoordinator` for shard-local routing, feature gathering, and partition-scoped graph queries. The local partition writer and shard loader now support homogeneous, temporal homogeneous, single-node-type multi-relation, and true multi-node-type heterogeneous graphs while keeping the same local-first manifest/payload workflow. `LocalGraphShard` can map local ids back to global ids per node type, expose relation-scoped global edge ids, preserve cross-partition boundary edges in global-id space, and recover owned-local versus full-incident edge frontiers per relation. `LocalSamplingCoordinator` can route node ids with `node_type`, route edge ids with `edge_type`, surface partition node ids, owned edge ids, boundary edge ids, incident edge ids, fetch node features, fetch edge features through keys such as `('edge', edge_type, 'weight')`, and expose local adjacency plus global edge-index views directly from loaded shards. The current implementation is still intentionally local-first, but shard-local homogeneous `NodeNeighborSampler` and `LinkNeighborSampler` workloads can now stitch cross-partition frontier structure through the coordinator, homogeneous `TemporalNeighborSampler` workloads can now stitch earlier cross-partition history, shard-local non-temporal heterogeneous `NodeNeighborSampler` and `LinkNeighborSampler` workloads can now stitch typed cross-partition frontier structure without changing model code, and shard-local typed heterogeneous `TemporalNeighborSampler` workloads can now stitch earlier cross-partition relation-local history without changing model code.
