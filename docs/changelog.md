# 更新日志

## Unreleased

### API

- 暂无条目。

### Performance

- 暂无条目。

### Interop

- 暂无条目。

### Migration

- 暂无条目。

### Docs

- 补齐 `vgl.distributed` 中 `SamplingCoordinator` / `StoreBackedSamplingCoordinator` / `ShardRoute` / `PartitionManifest` / `PartitionShard` / `load_partition_manifest` / `save_partition_manifest` / 各 `Distributed*` 与 `Partitioned*Store` / `Local*Adapter` / `load_partitioned_stores` 的 API 参考。
- 补齐 `vgl.sparse` 的 `SparseLayout`、`to_coo/csr/csc`、`from_edge_index`、`from/to_torch_sparse`、`degree`、`sum`、`transpose`、`select_rows/cols` API 页条目。
- 补齐 `vgl.compat` 中 `to_networkx/from_networkx` 与 `block_*/hetero_block_*` 的 API 参考块。
- 补齐 `vgl.ops.to_simple` 的去重/归并语义说明。
- 修正文档里的 distributed store bootstrap 示例,明确 `load_partitioned_stores(...)` 返回 `(manifest, feature_store, graph_store)`,并补充 `StoreBackedSamplingCoordinator.from_partition_dir(...)` 作为最短引导路径。
- 修正文档里的 interop CSV helper 名称为 `Graph.from_edge_list_csv(...)` 与 `Graph.from_csv_tables(...)`。
- 修正 `docs/api/data.md` 对 `OnDiskGraphDataset` 的逐图 payload 布局、按需逐条 `torch.load(...)`、manifest-backed split view 与旧版 `graphs.pt` 兼容说明。
- 在 `docs/api/engine.md` 增加 `restore_checkpoint` 的校验契约与 `Evaluator` 条目。
- 新增用户指南小节:异构多关系采样块(`guide/hetero.md`)、跨分区 `StoreBackedSamplingCoordinator` 采样(`guide/sampling.md`)、NetworkX 与 torch.sparse 互操作(`guide/graph.md`)。
- 在 `architecture.md` 与 `core-concepts.md` 登记分布式存储与采样协调、`HeteroBlock` / `PartitionManifest` 概念。
- 在 `migration-guide.md` 明确 `vgl.core` / `vgl.train` 弃用告警与首选导入路径。
- 新增示例页 `examples/distributed-partition.md` 与 `examples/interop.md`,并纳入 mkdocs 导航。

## v0.1.9

### API

- Release and public-surface policy now lives in a shared machine-readable contract used by scans, tests, and smoke tooling.

### Performance

- Documentation entry pages now include a verified support matrix and bilingual summary blocks for the highest-traffic entry points.
- Benchmark hotpath artifacts are now documented as versioned JSON contracts with timestamp and runner metadata.
- Release smoke now supports an explicit import-time budget and reports import timing evidence for installed artifacts.

### Interop

- Packaging metadata now points PyPI users to the published docs site instead of the repository README.
- Example smoke coverage now tracks the full public example catalog rather than a hand-picked subset.

### Migration

- Legacy `vgl.core` and `vgl.train` namespaces now emit one-time compatibility warnings while preserving the existing import surface; `vgl.data` remains the public dataset namespace while legacy loader/sampler-style `vgl.data.*` module paths stay available for compatibility.

## v0.1.5

### Added

- 60+ 种图卷积层
- Graph Transformer 支持（Graphormer、GPS、NAGphormer、SGFormer）
- 时序图模块（TGN、TGAT）
- 完整的训练 pipeline（Trainer + Task + Metric）
- 多种采样策略（Neighbor、GraphSAINT、ClusterGCN、RandomWalk）
- 链接预测和时序事件预测任务
- Storage 后端（FeatureStore、MmapTensorStore）
- 分布式分区和本地采样协调
- DGL / PyG / NetworkX 互操作
- 内置数据集（Cora、Citeseer、PubMed、MUTAG、PROTEINS 等）
- Block / HeteroBlock 消息流容器
- 多种 Logger（Console、JSONL、CSV、TensorBoard）
- 高级优化器（SAM、ASAM、Lookahead、EMA、SWA）
- 鲁棒训练任务（Bootstrap、Flooding、GCE、SCE 等）

---

*更多版本历史请参见 [GitHub Releases](https://github.com/skygazer42/sky-vgl/releases)。*
