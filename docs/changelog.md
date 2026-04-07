# 更新日志

## Unreleased

### Changed

- Release and public-surface policy now lives in a shared machine-readable contract used by scans, tests, and smoke tooling.
- Documentation entry pages now include a verified support matrix and bilingual summary blocks for the highest-traffic entry points.

### Fixed

- Packaging metadata now points PyPI users to the published docs site instead of the repository README.
- Example smoke coverage now tracks the full public example catalog rather than a hand-picked subset.

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
