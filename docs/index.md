---
hide:
  - navigation
  - toc
---

<div class="vgl-hero" markdown>

# VGL

### 基于 PyTorch 的统一图学习框架

[![Python](https://img.shields.io/badge/python-%E2%89%A53.10-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A52.4-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Version](https://img.shields.io/pypi/v/sky-vgl?style=for-the-badge)](https://pypi.org/project/sky-vgl/)
[![PyPI](https://img.shields.io/badge/PyPI-sky--vgl-3775A9?style=for-the-badge&logo=pypi&logoColor=white)](https://pypi.org/project/sky-vgl/)

</div>

<div class="grid" markdown>

<div class="card" markdown>

### 🇨🇳 站点介绍

VGL 提供统一 Graph 抽象、60+ 图卷积、完整训练 pipeline 与多种采样策略，覆盖同构/异构/时序图。

[:octicons-arrow-right-24: 用户指南](getting-started/index.md)

</div>

<div class="card" markdown>

### 🇬🇧 English Summary

VGL is a PyTorch-first graph framework that bundles Graph, Task, Trainer, sparse tooling, and dataset adapters so you can address homogeneous, heterogeneous, and temporal workloads from a single API.

[:octicons-arrow-right-24: Quick Start](getting-started/index.md)

</div>

</div>

---

**VGL** (Versatile Graph Learning) 提供**一个统一的 `Graph` 抽象**，同时覆盖同构图、异构图和时序图，并内置从数据加载到模型训练的完整 pipeline。

---

<div class="grid" markdown>

<div class="card" markdown>

### :material-rocket-launch: 快速开始

5 分钟完成安装并运行第一个图学习任务。

[:octicons-arrow-right-24: 开始体验](getting-started/index.md)

</div>

<div class="card" markdown>

### :material-lightbulb: 核心概念

了解 Graph、Task、Trainer 等核心抽象。

[:octicons-arrow-right-24: 核心概念](core-concepts.md)

</div>

<div class="card" markdown>

### :material-book-open-variant: 用户指南

节点分类、图分类、链接预测等完整教程。

[:octicons-arrow-right-24: 用户指南](guide/index.md)

</div>

<div class="card" markdown>

### :material-api: API 参考

所有公共模块和类的完整 API 文档。

[:octicons-arrow-right-24: API 参考](api/index.md)

</div>

<div class="card" markdown>

### :material-code-braces: 示例

端到端代码示例和卷积层速查表。

[:octicons-arrow-right-24: 示例](examples/index.md)

</div>

<div class="card" markdown>

### :material-sitemap: 架构概览

包结构、分层设计和模块职责。

[:octicons-arrow-right-24: 架构概览](architecture.md)

</div>

</div>

---

## 运行与发布入口

如果你不是第一次打开文档，下面这三个页面最适合直接定位运行、发布和支持边界：

- [Support Matrix](support-matrix.md) — 查看当前 CI / release 实际验证过的 Python、PyTorch、extras、interop 组合
- [发布指南](releasing.md) — 查看 release smoke、artifact interop、失败分流和 issue intake 入口
- [FAQ](faq.md) — 查看安装、Graph、采样、训练和迁移的高频问题

---

## 核心特性

| 特性 | 说明 |
|------|------|
| **统一 Graph 对象** | 一个数据结构同时支持同构图、异构图和时序图，内置 schema 校验、轻量视图和自动 batching |
| **60+ 卷积层** | GCN、GAT、SAGE、GIN、Transformer 等主流图卷积，以及 RGCN、HGT、HAN 等异构算子 |
| **完整训练 pipeline** | Trainer + Task + Metric 三件套，支持早停、checkpoint、多种 logger |
| **灵活数据加载** | NeighborSampler、GraphSAINT、ClusterGCN、RandomWalk 等多种采样策略 |
| **丰富任务类型** | 节点分类、图分类、链接预测、时序事件预测 |
| **高级基础设施** | 稀疏张量、特征存储、mmap 后端、分布式分区 |
| **框架互操作** | DGL、PyG、NetworkX、SciPy 双向转换 |
| **内置数据集** | Cora、Citeseer、PubMed、MUTAG、PROTEINS 等经典数据集 |

---

## VGL vs PyG vs DGL

下面这张表不是性能排行榜，而是基于当前公开 API、文档和官方资料的定位说明。

| 框架 | 更适合什么场景 | 当前公开强项 | 当前取舍 |
|------|----------------|--------------|----------|
| **VGL** | 想用一套 API 同时覆盖同构图、异构图、时序图，并直接接上 Task / Trainer / Metric / sampling / interop | 当前公开表面强调统一 `Graph` 抽象、内建训练 pipeline、采样层、稀疏与存储能力，以及 DGL / PyG 互操作 | 外部生态规模和分布式运行时成熟度目前不如 PyG / DGL；更适合重视统一抽象和内建 workflow 的团队 |
| **PyG** | 已经在 PyTorch 生态里，希望优先获得成熟的 `Data` / `HeteroData`、heterogeneous modeling、mini-batch loaders 与可选编译扩展 | 官方文档把 `to_hetero()`、`HeteroConv`、`NeighborLoader`、`HGTLoader` 作为异构图与大图训练的重要入口；官方 README 也把 `pyg-lib` 的 heterogeneous operators 和 graph sampling routines 作为扩展重点 | 训练 loop、任务抽象、日志/检查点策略更偏组合式，项目级 pipeline 往往需要自行组织 |
| **DGL** | 把 stage 化 dataloading、GraphBolt 或多机分布式图训练当成核心要求 | 官方文档将 `dgl.graphbolt` 定义为按数据管线阶段组织的 GNN dataloading framework，也提供 `dgl.distributed` 与 `DistGraph` 支持分区图、分布式采样和集群训练 | 对于更偏 PyTorch-native 对象和 PyG 风格模型/loader 的团队，DGL 的运行时与 API 习惯会更像一套独立体系 |

### VGL 当前更强的点

- **统一图抽象优先。** 当前文档和包结构把同构图、异构图、时序图都放在一个 `Graph` 公共入口下，而不是要求用户在不同图类型之间切换不同顶层对象。
- **训练栈内建。** `Trainer`、`Task`、`Metric`、checkpoint、logger、sampling 都在同一项目表面上，适合想要少拼装一些基础设施的项目。
- **互操作是公共能力，不是隐藏脚本。** 当前迁移指南和兼容层把 DGL / PyG 双向适配作为显式入口暴露出来。

### VGL 当前更薄的点

- **外部生态还不如 PyG / DGL 深。** 如果你的第一优先级是社区现成 recipe、第三方教程或更广泛的扩展生态，PyG 和 DGL 仍然更成熟。
- **分布式运行时不以 DGL 为目标对齐。** DGL 官方文档已经公开 `GraphBolt` 和 `DistGraph` 这类分布式与 stage-based runtime 表面；VGL 当前更偏本地优先、统一抽象优先。
- **如果你只想要 PyTorch-native heterogeneous mini-batching，PyG 往往更直接。** 这是因为 PyG 已经把 `HeteroData`、`NeighborLoader` 和 `to_hetero()` 路径文档化并长期围绕它演进。

### 参考资料

- [PyG README](https://github.com/pyg-team/pytorch_geometric)
- [PyG heterogeneous graph docs](https://pytorch-geometric.readthedocs.io/en/stable/notes/heterogeneous.html)
- [DGL README](https://github.com/dmlc/dgl)
- [DGL GraphBolt docs](https://www.dgl.ai/dgl_docs/api/python/dgl.graphbolt.html)
- [DGL distributed docs](https://www.dgl.ai/dgl_docs/api/python/dgl.distributed.html)
- [VGL 架构概览](architecture.md)
- [VGL 迁移指南](migration-guide.md)

---

## 包架构

| 模块 | 功能 |
|------|------|
| `vgl.graph` | Graph、GraphBatch、GraphView、GraphSchema、Block、HeteroBlock |
| `vgl.nn` | 60+ 图卷积层、Graph Transformer、池化函数 |
| `vgl.dataloading` | DataLoader、采样器、数据集、样本记录 |
| `vgl.engine` | Trainer、Callback、Logger、Checkpoint |
| `vgl.tasks` | 节点分类、图分类、链接预测、时序预测任务 |
| `vgl.metrics` | Accuracy、MRR、HitsAtK 等评估指标 |
| `vgl.transforms` | Compose、NormalizeFeatures、RandomNodeSplit 等变换 |
| `vgl.ops` | 图结构操作：子图、度数、邻接、拉普拉斯、随机游走 |
| `vgl.sparse` | SparseTensor、COO/CSR/CSC、spmm、sddmm |
| `vgl.storage` | FeatureStore、GraphStore、MmapTensorStore |
| `vgl.data` | 数据集清单、缓存、内置数据集、磁盘数据集 |
| `vgl.distributed` | 分区元数据、本地分片、采样协调器 |
| `vgl.compat` | DGL / PyG / NetworkX / CSV 互操作适配器 |

---

## 支持的图类型

| 图类型 | 构建方式 | 典型任务 |
|--------|----------|----------|
| 同构图 | `Graph.homo(edge_index=..., x=..., y=...)` | 节点分类、链接预测 |
| 异构图 | `Graph.hetero(nodes={...}, edges={...})` | 异构节点分类、异构链接预测 |
| 时序图 | `Graph.temporal(nodes=..., edges=..., time_attr=...)` | 时序事件预测 |

---

## Support Matrix

Track Python, PyTorch, and extras coverage so your environment matches the CI-tested surface.

[Verified combinations →](support-matrix.md)

Need installation and troubleshooting context before you pick an environment?

- [Installation guide →](getting-started/installation.md)
- [FAQ →](faq.md)
- [Release verification and triage →](releasing.md)

---

<div style="text-align: center; color: var(--md-default-fg-color--light); margin-top: 2rem;">
  <small>VGL 采用 MIT 许可证发布。</small>
</div>
