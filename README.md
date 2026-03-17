<p align="center">
  <img src="assets/logo.svg" width="420" alt="VGL – Versatile Graph Learning"/>
</p>

<p align="center">
  <b>Unified graph learning framework with a stable core abstraction for homogeneous, heterogeneous, and temporal graphs.</b>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-%E2%89%A53.11-blue?style=flat-square&logo=python&logoColor=white" alt="Python 3.11+"/></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-%E2%89%A52.4-ee4c2c?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch 2.4+"/></a>
  <img src="https://img.shields.io/badge/version-0.1.0-8b5cf6?style=flat-square" alt="Version 0.1.0"/>
  <img src="https://img.shields.io/badge/license-see%20LICENSE-green?style=flat-square" alt="License"/>
</p>

---

**VGL** (Versatile Graph Learning) is a PyTorch-first graph learning library that provides **one canonical `Graph` abstraction** for homogeneous, heterogeneous, and temporal graphs — plus a batteries-included training pipeline from data loading to evaluation.

<p align="center">
  <img src="assets/graph-types.svg" width="780" alt="Supported Graph Types"/>
</p>

---

## Highlights

- **Unified `Graph` object** — a single data structure for homogeneous, heterogeneous, and temporal graphs with schema validation, lightweight views, and batching.
- **50+ GNN convolution layers** — all built on a clean `MessagePassing` interface: `GCNConv`, `GATConv`, `SAGEConv`, `GINConv`, `TransformerConv`, and [many more](#supported-convolution-layers).
- **Graph transformer encoders** — reusable encoder blocks such as `GraphTransformerEncoder`, `GraphormerEncoder`, `GPSLayer`, `NAGphormerEncoder`, and `SGFormerEncoder`.
- **Temporal encoders & memory** — temporal modules such as `TimeEncoder`, `TGATLayer`, `TGATEncoder`, `IdentityTemporalMessage`, and `TGNMemory` plug into event prediction without changing the training loop.
- **Edge-aware operators** — homogeneous graphs can now carry `edge_data`, enabling operators such as `NNConv`, `ECConv`, `GINEConv`, `GMMConv`, `CGConv`, `SplineConv`, `GatedGCNConv`, and `PDNConv`.
- **Point / geometric operators** — homogeneous graphs can also carry node positions via `pos`, enabling operators such as `PointNetConv` and `PointTransformerConv`.
- **End-to-end training** — `Trainer` handles the full loop including `fit()`, `evaluate()`, `test()`, early stopping, best-checkpoint saving, full training-state checkpoint/resume, epoch history tracking, gradient accumulation, gradient clipping, adaptive gradient clipping, gradient centralization, layer-wise learning-rate strategies, classification label smoothing, focal loss, `LDAM`, logit adjustment, balanced softmax, class weighting / `pos_weight`, `R-Drop` regularization, sharpness-aware optimization with `SAM`, `ASAM`, and `GSAM`, epoch-wise or step-wise LR scheduling including built-in warmup/cosine support and schedulers such as `OneCycleLR`, mixed precision, and training callbacks such as gradual unfreezing, deferred reweighting (`DRW`), EMA, SWA, and Lookahead.
- **Multiple graph tasks** — node classification, graph classification, link prediction, and temporal event prediction out of the box.
- **PyG & DGL compatibility** — seamless conversion with `from_pyg()` / `to_pyg()` and `from_dgl()` / `to_dgl()` adapters.
- **Clean, modular design** — domain-oriented package layout that separates concerns and stays easy to extend.

---

## Architecture

<p align="center">
  <img src="assets/architecture.svg" width="780" alt="VGL Architecture Overview"/>
</p>

| Package | Description |
|:--|:--|
| `vgl.graph` | `Graph`, `GraphBatch`, `GraphSchema`, `GraphView`, stores |
| `vgl.nn` | `MessagePassing`, 50+ convolution layers, graph/temporal encoders, `HeteroConv`, readout, `GroupRevRes` |
| `vgl.tasks` | `NodeClassificationTask`, `GraphClassificationTask`, `LinkPredictionTask`, `TemporalEventPredictionTask` |
| `vgl.engine` | `Trainer`, callbacks, checkpoints, `TrainingHistory`, evaluator, training strategies |
| `vgl.metrics` | `Accuracy`, `Metric` base, `build_metric` |
| `vgl.dataloading` | `DataLoader`, `ListDataset`, samplers, sample records |
| `vgl.transforms` | Graph transforms (identity, extensible) |
| `vgl.compat` | PyG and DGL bidirectional converters |

> Legacy imports (`vgl.core`, `vgl.data`, `vgl.train`) remain as compatibility layers but new code should use the layout above.

---

## Quick Tour

<p align="center">
  <img src="assets/pipeline.svg" width="780" alt="VGL Training Pipeline"/>
</p>

### Node Classification (10 lines)

```python
import torch
from vgl.graph import Graph
from vgl.tasks import NodeClassificationTask
from vgl.engine import Trainer

graph = Graph.homo(edge_index=edge_index, x=x, y=y,
                   train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

task = NodeClassificationTask(target="y",
                              split=("train_mask", "val_mask", "test_mask"),
                              metrics=["accuracy"])

trainer = Trainer(model=model, task=task,
                  optimizer=torch.optim.Adam, lr=1e-3, max_epochs=200,
                  monitor="val_accuracy", save_best_path="best.pt")

history = trainer.fit(graph, val_data=graph)
result  = trainer.test(graph)
print(f"Test accuracy: {result['accuracy']:.4f}")
```

### Graph Classification

```python
from vgl.dataloading import DataLoader, ListDataset, FullGraphSampler, SampleRecord
from vgl.tasks import GraphClassificationTask

samples = [SampleRecord(graph=g, metadata={}, sample_id=str(i)) for i, g in enumerate(graphs)]
loader  = DataLoader(dataset=ListDataset(samples), sampler=FullGraphSampler(),
                     batch_size=32, label_source="graph", label_key="y")

task    = GraphClassificationTask(target="y", label_source="graph")
trainer = Trainer(model=model, task=task, optimizer=torch.optim.Adam, lr=1e-3, max_epochs=50)
trainer.fit(loader)
```

### Link Prediction

```python
from vgl.dataloading import DataLoader, ListDataset, FullGraphSampler
from vgl.dataloading import LinkPredictionRecord
from vgl.tasks import LinkPredictionTask

samples = [
    LinkPredictionRecord(graph=graph, src_index=0, dst_index=1, label=1),
    LinkPredictionRecord(graph=graph, src_index=2, dst_index=0, label=0),
]
loader  = DataLoader(dataset=ListDataset(samples), sampler=FullGraphSampler(), batch_size=2)
task    = LinkPredictionTask(target="label")
trainer = Trainer(model=model, task=task, optimizer=torch.optim.Adam, lr=1e-3, max_epochs=50)
trainer.fit(loader)
```

### Temporal Event Prediction

```python
from vgl.dataloading import TemporalEventRecord
from vgl.tasks import TemporalEventPredictionTask

graph = Graph.temporal(nodes=nodes, edges=edges, time_attr="timestamp")
samples = [
    TemporalEventRecord(graph=graph, src_index=0, dst_index=1, timestamp=3, label=1),
    TemporalEventRecord(graph=graph, src_index=2, dst_index=0, timestamp=5, label=0),
]
task = TemporalEventPredictionTask(target="label")
```

---

## Installation

### Requirements

- Python ≥ 3.11
- PyTorch ≥ 2.4

### From Source

```bash
git clone https://github.com/<your-org>/VGL.git
cd VGL
pip install -e .
```

### With Development Dependencies

```bash
pip install -e ".[dev]"    # adds pytest, ruff, mypy
```

---

## Supported Convolution Layers

<p align="center">
  <img src="assets/conv-layers.svg" width="780" alt="VGL Convolution Layers"/>
</p>

All layers are built on the `MessagePassing` base class and share a consistent `forward(x, edge_index, ...)` interface.

For edge-aware operators on homogeneous graphs, `Graph.homo(...)` also accepts `edge_data={...}` and exposes it through `graph.edata`.

<details>
<summary><b>Full list of 50+ convolution operators</b></summary>

| Category | Layers |
|:--|:--|
| **Spectral** | `GCNConv`, `ChebConv`, `SGConv`, `TAGConv`, `ARMAConv`, `APPNPConv`, `BernConv`, `SSGConv` |
| **Attention** | `GATConv`, `GATv2Conv`, `TransformerConv`, `SuperGATConv`, `AGNNConv`, `FAGCNConv`, `FAConv`, `FeaStConv`, `DNAConv` |
| **Relation-aware** | `RGCNConv`, `RGATConv`, `HGTConv`, `HEATConv`, `HeteroConv` |
| **Edge-aware** | `NNConv`, `ECConv`, `GINEConv`, `GMMConv`, `CGConv`, `SplineConv`, `GatedGCNConv`, `PDNConv` |
| **Point / Geometric** | `PointNetConv`, `PointTransformerConv` |
| **Semantic Hetero** | `HANConv` |
| **Aggregation** | `SAGEConv`, `GINConv`, `PNAConv`, `MixHopConv`, `GraphConv`, `EdgeConv`, `GENConv`, `FiLMConv`, `MFConv`, `GeneralConv`, `SimpleConv`, `EGConv`, `LEConv`, `LightGCNConv`, `LGConv`, `ClusterGCNConv` |
| **Deep / Residual** | `GCN2Conv`, `DAGNNConv`, `GPRGNNConv`, `H2GCNConv`, `DirGNNConv`, `TWIRLSConv`, `AntiSymmetricConv`, `GatedGraphConv`, `ResGatedGraphConv` |
| **Transformer Encoders** | `GraphTransformerEncoder`, `GraphormerEncoder`, `GPSLayer`, `NAGphormerEncoder`, `SGFormerEncoder` |
| **Temporal** | `TimeEncoder`, `TGATLayer`, `TGATEncoder`, `IdentityTemporalMessage`, `LastMessageAggregator`, `MeanMessageAggregator`, `TGNMemory` |
| **Other** | `WLConvContinuous`, `GroupRevRes` (grouped reversible residual wrapper) |

</details>

Additionally, `HeteroConv` provides a wrapper for applying different convolution operators per edge type in heterogeneous graphs.

### Readout / Pooling

| Function | Description |
|:--|:--|
| `global_mean_pool` | Mean readout over all nodes |
| `global_sum_pool` | Sum readout over all nodes |
| `global_max_pool` | Max readout over all nodes |

---

## Framework Compatibility

VGL provides bidirectional conversion with the two most popular graph learning libraries:

```python
from vgl.compat.pyg import from_pyg, to_pyg
from vgl.compat.dgl import from_dgl, to_dgl

vgl_graph = from_pyg(pyg_data)       # PyG Data → VGL Graph
pyg_data  = to_pyg(vgl_graph)        # VGL Graph → PyG Data

vgl_graph = from_dgl(dgl_graph)      # DGL DGLGraph → VGL Graph
dgl_graph = to_dgl(vgl_graph)        # VGL Graph → DGL DGLGraph
```

---

## Examples

| Task | Script | Graph Type |
|:--|:--|:--|
| Node Classification | `examples/homo/node_classification.py` | Homogeneous |
| Graph Classification | `examples/homo/graph_classification.py` | Homogeneous |
| Link Prediction | `examples/homo/link_prediction.py` | Homogeneous |
| Conv Zoo (50+ layers) | `examples/homo/conv_zoo.py` | Homogeneous |
| Node Classification | `examples/hetero/node_classification.py` | Heterogeneous |
| Graph Classification | `examples/hetero/graph_classification.py` | Heterogeneous |
| Event Prediction (TGAT) | `examples/temporal/event_prediction.py` | Temporal |
| Event Prediction (TGN Memory) | `examples/temporal/memory_event_prediction.py` | Temporal |

Run any example with:

```bash
python examples/homo/node_classification.py
```

---

## Testing & Quality

```bash
# Run the full test suite
python -m pytest -v

# Lint check
python -m ruff check .

# Type check
python -m mypy vgl
```

The test suite covers:
- **Core**: Graph, batch, schema, views, heterogeneous/temporal constructors
- **Data**: Loaders, samplers, graph/link/temporal record pipelines
- **Training**: Trainer, tasks, metrics, callbacks, checkpoints, history
- **NN**: MessagePassing, all convolution operators, readout, GroupRevRes
- **Compat**: PyG and DGL adapter round-trips
- **Integration**: End-to-end workflows for all graph types and tasks

---

## Project Structure

```
VGL/
├── vgl/
│   ├── graph/          # Graph, GraphBatch, Schema, View, Stores
│   ├── nn/
│   │   ├── conv/       # 50+ convolution operators
│   │   ├── message_passing.py
│   │   ├── hetero.py   # HeteroConv
│   │   ├── readout.py  # global_mean/sum/max_pool
│   │   └── grouprevres.py
│   ├── tasks/          # Node/Graph/Link/Temporal task definitions
│   ├── engine/         # Trainer, callbacks, checkpoints, history
│   ├── metrics/        # Accuracy, Metric base
│   ├── dataloading/    # DataLoader, datasets, samplers, records
│   ├── transforms/     # Graph transforms
│   └── compat/         # PyG & DGL converters
├── examples/
│   ├── homo/           # Homogeneous graph examples
│   ├── hetero/         # Heterogeneous graph examples
│   └── temporal/       # Temporal graph examples
├── tests/              # Comprehensive test suite
├── docs/               # Documentation & design plans
└── pyproject.toml
```

---

## Documentation

- [Quickstart Guide](docs/quickstart.md)
- [Core Concepts](docs/core-concepts.md)
- [Migration Guide](docs/migration-guide.md)

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Ensure all tests pass (`python -m pytest -v`)
4. Ensure code quality (`python -m ruff check .` and `python -m mypy vgl`)
5. Submit a pull request

---

## License

See [LICENSE](LICENSE) for details.
