<p align="center">
  <img src="https://raw.githubusercontent.com/skygazer42/sky-vgl/main/assets/logo.svg" width="420" alt="VGL – Versatile Graph Learning"/>
</p>

<p align="center">
  <b>Unified graph learning framework with a stable core abstraction for homogeneous, heterogeneous, and temporal graphs.</b>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-%E2%89%A53.10-blue?style=flat-square&logo=python&logoColor=white" alt="Python 3.10+"/></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-%E2%89%A52.4-ee4c2c?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch 2.4+"/></a>
  <img src="https://img.shields.io/pypi/v/sky-vgl?style=flat-square" alt="Latest sky-vgl version"/>
  <img src="https://img.shields.io/badge/license-see%20LICENSE-green?style=flat-square" alt="License"/>
</p>

---

**VGL** (Versatile Graph Learning) is a PyTorch-first graph learning library that provides **one canonical `Graph` abstraction** for homogeneous, heterogeneous, and temporal graphs — plus a batteries-included training pipeline from data loading to evaluation.

<p align="center">
  <img src="https://raw.githubusercontent.com/skygazer42/sky-vgl/main/assets/graph-types.svg" width="780" alt="Supported Graph Types"/>
</p>

---

## Key Features

- **Unified `Graph` object** — one data structure for homogeneous, heterogeneous, and temporal graphs with schema validation, views, and batching
- **60+ convolution layers** — GCN, GAT, SAGE, GIN, Transformer, RGCN, HGT, HAN, and more on a clean `MessagePassing` base
- **Complete training pipeline** — `Trainer` + `Task` + `Metric` with early stopping, checkpoints, mixed precision, and multiple loggers
- **Flexible sampling strategies** — NeighborSampler, GraphSAINT, ClusterGCN, RandomWalk for node, link, and temporal workloads
- **Framework interoperability** — bidirectional adapters for DGL, PyG, NetworkX, and CSV/edge-list formats
- **Built-in datasets** — Cora, Citeseer, PubMed, MUTAG, PROTEINS, and more with composable transforms

---

## Quick Start

```python
import torch
from vgl import PlanetoidDataset, Trainer
from vgl.transforms import Compose, NormalizeFeatures
from vgl.nn import GCNConv
from vgl.tasks import NodeClassificationTask
import torch.nn as nn
import torch.nn.functional as F

# Load data
dataset = PlanetoidDataset(root="data", name="Cora", transform=Compose([NormalizeFeatures()]))
graph = dataset[0]

# Define model
class GCN(nn.Module):
    def __init__(self, in_ch, hid_ch, out_ch):
        super().__init__()
        self.conv1 = GCNConv(in_ch, hid_ch)
        self.conv2 = GCNConv(hid_ch, out_ch)
    def forward(self, graph):
        x = F.dropout(F.relu(self.conv1(graph.x, graph)), p=0.5, training=self.training)
        return self.conv2(x, graph)

model = GCN(graph.x.size(1), 64, graph.y.max().item() + 1)

# Train
task = NodeClassificationTask(target="y", split=("train_mask", "val_mask", "test_mask"), metrics=["accuracy"])
trainer = Trainer(model=model, task=task, optimizer=torch.optim.Adam, lr=1e-2, max_epochs=200)
history = trainer.fit(graph, val_data=graph)
print(trainer.test(graph))
```

> **More tutorials** — graph classification, link prediction, temporal events, and mini-batch sampling are covered in the [Quick Start guide](https://skygazer42.github.io/sky-vgl/getting-started/quickstart/).

---

## Installation

```bash
pip install sky-vgl                       # core
pip install "sky-vgl[full]"               # all optional extras (scipy, networkx, dgl, pyg, tensorboard)
pip install "sky-vgl[networkx]"          # NetworkX interoperability
pip install "sky-vgl[dgl]"               # DGL interoperability
pip install "sky-vgl[pyg]"               # PyTorch Geometric interoperability
```

> **From source, dev deps, and extras** — see the [Installation guide](https://skygazer42.github.io/sky-vgl/getting-started/installation/).

### Source Install

```bash
git clone https://github.com/skygazer42/sky-vgl.git
cd sky-vgl
pip install -e ".[dev]"
```

Use the editable install when developing locally or validating the latest branch state.

---

## Architecture

<p align="center">
  <img src="https://raw.githubusercontent.com/skygazer42/sky-vgl/main/assets/architecture.svg" width="780" alt="VGL Architecture Overview"/>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/skygazer42/sky-vgl/main/assets/pipeline.svg" width="780" alt="VGL Training Pipeline"/>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/skygazer42/sky-vgl/main/assets/conv-layers.svg" width="780" alt="VGL Convolution Layer Coverage"/>
</p>

| Module | Description |
|:--|:--|
| `vgl.graph` | `Graph`, `GraphBatch`, `GraphSchema`, `GraphView`, `Block`, `HeteroBlock` |
| `vgl.nn` | 60+ convolution layers, graph/temporal encoders, readout, `HeteroConv` |
| `vgl.dataloading` | `DataLoader`, samplers, sampling plans, sample records |
| `vgl.engine` | `Trainer`, callbacks, checkpoints, loggers, training history |
| `vgl.tasks` | Node classification, graph classification, link prediction, temporal prediction |
| `vgl.ops` | Subgraph extraction, adjacency/Laplacian views, random walks, message-flow rewrites |
| `vgl.sparse` | `SparseTensor`, COO/CSR/CSC, spmm, sddmm, edge softmax |
| `vgl.storage` | `FeatureStore`, `GraphStore`, mmap-backed tensors |
| `vgl.data` | Built-in datasets, dataset registry, on-disk formats |
| `vgl.distributed` | Partition metadata, local shards, sampling coordination |
| `vgl.transforms` | `Compose`, `NormalizeFeatures`, `RandomNodeSplit`, and more |
| `vgl.compat` | DGL, PyG, NetworkX, CSV interoperability adapters |

> **Deep dive** — see the [Architecture overview](https://skygazer42.github.io/sky-vgl/architecture/).

---

## Documentation

Full documentation is hosted at **[skygazer42.github.io/sky-vgl](https://skygazer42.github.io/sky-vgl/)**.

| Section | Description |
|:--|:--|
| [Quick Start](https://skygazer42.github.io/sky-vgl/getting-started/quickstart/) | 5-minute tutorial from install to first training run |
| [User Guide](https://skygazer42.github.io/sky-vgl/guide/) | In-depth guides for every task type and graph type |
| [API Reference](https://skygazer42.github.io/sky-vgl/api/) | Complete API docs for all public modules |
| [Examples](https://skygazer42.github.io/sky-vgl/examples/) | End-to-end code examples and conv-layer zoo |
| [Architecture](https://skygazer42.github.io/sky-vgl/architecture/) | Package structure, design layers, and module responsibilities |
| [Core Concepts](https://skygazer42.github.io/sky-vgl/core-concepts/) | Graph, Task, Trainer, and framework interop explained |

---

## Examples

| Task | Script | Graph Type |
|:--|:--|:--|
| Node Classification | `examples/homo/node_classification.py` | Homogeneous |
| Graph Classification | `examples/homo/graph_classification.py` | Homogeneous |
| Link Prediction | `examples/homo/link_prediction.py` | Homogeneous |
| Conv Zoo (60+ layers) | `examples/homo/conv_zoo.py` | Homogeneous |
| Hetero Node Classification | `examples/hetero/node_classification.py` | Heterogeneous |
| Hetero Link Prediction | `examples/hetero/link_prediction.py` | Heterogeneous |
| Hetero Graph Classification | `examples/hetero/graph_classification.py` | Heterogeneous |
| Event Prediction (TGAT) | `examples/temporal/event_prediction.py` | Temporal |
| Event Prediction (TGN) | `examples/temporal/memory_event_prediction.py` | Temporal |

> **Full examples with explanations** — see the [Examples page](https://skygazer42.github.io/sky-vgl/examples/).

---

## Testing

```bash
python -m pytest -v              # full test suite
python -m ruff check .           # lint
python -m mypy vgl               # type check
```

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
