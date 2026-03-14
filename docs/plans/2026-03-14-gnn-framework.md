# GNN Framework Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a PyTorch-first graph learning framework with a stable public API, a unified `Graph` abstraction for homogeneous, heterogeneous, and temporal graphs, and PyG/DGL-style ergonomics at the API boundary.

**Architecture:** Implement the package from the inside out. Start by bootstrapping the repository and locking the public contract with tests, then build `gnn.core` around a schema-driven `Graph` and store model, layer in graph views and batching, add a small data pipeline, implement `MessagePassing` plus three reference convolution layers, then finish with tasks, a trainer, and compatibility adapters. Keep the operator set deliberately small so the public surface area stays stable.

**Tech Stack:** Python 3.11+, PyTorch, pytest, hatchling, ruff, mypy, typing_extensions

---

### Task 1: Bootstrap the Repository and Public Package Boundary

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `README.md`
- Create: `src/gnn/__init__.py`
- Create: `src/gnn/version.py`
- Test: `tests/test_package_exports.py`

**Step 1: Initialize version control and the source layout**

Run:

```bash
git init
mkdir src tests examples docs
mkdir src/gnn
```

Expected: a new git repository exists in `F:\gnn\gnn` and the top-level source directories exist.

**Step 2: Write the failing package export test**

```python
from gnn import __version__


def test_package_exposes_version_string():
    assert isinstance(__version__, str)
    assert __version__ == "0.1.0"
```

**Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_package_exports.py::test_package_exposes_version_string -v`
Expected: `FAIL` with `ModuleNotFoundError: No module named 'gnn'`

**Step 4: Write minimal package scaffolding**

`pyproject.toml`

```toml
[build-system]
requires = ["hatchling>=1.26.0"]
build-backend = "hatchling.build"

[project]
name = "gnn"
version = "0.1.0"
description = "Unified graph learning framework with PyTorch"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
  "torch>=2.4",
  "typing_extensions>=4.12",
]

[project.optional-dependencies]
dev = [
  "pytest>=8.3",
  "ruff>=0.6",
  "mypy>=1.11",
]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

`src/gnn/version.py`

```python
__version__ = "0.1.0"
```

`src/gnn/__init__.py`

```python
from gnn.version import __version__
```

**Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_package_exports.py::test_package_exposes_version_string -v`
Expected: `PASS`

**Step 6: Commit**

```bash
git add pyproject.toml .gitignore README.md src/gnn/__init__.py src/gnn/version.py tests/test_package_exports.py
git commit -m "chore: bootstrap package skeleton"
```

### Task 2: Define Core Errors and the Schema Contract

**Files:**
- Create: `src/gnn/core/__init__.py`
- Create: `src/gnn/core/errors.py`
- Create: `src/gnn/core/schema.py`
- Test: `tests/core/test_schema.py`

**Step 1: Write the failing schema contract test**

```python
import pytest

from gnn.core.schema import GraphSchema


def test_schema_tracks_node_edge_and_time_metadata():
    schema = GraphSchema(
        node_types=("paper", "author"),
        edge_types=(("author", "writes", "paper"),),
        node_features={"paper": ("x", "y"), "author": ("x",)},
        edge_features={("author", "writes", "paper"): ("timestamp",)},
        time_attr="timestamp",
    )

    assert schema.node_types == ("paper", "author")
    assert schema.edge_types == (("author", "writes", "paper"),)
    assert schema.time_attr == "timestamp"


def test_schema_rejects_unknown_time_field():
    with pytest.raises(ValueError, match="time_attr"):
        GraphSchema(
            node_types=("paper",),
            edge_types=(("paper", "cites", "paper"),),
            node_features={"paper": ("x",)},
            edge_features={("paper", "cites", "paper"): ("weight",)},
            time_attr="timestamp",
        )
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/core/test_schema.py -v`
Expected: `FAIL` with `ImportError` because `GraphSchema` does not exist yet.

**Step 3: Write minimal errors and schema**

`src/gnn/core/errors.py`

```python
class GNNError(Exception):
    pass


class SchemaError(GNNError, ValueError):
    pass
```

`src/gnn/core/schema.py`

```python
from dataclasses import dataclass

from gnn.core.errors import SchemaError


@dataclass(frozen=True, slots=True)
class GraphSchema:
    node_types: tuple[str, ...]
    edge_types: tuple[tuple[str, str, str], ...]
    node_features: dict[str, tuple[str, ...]]
    edge_features: dict[tuple[str, str, str], tuple[str, ...]]
    time_attr: str | None = None

    def __post_init__(self) -> None:
        if self.time_attr is None:
            return
        if any(self.time_attr in fields for fields in self.node_features.values()):
            return
        if any(self.time_attr in fields for fields in self.edge_features.values()):
            return
        raise SchemaError(f"time_attr '{self.time_attr}' is not declared in schema")
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/core/test_schema.py -v`
Expected: `PASS`

**Step 5: Commit**

```bash
git add src/gnn/core/__init__.py src/gnn/core/errors.py src/gnn/core/schema.py tests/core/test_schema.py
git commit -m "feat: add graph schema contracts"
```

### Task 3: Implement NodeStore, EdgeStore, and Unified Graph Constructors

**Files:**
- Create: `src/gnn/core/stores.py`
- Create: `src/gnn/core/graph.py`
- Modify: `src/gnn/core/__init__.py`
- Modify: `src/gnn/__init__.py`
- Test: `tests/core/test_graph_homo.py`
- Test: `tests/core/test_graph_multi_type.py`

**Step 1: Write failing graph constructor tests**

`tests/core/test_graph_homo.py`

```python
import torch

from gnn import Graph


def test_homo_graph_exposes_pyg_style_fields():
    edge_index = torch.tensor([[0, 1], [1, 0]])
    x = torch.randn(2, 4)
    y = torch.tensor([0, 1])

    graph = Graph.homo(edge_index=edge_index, x=x, y=y)

    assert torch.equal(graph.edge_index, edge_index)
    assert torch.equal(graph.x, x)
    assert torch.equal(graph.y, y)
    assert torch.equal(graph.ndata["x"], x)
```

`tests/core/test_graph_multi_type.py`

```python
import torch

from gnn import Graph


def test_hetero_graph_exposes_typed_node_and_edge_stores():
    graph = Graph.hetero(
        nodes={
            "paper": {"x": torch.randn(3, 8)},
            "author": {"x": torch.randn(2, 8)},
        },
        edges={
            ("author", "writes", "paper"): {
                "edge_index": torch.tensor([[0, 1], [1, 2]])
            }
        },
    )

    assert graph.schema.node_types == ("author", "paper")
    assert graph.nodes["paper"].x.shape == (3, 8)
    assert graph.edges[("author", "writes", "paper")].edge_index.shape == (2, 2)


def test_temporal_graph_keeps_time_metadata():
    graph = Graph.temporal(
        nodes={"user": {"x": torch.randn(2, 4)}},
        edges={
            ("user", "interacts", "user"): {
                "edge_index": torch.tensor([[0, 1], [1, 0]]),
                "timestamp": torch.tensor([1, 2]),
            }
        },
        time_attr="timestamp",
    )

    assert graph.schema.time_attr == "timestamp"
    assert torch.equal(
        graph.edges[("user", "interacts", "user")].timestamp,
        torch.tensor([1, 2]),
    )
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/core/test_graph_homo.py tests/core/test_graph_multi_type.py -v`
Expected: `FAIL` with `ImportError` because `Graph` does not exist yet.

**Step 3: Write minimal stores and graph implementation**

`src/gnn/core/stores.py`

```python
from dataclasses import dataclass, field

import torch


@dataclass(slots=True)
class NodeStore:
    type_name: str
    data: dict[str, torch.Tensor] = field(default_factory=dict)

    def __getattr__(self, name: str) -> torch.Tensor:
        try:
            return self.data[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


@dataclass(slots=True)
class EdgeStore:
    type_name: tuple[str, str, str]
    data: dict[str, torch.Tensor] = field(default_factory=dict)

    def __getattr__(self, name: str) -> torch.Tensor:
        try:
            return self.data[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
```

`src/gnn/core/graph.py`

```python
from dataclasses import dataclass

from gnn.core.schema import GraphSchema
from gnn.core.stores import EdgeStore, NodeStore


@dataclass(slots=True)
class Graph:
    schema: GraphSchema
    nodes: dict[str, NodeStore]
    edges: dict[tuple[str, str, str], EdgeStore]

    @classmethod
    def homo(cls, *, edge_index, **node_data):
        nodes = {"node": NodeStore("node", dict(node_data))}
        edges = {("node", "to", "node"): EdgeStore(("node", "to", "node"), {"edge_index": edge_index})}
        schema = GraphSchema(
            node_types=("node",),
            edge_types=(("node", "to", "node"),),
            node_features={"node": tuple(node_data.keys())},
            edge_features={("node", "to", "node"): ("edge_index",)},
        )
        return cls(schema=schema, nodes=nodes, edges=edges)

    @classmethod
    def hetero(cls, *, nodes, edges, time_attr=None):
        node_stores = {name: NodeStore(name, dict(data)) for name, data in nodes.items()}
        edge_stores = {etype: EdgeStore(etype, dict(data)) for etype, data in edges.items()}
        schema = GraphSchema(
            node_types=tuple(sorted(node_stores)),
            edge_types=tuple(sorted(edge_stores)),
            node_features={name: tuple(store.data.keys()) for name, store in node_stores.items()},
            edge_features={name: tuple(store.data.keys()) for name, store in edge_stores.items()},
            time_attr=time_attr,
        )
        return cls(schema=schema, nodes=node_stores, edges=edge_stores)

    @classmethod
    def temporal(cls, *, nodes, edges, time_attr):
        return cls.hetero(nodes=nodes, edges=edges, time_attr=time_attr)

    @property
    def x(self):
        return self.nodes["node"].x

    @property
    def y(self):
        return self.nodes["node"].y

    @property
    def edge_index(self):
        return self.edges[("node", "to", "node")].edge_index

    @property
    def ndata(self):
        return self.nodes["node"].data

    @property
    def edata(self):
        return self.edges[("node", "to", "node")].data
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/core/test_graph_homo.py tests/core/test_graph_multi_type.py -v`
Expected: `PASS`

**Step 5: Commit**

```bash
git add src/gnn/core/stores.py src/gnn/core/graph.py src/gnn/core/__init__.py src/gnn/__init__.py tests/core/test_graph_homo.py tests/core/test_graph_multi_type.py
git commit -m "feat: add unified graph constructors"
```

### Task 4: Add GraphView, Snapshot, Window, and GraphBatch

**Files:**
- Create: `src/gnn/core/view.py`
- Create: `src/gnn/core/batch.py`
- Modify: `src/gnn/core/graph.py`
- Test: `tests/core/test_graph_view.py`
- Test: `tests/core/test_graph_batch.py`

**Step 1: Write failing view and batch tests**

```python
import torch

from gnn import Graph
from gnn.core.batch import GraphBatch


def test_snapshot_filters_temporal_edges_without_copying_features():
    graph = Graph.temporal(
        nodes={"user": {"x": torch.randn(2, 4)}},
        edges={
            ("user", "interacts", "user"): {
                "edge_index": torch.tensor([[0, 1], [1, 0]]),
                "timestamp": torch.tensor([1, 3]),
            }
        },
        time_attr="timestamp",
    )

    snapshot = graph.snapshot(2)

    assert snapshot.schema.time_attr == "timestamp"
    assert snapshot.edges[("user", "interacts", "user")].edge_index.shape[1] == 1
    assert snapshot.nodes["user"].x.data_ptr() == graph.nodes["user"].x.data_ptr()


def test_graph_batch_tracks_membership():
    g1 = Graph.homo(edge_index=torch.tensor([[0], [1]]), x=torch.randn(2, 4), y=torch.tensor([0, 1]))
    g2 = Graph.homo(edge_index=torch.tensor([[0], [1]]), x=torch.randn(2, 4), y=torch.tensor([1, 0]))

    batch = GraphBatch.from_graphs([g1, g2])

    assert batch.num_graphs == 2
    assert batch.graph_index.shape[0] == 4
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/core/test_graph_view.py tests/core/test_graph_batch.py -v`
Expected: `FAIL` because `snapshot` and `GraphBatch` do not exist yet.

**Step 3: Write minimal view and batch implementation**

`src/gnn/core/view.py`

```python
from dataclasses import dataclass


@dataclass(slots=True)
class GraphView:
    base: "Graph"
    nodes: dict
    edges: dict
    schema: object
```

`src/gnn/core/batch.py`

```python
from dataclasses import dataclass

import torch


@dataclass(slots=True)
class GraphBatch:
    graphs: list
    graph_index: torch.Tensor

    @classmethod
    def from_graphs(cls, graphs):
        counts = [graph.x.size(0) for graph in graphs]
        graph_index = torch.repeat_interleave(torch.arange(len(graphs)), torch.tensor(counts))
        return cls(graphs=graphs, graph_index=graph_index)

    @property
    def num_graphs(self) -> int:
        return len(self.graphs)
```

Extend `Graph` with:

```python
def snapshot(self, t):
    if self.schema.time_attr is None:
        raise ValueError("snapshot requires a temporal graph")
    edges = {}
    for edge_type, store in self.edges.items():
        mask = store.data[self.schema.time_attr] <= t
        edge_data = dict(store.data)
        edge_data["edge_index"] = store.edge_index[:, mask]
        edge_data[self.schema.time_attr] = store.data[self.schema.time_attr][mask]
        edges[edge_type] = type(store)(edge_type, edge_data)
    return GraphView(base=self, nodes=self.nodes, edges=edges, schema=self.schema)
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/core/test_graph_view.py tests/core/test_graph_batch.py -v`
Expected: `PASS`

**Step 5: Commit**

```bash
git add src/gnn/core/view.py src/gnn/core/batch.py src/gnn/core/graph.py tests/core/test_graph_view.py tests/core/test_graph_batch.py
git commit -m "feat: add graph views and batching"
```

### Task 5: Add Dataset, Sampler, Transform, and Loader Contracts

**Files:**
- Create: `src/gnn/data/__init__.py`
- Create: `src/gnn/data/dataset.py`
- Create: `src/gnn/data/transform.py`
- Create: `src/gnn/data/sampler.py`
- Create: `src/gnn/data/loader.py`
- Test: `tests/data/test_loader.py`

**Step 1: Write the failing data pipeline test**

```python
import torch

from gnn import Graph
from gnn.data.dataset import ListDataset
from gnn.data.loader import Loader
from gnn.data.sampler import FullGraphSampler


def test_loader_returns_graph_batch_for_list_dataset():
    graphs = [
        Graph.homo(edge_index=torch.tensor([[0], [1]]), x=torch.randn(2, 4), y=torch.tensor([0, 1])),
        Graph.homo(edge_index=torch.tensor([[0], [1]]), x=torch.randn(2, 4), y=torch.tensor([1, 0])),
    ]

    dataset = ListDataset(graphs)
    loader = Loader(dataset=dataset, sampler=FullGraphSampler(), batch_size=2)

    batch = next(iter(loader))

    assert batch.num_graphs == 2
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/data/test_loader.py -v`
Expected: `FAIL` because the data pipeline classes do not exist yet.

**Step 3: Write minimal data pipeline implementation**

`src/gnn/data/dataset.py`

```python
class ListDataset:
    def __init__(self, graphs):
        self.graphs = list(graphs)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, index):
        return self.graphs[index]
```

`src/gnn/data/sampler.py`

```python
class FullGraphSampler:
    def sample(self, graph):
        return graph
```

`src/gnn/data/loader.py`

```python
from gnn.core.batch import GraphBatch


class Loader:
    def __init__(self, dataset, sampler, batch_size):
        self.dataset = dataset
        self.sampler = sampler
        self.batch_size = batch_size

    def __iter__(self):
        batch = []
        for graph in self.dataset.graphs:
            batch.append(self.sampler.sample(graph))
            if len(batch) == self.batch_size:
                yield GraphBatch.from_graphs(batch)
                batch = []
        if batch:
            yield GraphBatch.from_graphs(batch)
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/data/test_loader.py -v`
Expected: `PASS`

**Step 5: Commit**

```bash
git add src/gnn/data/__init__.py src/gnn/data/dataset.py src/gnn/data/transform.py src/gnn/data/sampler.py src/gnn/data/loader.py tests/data/test_loader.py
git commit -m "feat: add dataset and loader contracts"
```

### Task 6: Implement MessagePassing and Reference Convolution Layers

**Files:**
- Create: `src/gnn/nn/__init__.py`
- Create: `src/gnn/nn/message_passing.py`
- Create: `src/gnn/nn/conv/__init__.py`
- Create: `src/gnn/nn/conv/gcn.py`
- Create: `src/gnn/nn/conv/sage.py`
- Create: `src/gnn/nn/conv/gat.py`
- Create: `src/gnn/nn/hetero.py`
- Create: `src/gnn/nn/temporal.py`
- Test: `tests/nn/test_message_passing.py`
- Test: `tests/nn/test_convs.py`

**Step 1: Write failing operator tests**

```python
import torch

from gnn import Graph
from gnn.nn.conv.gcn import GCNConv
from gnn.nn.conv.sage import SAGEConv


def test_gcn_conv_accepts_graph_input():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        x=torch.randn(2, 4),
        y=torch.tensor([0, 1]),
    )
    conv = GCNConv(in_channels=4, out_channels=3)

    out = conv(graph)

    assert out.shape == (2, 3)


def test_sage_conv_accepts_x_and_edge_index():
    x = torch.randn(2, 4)
    edge_index = torch.tensor([[0, 1], [1, 0]])
    conv = SAGEConv(in_channels=4, out_channels=3)

    out = conv(x, edge_index)

    assert out.shape == (2, 3)
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/nn/test_message_passing.py tests/nn/test_convs.py -v`
Expected: `FAIL` because the operator modules do not exist yet.

**Step 3: Write minimal message passing core and two operators**

`src/gnn/nn/message_passing.py`

```python
import torch
from torch import nn


class MessagePassing(nn.Module):
    def forward(self, graph_or_x, edge_index=None):
        if edge_index is None:
            x = graph_or_x.x
            edge_index = graph_or_x.edge_index
        else:
            x = graph_or_x
        row, col = edge_index
        messages = self.message(x[row], x[col])
        out = torch.zeros_like(x)
        out.index_add_(0, col, messages)
        return self.update(out)

    def message(self, x_j, x_i):
        return x_j

    def update(self, aggr_out):
        return aggr_out
```

`src/gnn/nn/conv/gcn.py`

```python
from torch import nn

from gnn.nn.message_passing import MessagePassing


class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def update(self, aggr_out):
        return self.linear(aggr_out)
```

`src/gnn/nn/conv/sage.py`

```python
import torch
from torch import nn

from gnn.nn.message_passing import MessagePassing


class SAGEConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = nn.Linear(in_channels * 2, out_channels)

    def forward(self, graph_or_x, edge_index=None):
        if edge_index is None:
            x = graph_or_x.x
            edge_index = graph_or_x.edge_index
        else:
            x = graph_or_x
        neigh = super().forward(x, edge_index)
        return self.linear(torch.cat([x, neigh], dim=-1))
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/nn/test_convs.py -v`
Expected: `PASS`

**Step 5: Expand to `GATConv`, `HeteroConv`, and temporal encoding with tests, then commit**

```bash
python -m pytest tests/nn -v
git add src/gnn/nn src/gnn/__init__.py tests/nn
git commit -m "feat: add message passing and reference convs"
```

### Task 7: Add Tasks, Metrics, Evaluator, and Trainer

**Files:**
- Create: `src/gnn/train/__init__.py`
- Create: `src/gnn/train/task.py`
- Create: `src/gnn/train/tasks.py`
- Create: `src/gnn/train/metrics.py`
- Create: `src/gnn/train/evaluator.py`
- Create: `src/gnn/train/trainer.py`
- Test: `tests/train/test_tasks.py`
- Test: `tests/train/test_trainer.py`

**Step 1: Write failing task and trainer tests**

```python
import torch
from torch import nn

from gnn import Graph
from gnn.train.tasks import NodeClassificationTask
from gnn.train.trainer import Trainer


class LinearNodeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, graph):
        return self.linear(graph.x)


def test_node_classification_task_computes_loss():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        x=torch.randn(2, 4),
        y=torch.tensor([0, 1]),
        train_mask=torch.tensor([True, True]),
    )
    task = NodeClassificationTask(target="y", split=("train_mask", "train_mask", "train_mask"))
    logits = torch.randn(2, 2, requires_grad=True)

    loss = task.loss(graph, logits, stage="train")

    assert loss.ndim == 0


def test_trainer_runs_single_epoch():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        x=torch.randn(2, 4),
        y=torch.tensor([0, 1]),
        train_mask=torch.tensor([True, True]),
        val_mask=torch.tensor([True, True]),
        test_mask=torch.tensor([True, True]),
    )
    model = LinearNodeModel()
    task = NodeClassificationTask(target="y", split=("train_mask", "val_mask", "test_mask"))
    trainer = Trainer(model=model, task=task, optimizer=torch.optim.Adam, lr=1e-2, max_epochs=1)

    history = trainer.fit(graph)

    assert history["epochs"] == 1
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/train/test_tasks.py tests/train/test_trainer.py -v`
Expected: `FAIL` because the training layer does not exist yet.

**Step 3: Write minimal task and trainer loop**

`src/gnn/train/tasks.py`

```python
import torch.nn.functional as F


class NodeClassificationTask:
    def __init__(self, target, split, loss="cross_entropy", metrics=None):
        self.target = target
        self.train_key, self.val_key, self.test_key = split
        self.loss_name = loss
        self.metrics = metrics or []

    def loss(self, graph, logits, stage):
        mask = getattr(graph, f"{stage}_mask")
        target = getattr(graph, self.target)
        return F.cross_entropy(logits[mask], target[mask])
```

`src/gnn/train/trainer.py`

```python
class Trainer:
    def __init__(self, model, task, optimizer, lr, max_epochs):
        self.model = model
        self.task = task
        self.optimizer = optimizer(self.model.parameters(), lr=lr)
        self.max_epochs = max_epochs

    def fit(self, graph):
        for _ in range(self.max_epochs):
            self.model.train()
            self.optimizer.zero_grad()
            logits = self.model(graph)
            loss = self.task.loss(graph, logits, stage="train")
            loss.backward()
            self.optimizer.step()
        return {"epochs": self.max_epochs}
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/train/test_tasks.py tests/train/test_trainer.py -v`
Expected: `PASS`

**Step 5: Expand to graph classification, link prediction, temporal event prediction, metrics, evaluator, then commit**

```bash
python -m pytest tests/train -v
git add src/gnn/train tests/train
git commit -m "feat: add training task and trainer layer"
```

### Task 8: Add PyG and DGL Compatibility Adapters

**Files:**
- Create: `src/gnn/compat/__init__.py`
- Create: `src/gnn/compat/pyg.py`
- Create: `src/gnn/compat/dgl.py`
- Modify: `src/gnn/core/graph.py`
- Test: `tests/compat/test_pyg_adapter.py`
- Test: `tests/compat/test_dgl_adapter.py`

**Step 1: Write failing adapter tests**

```python
import pytest
import torch

from gnn import Graph


def test_graph_round_trips_to_pyg_data():
    pyg = pytest.importorskip("torch_geometric.data")
    data = pyg.Data(
        x=torch.randn(2, 4),
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        y=torch.tensor([0, 1]),
    )

    graph = Graph.from_pyg(data)
    restored = graph.to_pyg()

    assert torch.equal(restored.edge_index, data.edge_index)
    assert torch.equal(restored.x, data.x)


def test_graph_round_trips_to_dgl_graph():
    dgl = pytest.importorskip("dgl")
    source = torch.tensor([0, 1])
    destination = torch.tensor([1, 0])
    dgl_graph = dgl.graph((source, destination))
    dgl_graph.ndata["x"] = torch.randn(2, 4)

    graph = Graph.from_dgl(dgl_graph)
    restored = graph.to_dgl()

    assert restored.num_nodes() == dgl_graph.num_nodes()
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/compat/test_pyg_adapter.py tests/compat/test_dgl_adapter.py -v`
Expected: `FAIL` because the compatibility entrypoints do not exist yet.

**Step 3: Write lazy compatibility adapters**

`src/gnn/compat/pyg.py`

```python
from gnn.core.graph import Graph


def from_pyg(data):
    return Graph.homo(edge_index=data.edge_index, x=data.x, y=getattr(data, "y", None))


def to_pyg(graph):
    from torch_geometric.data import Data

    return Data(x=graph.x, edge_index=graph.edge_index, y=getattr(graph, "y", None))
```

`src/gnn/compat/dgl.py`

```python
import torch

from gnn.core.graph import Graph


def from_dgl(dgl_graph):
    src, dst = dgl_graph.edges()
    return Graph.homo(
        edge_index=torch.stack([src, dst]),
        **dict(dgl_graph.ndata),
    )


def to_dgl(graph):
    import dgl

    row, col = graph.edge_index
    dgl_graph = dgl.graph((row, col), num_nodes=graph.x.size(0))
    for key, value in graph.ndata.items():
        dgl_graph.ndata[key] = value
    return dgl_graph
```

**Step 4: Run tests to verify they pass or skip correctly**

Run: `python -m pytest tests/compat -v`
Expected:

- `PASS` when optional dependencies are installed
- `SKIP` when `torch_geometric` or `dgl` is unavailable

**Step 5: Commit**

```bash
git add src/gnn/compat src/gnn/core/graph.py tests/compat
git commit -m "feat: add pyg and dgl adapters"
```

### Task 9: Add Examples, Concept Docs, and End-to-End Integration Tests

**Files:**
- Create: `examples/homo/node_classification.py`
- Create: `examples/hetero/node_classification.py`
- Create: `examples/temporal/event_prediction.py`
- Create: `docs/quickstart.md`
- Create: `docs/core-concepts.md`
- Create: `docs/migration-guide.md`
- Test: `tests/integration/test_end_to_end_homo.py`
- Test: `tests/integration/test_end_to_end_hetero.py`
- Test: `tests/integration/test_end_to_end_temporal.py`

**Step 1: Write the failing homogeneous integration test**

```python
import torch
from torch import nn

from gnn import Graph
from gnn.train.tasks import NodeClassificationTask
from gnn.train.trainer import Trainer


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, graph):
        return self.linear(graph.x)


def test_end_to_end_homo_training_runs():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        x=torch.randn(2, 4),
        y=torch.tensor([0, 1]),
        train_mask=torch.tensor([True, True]),
        val_mask=torch.tensor([True, True]),
        test_mask=torch.tensor([True, True]),
    )
    task = NodeClassificationTask(target="y", split=("train_mask", "val_mask", "test_mask"))
    trainer = Trainer(model=TinyModel(), task=task, optimizer=torch.optim.Adam, lr=1e-2, max_epochs=1)

    result = trainer.fit(graph)

    assert result["epochs"] == 1
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/integration/test_end_to_end_homo.py -v`
Expected: `FAIL` until the full training path is wired together cleanly.

**Step 3: Add the minimal examples and docs that mirror the tested flows**

`examples/homo/node_classification.py`

```python
import torch

from gnn import Graph
from gnn.nn.conv.gcn import GCNConv
from gnn.train.tasks import NodeClassificationTask
from gnn.train.trainer import Trainer
```

Docs should explain:

- what `Graph`, `GraphView`, and `GraphBatch` mean
- how the package differs from PyG and DGL internally
- how to migrate a simple `Data` or `DGLGraph` flow

**Step 4: Run the full test suite and static checks**

```bash
python -m pytest -v
python -m ruff check .
python -m mypy src
```

Expected:

- required tests `PASS`
- optional compat tests `PASS` or `SKIP`
- static checks are clean

**Step 5: Commit**

```bash
git add examples docs tests/integration README.md
git commit -m "docs: add examples and integration coverage"
```

### Task 10: Final Verification Gate Before Any Release Tag

**Files:**
- Modify: `README.md`
- Modify: `docs/core-concepts.md`
- Modify: `docs/migration-guide.md`

**Step 1: Verify the public API list matches the implementation**

Run: `python -c "import gnn; print(sorted(name for name in dir(gnn) if not name.startswith('_')))"`.
Expected: only intended public objects are exported.

**Step 2: Verify the contract tests still describe the current API**

Run: `python -m pytest tests/core tests/compat -v`
Expected: no contract tests fail due to API drift.

**Step 3: Verify the examples still execute**

```bash
python examples/homo/node_classification.py
python examples/hetero/node_classification.py
python examples/temporal/event_prediction.py
```

Expected: each example runs without import or API errors.

**Step 4: Update docs if any public name or workflow changed**

Expected: README and concept docs match the actual shipped API.

**Step 5: Commit**

```bash
git add README.md docs/core-concepts.md docs/migration-guide.md
git commit -m "chore: finalize api verification docs"
```

## Notes

- Keep `compat` dependencies optional. Adapter tests should skip when external libraries are unavailable.
- Do not add more public exports than the design document allows.
- If a new feature cannot be expressed cleanly through the unified `Graph` semantics, revisit the design before implementing it.
- Prefer contract tests before implementation changes for every public API adjustment.
