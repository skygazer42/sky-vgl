# Graph Classification Phase 2 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a stable graph classification training path that supports both many-small-graph datasets and sampled subgraph classification from a single large graph, with labels coming from graph objects or sample metadata.

**Architecture:** Keep the existing unified graph core intact and extend the training pipeline around a richer `GraphBatch`. Introduce `SampleRecord` as the pre-collation unit, add graph-level readout helpers, add `GraphClassificationTask`, and expand the data pipeline so that both many-small-graph and sampled-subgraph paths end at the same batch contract. Preserve the node classification workflow and keep phase 2 narrowly focused on graph classification.

**Tech Stack:** Python 3.11+, PyTorch, pytest, ruff, mypy

---

### Task 1: Add SampleRecord and Rich GraphBatch Metadata

**Files:**
- Modify: `src/gnn/core/batch.py`
- Create: `src/gnn/data/sample.py`
- Modify: `src/gnn/core/__init__.py`
- Test: `tests/core/test_graph_batch_graph_classification.py`

**Step 1: Write the failing batch metadata test**

```python
import torch

from gnn import Graph
from gnn.core.batch import GraphBatch
from gnn.data.sample import SampleRecord


def test_graph_batch_tracks_graph_ptr_labels_and_metadata():
    graphs = [
        Graph.homo(edge_index=torch.tensor([[0], [1]]), x=torch.randn(2, 4), y=torch.tensor([1])),
        Graph.homo(edge_index=torch.tensor([[0], [1]]), x=torch.randn(3, 4), y=torch.tensor([0])),
    ]
    samples = [
        SampleRecord(graph=graphs[0], metadata={"label": 1}, sample_id="g0"),
        SampleRecord(graph=graphs[1], metadata={"label": 0}, sample_id="g1"),
    ]

    batch = GraphBatch.from_samples(samples, label_key="y", label_source="graph")

    assert batch.num_graphs == 2
    assert torch.equal(batch.graph_ptr, torch.tensor([0, 2, 5]))
    assert torch.equal(batch.labels, torch.tensor([1, 0]))
    assert batch.metadata[0]["label"] == 1
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/core/test_graph_batch_graph_classification.py -v`
Expected: `FAIL` because `SampleRecord` and rich batch metadata do not exist yet.

**Step 3: Write minimal implementation**

`src/gnn/data/sample.py`

```python
from dataclasses import dataclass, field


@dataclass(slots=True)
class SampleRecord:
    graph: object
    metadata: dict = field(default_factory=dict)
    sample_id: str | None = None
    source_graph_id: str | None = None
    subgraph_seed: object | None = None
```

Update `src/gnn/core/batch.py` so `GraphBatch` can be built from `SampleRecord` objects and stores:

- `graph_ptr`
- `labels`
- `metadata`

Use the smallest possible implementation that satisfies the test.

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/core/test_graph_batch_graph_classification.py -v`
Expected: `PASS`

**Step 5: Commit**

```bash
git add src/gnn/core/batch.py src/gnn/data/sample.py src/gnn/core/__init__.py tests/core/test_graph_batch_graph_classification.py
git commit -m "feat: add graph classification batch metadata"
```

### Task 2: Upgrade Loader to Collate SampleRecord Inputs

**Files:**
- Modify: `src/gnn/data/loader.py`
- Modify: `src/gnn/data/dataset.py`
- Modify: `src/gnn/data/__init__.py`
- Test: `tests/data/test_graph_classification_loader.py`

**Step 1: Write the failing loader test**

```python
import torch

from gnn import Graph
from gnn.data.dataset import ListDataset
from gnn.data.loader import Loader
from gnn.data.sample import SampleRecord
from gnn.data.sampler import FullGraphSampler


def test_loader_collates_graph_samples_with_metadata_labels():
    samples = [
        SampleRecord(
            graph=Graph.homo(edge_index=torch.tensor([[0], [1]]), x=torch.randn(2, 4), y=torch.tensor([1])),
            metadata={"label": 1},
            sample_id="a",
        ),
        SampleRecord(
            graph=Graph.homo(edge_index=torch.tensor([[0], [1]]), x=torch.randn(2, 4), y=torch.tensor([0])),
            metadata={"label": 0},
            sample_id="b",
        ),
    ]

    dataset = ListDataset(samples)
    loader = Loader(dataset=dataset, sampler=FullGraphSampler(), batch_size=2, label_source="metadata", label_key="label")
    batch = next(iter(loader))

    assert batch.num_graphs == 2
    assert torch.equal(batch.labels, torch.tensor([1, 0]))
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/data/test_graph_classification_loader.py -v`
Expected: `FAIL` because `Loader` does not yet understand `SampleRecord` or graph classification labels.

**Step 3: Write minimal implementation**

Update `ListDataset` so it can carry either raw graphs or `SampleRecord` objects without breaking old behavior. Update `Loader` to:

- detect `SampleRecord`
- build `GraphBatch.from_samples(...)`
- keep old node-classification behavior working

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/data/test_graph_classification_loader.py -v`
Expected: `PASS`

**Step 5: Commit**

```bash
git add src/gnn/data/loader.py src/gnn/data/dataset.py src/gnn/data/__init__.py tests/data/test_graph_classification_loader.py
git commit -m "feat: collate graph classification samples"
```

### Task 3: Add Graph-Level Readout Utilities

**Files:**
- Create: `src/gnn/nn/readout.py`
- Modify: `src/gnn/nn/__init__.py`
- Test: `tests/nn/test_readout.py`

**Step 1: Write the failing readout test**

```python
import torch

from gnn.nn.readout import global_mean_pool, global_sum_pool, global_max_pool


def test_global_mean_pool_reduces_node_embeddings_per_graph():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [10.0, 20.0]])
    graph_index = torch.tensor([0, 0, 1])

    out = global_mean_pool(x, graph_index)

    assert torch.equal(out, torch.tensor([[2.0, 3.0], [10.0, 20.0]]))
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/nn/test_readout.py -v`
Expected: `FAIL` because the readout helpers do not exist yet.

**Step 3: Write minimal implementation**

Implement:

- `global_mean_pool`
- `global_sum_pool`
- `global_max_pool`

Keep the implementation explicit and simple.

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/nn/test_readout.py -v`
Expected: `PASS`

**Step 5: Commit**

```bash
git add src/gnn/nn/readout.py src/gnn/nn/__init__.py tests/nn/test_readout.py
git commit -m "feat: add graph readout utilities"
```

### Task 4: Add GraphClassificationTask for Graph and Metadata Labels

**Files:**
- Modify: `src/gnn/train/tasks.py`
- Modify: `src/gnn/train/__init__.py`
- Test: `tests/train/test_graph_classification_task.py`

**Step 1: Write the failing task test**

```python
import torch

from gnn.train.tasks import GraphClassificationTask


class FakeBatch:
    labels = torch.tensor([1, 0])
    metadata = [{"label": 1}, {"label": 0}]


def test_graph_classification_task_uses_batch_labels():
    task = GraphClassificationTask(target="y", label_source="graph")
    logits = torch.randn(2, 2, requires_grad=True)

    loss = task.loss(FakeBatch(), logits, stage="train")

    assert loss.ndim == 0


def test_graph_classification_task_uses_metadata_labels():
    task = GraphClassificationTask(target="label", label_source="metadata")
    logits = torch.randn(2, 2, requires_grad=True)

    loss = task.loss(FakeBatch(), logits, stage="train")

    assert loss.ndim == 0
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/train/test_graph_classification_task.py -v`
Expected: `FAIL` because `GraphClassificationTask` does not exist yet.

**Step 3: Write minimal implementation**

Extend `src/gnn/train/tasks.py` with `GraphClassificationTask` that:

- reads labels from `batch.labels` when `label_source="graph"`
- reads labels from `batch.metadata` when `label_source="metadata"`
- optionally supports `auto`

Do not expand trainer logic yet.

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/train/test_graph_classification_task.py -v`
Expected: `PASS`

**Step 5: Commit**

```bash
git add src/gnn/train/tasks.py src/gnn/train/__init__.py tests/train/test_graph_classification_task.py
git commit -m "feat: add graph classification task"
```

### Task 5: Add Graph Classification Model Path and Trainer Support

**Files:**
- Modify: `src/gnn/train/trainer.py`
- Test: `tests/train/test_graph_classification_trainer.py`

**Step 1: Write the failing trainer test**

```python
import torch
from torch import nn

from gnn.train.tasks import GraphClassificationTask
from gnn.train.trainer import Trainer


class FakeBatch:
    labels = torch.tensor([1, 0])
    metadata = [{"label": 1}, {"label": 0}]


class TinyGraphClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, batch):
        return self.linear(torch.randn(2, 4))


def test_trainer_runs_graph_classification_epoch():
    trainer = Trainer(
        model=TinyGraphClassifier(),
        task=GraphClassificationTask(target="y", label_source="graph"),
        optimizer=torch.optim.Adam,
        lr=1e-2,
        max_epochs=1,
    )

    history = trainer.fit(FakeBatch())

    assert history["epochs"] == 1
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/train/test_graph_classification_trainer.py -v`
Expected: `FAIL` until the trainer can handle graph-classification style batches cleanly.

**Step 3: Write minimal implementation**

Adjust `Trainer` only as much as necessary so it can train against `GraphClassificationTask` batches without regressing node classification.

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/train/test_graph_classification_trainer.py -v`
Expected: `PASS`

**Step 5: Commit**

```bash
git add src/gnn/train/trainer.py tests/train/test_graph_classification_trainer.py
git commit -m "feat: support graph classification in trainer"
```

### Task 6: Add Minimal Subgraph Graph-Classification Sampling

**Files:**
- Modify: `src/gnn/data/sampler.py`
- Test: `tests/data/test_subgraph_sampler.py`

**Step 1: Write the failing sampler test**

```python
import torch

from gnn import Graph
from gnn.data.sample import SampleRecord
from gnn.data.sampler import NodeSeedSubgraphSampler


def test_node_seed_subgraph_sampler_returns_sample_record():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        x=torch.randn(3, 4),
        y=torch.tensor([1]),
    )
    sampler = NodeSeedSubgraphSampler()

    sample = sampler.sample((graph, {"seed": 1, "label": 1, "sample_id": "s1"}))

    assert isinstance(sample, SampleRecord)
    assert sample.sample_id == "s1"
    assert sample.metadata["label"] == 1
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/data/test_subgraph_sampler.py -v`
Expected: `FAIL` because the subgraph sampler does not exist yet.

**Step 3: Write minimal implementation**

Add a small `NodeSeedSubgraphSampler` that produces `SampleRecord` objects from `(graph, metadata)` pairs. Keep it intentionally minimal for phase 2.

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/data/test_subgraph_sampler.py -v`
Expected: `PASS`

**Step 5: Commit**

```bash
git add src/gnn/data/sampler.py tests/data/test_subgraph_sampler.py
git commit -m "feat: add minimal subgraph graph-classification sampler"
```

### Task 7: Add End-to-End Many-Small-Graph Graph Classification

**Files:**
- Create: `tests/integration/test_graph_classification_many_graphs.py`
- Create: `examples/homo/graph_classification.py`

**Step 1: Write the failing integration test**

```python
import torch
from torch import nn

from gnn import Graph
from gnn.core.batch import GraphBatch
from gnn.data.sample import SampleRecord
from gnn.nn.readout import global_mean_pool
from gnn.train.tasks import GraphClassificationTask
from gnn.train.trainer import Trainer


class TinyGraphClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(4, 4)
        self.head = nn.Linear(4, 2)

    def forward(self, batch):
        x = torch.cat([graph.x for graph in batch.graphs], dim=0)
        node_repr = self.encoder(x)
        graph_repr = global_mean_pool(node_repr, batch.graph_index)
        return self.head(graph_repr)


def test_many_graph_graph_classification_runs():
    samples = [
        SampleRecord(graph=Graph.homo(edge_index=torch.tensor([[0], [1]]), x=torch.randn(2, 4), y=torch.tensor([1])), metadata={}, sample_id="g1"),
        SampleRecord(graph=Graph.homo(edge_index=torch.tensor([[0], [1]]), x=torch.randn(2, 4), y=torch.tensor([0])), metadata={}, sample_id="g2"),
    ]
    batch = GraphBatch.from_samples(samples, label_key="y", label_source="graph")
    trainer = Trainer(
        model=TinyGraphClassifier(),
        task=GraphClassificationTask(target="y", label_source="graph"),
        optimizer=torch.optim.Adam,
        lr=1e-2,
        max_epochs=1,
    )

    result = trainer.fit(batch)

    assert result["epochs"] == 1
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/integration/test_graph_classification_many_graphs.py -v`
Expected: `FAIL` until the graph classification path is fully wired.

**Step 3: Write minimal implementation and example**

Add the smallest example that mirrors the tested path.

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/integration/test_graph_classification_many_graphs.py -v`
Expected: `PASS`

**Step 5: Commit**

```bash
git add tests/integration/test_graph_classification_many_graphs.py examples/homo/graph_classification.py
git commit -m "feat: add many-graph classification flow"
```

### Task 8: Add End-to-End Sampled Subgraph Graph Classification

**Files:**
- Create: `tests/integration/test_graph_classification_subgraph_samples.py`
- Create: `examples/hetero/graph_classification.py`

**Step 1: Write the failing integration test**

```python
import torch
from torch import nn

from gnn import Graph
from gnn.data.sample import SampleRecord
from gnn.nn.readout import global_mean_pool
from gnn.train.tasks import GraphClassificationTask
from gnn.train.trainer import Trainer


class TinyGraphClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(4, 4)
        self.head = nn.Linear(4, 2)

    def forward(self, batch):
        x = torch.cat([graph.x for graph in batch.graphs], dim=0)
        node_repr = self.encoder(x)
        graph_repr = global_mean_pool(node_repr, batch.graph_index)
        return self.head(graph_repr)


def test_subgraph_sample_graph_classification_runs():
    source_graph = Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        x=torch.randn(3, 4),
        y=torch.tensor([1]),
    )
    samples = [
        SampleRecord(graph=source_graph, metadata={"label": 1}, sample_id="s1", source_graph_id="root", subgraph_seed=1),
        SampleRecord(graph=source_graph, metadata={"label": 0}, sample_id="s2", source_graph_id="root", subgraph_seed=2),
    ]
    batch = GraphBatch.from_samples(samples, label_key="label", label_source="metadata")
    trainer = Trainer(
        model=TinyGraphClassifier(),
        task=GraphClassificationTask(target="label", label_source="metadata"),
        optimizer=torch.optim.Adam,
        lr=1e-2,
        max_epochs=1,
    )

    result = trainer.fit(batch)

    assert result["epochs"] == 1
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/integration/test_graph_classification_subgraph_samples.py -v`
Expected: `FAIL` until metadata-label graph classification works end to end.

**Step 3: Write minimal implementation and example**

Add the smallest example that mirrors sampled-subgraph graph classification.

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/integration/test_graph_classification_subgraph_samples.py -v`
Expected: `PASS`

**Step 5: Commit**

```bash
git add tests/integration/test_graph_classification_subgraph_samples.py examples/hetero/graph_classification.py
git commit -m "feat: add sampled-subgraph graph classification flow"
```

### Task 9: Regression and Docs Verification

**Files:**
- Modify: `README.md`
- Modify: `docs/quickstart.md`
- Modify: `docs/core-concepts.md`
- Modify: `docs/migration-guide.md`

**Step 1: Verify node classification still works**

Run: `python -m pytest tests/train/test_tasks.py tests/train/test_trainer.py tests/integration/test_end_to_end_homo.py -v`
Expected: all pass, proving graph classification additions did not regress the existing node classification path.

**Step 2: Verify graph classification coverage**

Run: `python -m pytest tests/core tests/data tests/nn tests/train tests/integration -v`
Expected: all pass.

**Step 3: Run repository-level verification**

Run:

```bash
python -m pytest -v
python -m ruff check .
python -m mypy src
python examples/homo/graph_classification.py
python examples/hetero/graph_classification.py
```

Expected:

- tests pass
- ruff passes
- mypy passes
- examples run successfully

**Step 4: Update docs to describe graph classification**

Make sure the docs explain:

- `GraphClassificationTask`
- label sources
- `GraphBatch.labels`
- the two supported graph classification entry paths

**Step 5: Commit**

```bash
git add README.md docs/quickstart.md docs/core-concepts.md docs/migration-guide.md
git commit -m "docs: document graph classification training"
```

## Notes

- Do not broaden this phase into link prediction or temporal prediction.
- Keep all new behavior driven by tests first.
- Preserve node classification APIs and examples unless a regression test justifies a change.
