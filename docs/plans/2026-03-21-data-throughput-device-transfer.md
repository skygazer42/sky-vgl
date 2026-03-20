# Data Throughput And Device Transfer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add worker-backed loading controls to `Loader` and explicit model-plus-batch device placement to `Trainer` without changing current default behavior or task contracts.

**Architecture:** Extend the existing `Loader` and `Trainer` instead of introducing parallel abstractions. Give VGL-owned graph and batch types stable non-mutating `.to()` and `pin_memory()` methods, keep final VGL batch assembly centralized in the main process, and let `Trainer(device=...)` orchestrate one shared batch preparation path across train, validation, test, sanity validation, and mid-epoch validation.

**Tech Stack:** Python 3.11+, PyTorch, pytest, ruff, mypy

**Execution Rules:** Use `@test-driven-development` for every code task, keep commits small, and use `@verification-before-completion` before claiming the phase is done.

---

### Task 1: Add Store And Graph Transfer Primitives

**Files:**
- Create: `tests/core/test_graph_transfer.py`
- Modify: `vgl/graph/stores.py`
- Modify: `vgl/graph/graph.py`
- Modify: `vgl/graph/view.py`

**Step 1: Write the failing test**

```python
import torch

from vgl.graph import Graph
from vgl.graph.stores import EdgeStore, NodeStore
from vgl.graph.view import GraphView


def test_node_store_to_returns_new_store():
    store = NodeStore("node", {"x": torch.randn(2, 3), "name": "demo"})
    moved = store.to(device="cpu")
    assert moved is not store
    assert moved.x.device.type == "cpu"
    assert moved.data["name"] == "demo"


def test_edge_store_pin_memory_pins_tensor_fields_only():
    store = EdgeStore(("node", "to", "node"), {"edge_index": torch.tensor([[0], [1]]), "kind": "edge"})
    pinned = store.pin_memory()
    assert pinned is not store
    assert pinned.edge_index.is_pinned()
    assert pinned.data["kind"] == "edge"


def test_graph_to_moves_homo_hetero_and_temporal_tensors():
    homo = Graph.homo(edge_index=torch.tensor([[0, 1], [1, 0]]), x=torch.randn(2, 4), y=torch.tensor([0, 1]))
    hetero = Graph.temporal(
        nodes={"node": {"x": torch.randn(3, 4)}, "paper": {"x": torch.randn(2, 5)}},
        edges={
            ("node", "to", "node"): {"edge_index": torch.tensor([[0, 1], [1, 2]]), "timestamp": torch.tensor([1, 2])},
            ("paper", "cites", "paper"): {"edge_index": torch.tensor([[0], [1]]), "timestamp": torch.tensor([4])},
        },
        time_attr="timestamp",
    )
    moved_homo = homo.to(device="cpu")
    moved_hetero = hetero.to(device="cpu")
    assert moved_homo.x.device.type == "cpu"
    assert moved_homo.edge_index.device.type == "cpu"
    assert moved_hetero.nodes["node"].x.device.type == "cpu"
    assert moved_hetero.edges[("paper", "cites", "paper")].edge_index.device.type == "cpu"


def test_graph_view_to_moves_visible_tensors_without_mutating_base():
    base = Graph.temporal(
        nodes={"node": {"x": torch.randn(3, 4)}},
        edges={("node", "to", "node"): {"edge_index": torch.tensor([[0, 1], [1, 2]]), "timestamp": torch.tensor([1, 3])}},
        time_attr="timestamp",
    )
    view = base.snapshot(1)
    moved = view.to(device="cpu")
    assert isinstance(moved, GraphView)
    assert moved is not view
    assert moved.x.device.type == "cpu"
    assert moved.base is view.base
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/core/test_graph_transfer.py -v`
Expected: `FAIL` because stores, graphs, and graph views do not yet implement `.to()` or `pin_memory()`.

**Step 3: Write minimal implementation**

Add non-mutating transfer helpers to `vgl/graph/stores.py`:

```python
def _move_mapping(data, *, device=None, dtype=None, non_blocking=False, pin_memory=False):
    moved = {}
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.pin_memory() if pin_memory else value.to(
                device=device,
                dtype=dtype,
                non_blocking=non_blocking,
            )
        else:
            moved[key] = value
    return moved
```

Then add:

```python
def to(self, device=None, dtype=None, non_blocking=False): ...
def pin_memory(self): ...
```

to both `NodeStore` and `EdgeStore`.

Update `vgl/graph/graph.py` and `vgl/graph/view.py` so they rebuild their node and edge dictionaries through each store's `.to()` and `pin_memory()` methods while preserving `schema` and `base`.

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/core/test_graph_transfer.py -v`
Expected: `PASS`

**Step 5: Commit**

```bash
git add tests/core/test_graph_transfer.py vgl/graph/stores.py vgl/graph/graph.py vgl/graph/view.py
git commit -m "feat: add graph transfer primitives"
```

### Task 2: Add Transfer Methods For VGL Batch Types

**Files:**
- Create: `tests/core/test_batch_transfer.py`
- Modify: `vgl/graph/batch.py`

**Step 1: Write the failing test**

```python
import torch

from vgl.dataloading.records import LinkPredictionRecord, SampleRecord, TemporalEventRecord
from vgl.graph import Graph
from vgl.graph.batch import GraphBatch, LinkPredictionBatch, NodeBatch, TemporalEventBatch


def _graph():
    return Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        x=torch.randn(2, 4),
        y=torch.tensor([1]),
    )


def test_graph_batch_to_moves_graph_index_graph_ptr_and_labels():
    batch = GraphBatch.from_samples(
        [SampleRecord(graph=_graph(), metadata={"label": 1}), SampleRecord(graph=_graph(), metadata={"label": 0})],
        label_key="label",
        label_source="metadata",
    )
    moved = batch.to(device="cpu")
    assert moved.graph_index.device.type == "cpu"
    assert moved.graph_ptr.device.type == "cpu"
    assert moved.labels.device.type == "cpu"
    assert all(graph.x.device.type == "cpu" for graph in moved.graphs)


def test_node_batch_to_moves_graph_and_seed_index():
    graph = _graph()
    batch = NodeBatch.from_samples(
        [
            SampleRecord(graph=graph, metadata={"node_type": "node"}, subgraph_seed=0),
            SampleRecord(graph=graph, metadata={"node_type": "node"}, subgraph_seed=1),
        ]
    )
    moved = batch.to(device="cpu")
    assert moved.graph.x.device.type == "cpu"
    assert moved.seed_index.device.type == "cpu"


def test_link_prediction_batch_to_moves_indices_labels_and_query_fields():
    graph = _graph()
    batch = LinkPredictionBatch.from_records(
        [
            LinkPredictionRecord(graph=graph, src_index=0, dst_index=1, label=1, query_id="q"),
            LinkPredictionRecord(graph=graph, src_index=1, dst_index=0, label=0, query_id="q"),
        ]
    )
    moved = batch.to(device="cpu")
    assert moved.graph.x.device.type == "cpu"
    assert moved.src_index.device.type == "cpu"
    assert moved.dst_index.device.type == "cpu"
    assert moved.labels.device.type == "cpu"
    assert moved.query_index.device.type == "cpu"
    assert moved.filter_mask.device.type == "cpu"


def test_temporal_event_batch_to_moves_timestamps_labels_and_features():
    graph = Graph.temporal(
        nodes={"node": {"x": torch.randn(3, 4)}},
        edges={("node", "interacts", "node"): {"edge_index": torch.tensor([[0, 1], [1, 2]]), "timestamp": torch.tensor([1, 4])}},
        time_attr="timestamp",
    )
    batch = TemporalEventBatch.from_records(
        [
            TemporalEventRecord(graph=graph, src_index=0, dst_index=1, timestamp=1, label=1, event_features=torch.tensor([1.0, 0.0])),
            TemporalEventRecord(graph=graph, src_index=1, dst_index=2, timestamp=4, label=0, event_features=torch.tensor([0.0, 1.0])),
        ]
    )
    moved = batch.to(device="cpu")
    assert moved.graph.x.device.type == "cpu"
    assert moved.src_index.device.type == "cpu"
    assert moved.dst_index.device.type == "cpu"
    assert moved.timestamp.device.type == "cpu"
    assert moved.labels.device.type == "cpu"
    assert moved.event_features.device.type == "cpu"
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/core/test_batch_transfer.py -v`
Expected: `FAIL` because VGL batch types do not yet implement `.to()` or `pin_memory()`.

**Step 3: Write minimal implementation**

Add non-mutating `.to()` and `pin_memory()` methods to:

- `GraphBatch`
- `NodeBatch`
- `LinkPredictionBatch`
- `TemporalEventBatch`

Each implementation should move or pin:

- the embedded graph or graphs
- index tensors
- labels
- timestamps
- optional tensors such as `graph_ptr`, `query_index`, `filter_mask`, and `event_features`

Use this pattern for `GraphBatch`:

```python
def to(self, device=None, dtype=None, non_blocking=False):
    return GraphBatch(
        graphs=[graph.to(device=device, dtype=dtype, non_blocking=non_blocking) for graph in self.graphs],
        graph_index=self.graph_index.to(device=device, non_blocking=non_blocking),
        graph_ptr=None if self.graph_ptr is None else self.graph_ptr.to(device=device, non_blocking=non_blocking),
        labels=None if self.labels is None else self.labels.to(device=device, non_blocking=non_blocking),
        metadata=self.metadata,
    )
```

Metadata should be preserved by reference.

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/core/test_batch_transfer.py -v`
Expected: `PASS`

**Step 5: Commit**

```bash
git add tests/core/test_batch_transfer.py vgl/graph/batch.py
git commit -m "feat: add batch transfer methods"
```

### Task 3: Add Throughput Controls To `Loader`

**Files:**
- Modify: `vgl/dataloading/loader.py`
- Modify: `tests/data/test_loader.py`
- Modify: `tests/data/test_node_neighbor_sampler.py`
- Modify: `tests/data/test_link_prediction_loader.py`

**Step 1: Write the failing test**

Add coverage in `tests/data/test_loader.py`:

```python
import pytest
import torch

from vgl.data.dataset import ListDataset
from vgl.dataloading.loader import Loader
from vgl.dataloading.sampler import FullGraphSampler
from vgl.graph import Graph


class IterableOnlyDataset:
    def __iter__(self):
        yield from [1, 2, 3]


def _graph(seed):
    torch.manual_seed(seed)
    return Graph.homo(edge_index=torch.tensor([[0], [1]]), x=torch.randn(2, 4), y=torch.tensor([seed % 2]))


def test_loader_rejects_workers_for_iterable_only_dataset():
    with pytest.raises(TypeError, match="map-style"):
        list(Loader(dataset=IterableOnlyDataset(), sampler=FullGraphSampler(), batch_size=1, num_workers=1))


def test_loader_rejects_prefetch_factor_without_workers():
    with pytest.raises(ValueError, match="prefetch_factor"):
        Loader(dataset=ListDataset([_graph(1)]), sampler=FullGraphSampler(), batch_size=1, prefetch_factor=2)


def test_loader_rejects_persistent_workers_without_workers():
    with pytest.raises(ValueError, match="persistent_workers"):
        Loader(dataset=ListDataset([_graph(1)]), sampler=FullGraphSampler(), batch_size=1, persistent_workers=True)


def test_loader_num_workers_one_supports_map_style_dataset():
    loader = Loader(dataset=ListDataset([_graph(1), _graph(2)]), sampler=FullGraphSampler(), batch_size=2, num_workers=1)
    batch = next(iter(loader))
    assert batch.num_graphs == 2
```

Add one pinning assertion to an existing loader-oriented test:

```python
def test_loader_pin_memory_pins_batch_tensors():
    loader = Loader(dataset=ListDataset([_graph(1), _graph(2)]), sampler=FullGraphSampler(), batch_size=2, pin_memory=True)
    batch = next(iter(loader))
    assert batch.graph_index.is_pinned()
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/data/test_loader.py tests/data/test_node_neighbor_sampler.py tests/data/test_link_prediction_loader.py -v`
Expected: `FAIL` because `Loader` does not yet accept throughput arguments or pin assembled batches.

**Step 3: Write minimal implementation**

Update `vgl/dataloading/loader.py` to accept:

```python
num_workers=0,
pin_memory=False,
prefetch_factor=None,
persistent_workers=False,
```

Add validation:

```python
if prefetch_factor is not None and num_workers == 0:
    raise ValueError("prefetch_factor requires num_workers > 0")
if persistent_workers and num_workers == 0:
    raise ValueError("persistent_workers requires num_workers > 0")
```

Add private helpers:

```python
class _SampledDataset:
    def __init__(self, dataset, sampler):
        self.dataset = dataset
        self.sampler = sampler

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.sampler.sample(self.dataset[index])


def _identity_collate(batch):
    return batch
```

Retain the current single-process iterator for `num_workers=0`.

For `num_workers>0`, require a map-style dataset and use an internal `torch.utils.data.DataLoader`:

```python
worker_loader = torch.utils.data.DataLoader(
    _SampledDataset(self.dataset, self.sampler),
    batch_size=self.batch_size,
    shuffle=False,
    num_workers=self.num_workers,
    pin_memory=False,
    prefetch_factor=self.prefetch_factor,
    persistent_workers=self.persistent_workers,
    collate_fn=_identity_collate,
)
```

Flatten worker outputs in the main process, call the existing `_build_batch()`, then apply `batch.pin_memory()` when requested.

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/data/test_loader.py tests/data/test_node_neighbor_sampler.py tests/data/test_link_prediction_loader.py -v`
Expected: `PASS`

**Step 5: Commit**

```bash
git add vgl/dataloading/loader.py tests/data/test_loader.py tests/data/test_node_neighbor_sampler.py tests/data/test_link_prediction_loader.py
git commit -m "feat: add loader throughput controls"
```

### Task 4: Add Explicit Device Placement To `Trainer`

**Files:**
- Create: `tests/train/test_trainer_device_transfer.py`
- Modify: `tests/train/test_mixed_precision.py`
- Modify: `vgl/engine/trainer.py`

**Step 1: Write the failing test**

Create `tests/train/test_trainer_device_transfer.py`:

```python
import pytest
import torch
from torch import nn

from vgl.engine import Trainer
from vgl.graph import Graph
from vgl.graph.batch import NodeBatch
from vgl.dataloading.records import SampleRecord
from vgl.train.task import Task


class ToyBatch:
    def __init__(self, target):
        self.target = torch.tensor([target], dtype=torch.float32)


class ToyTask(Task):
    def loss(self, batch, predictions, stage):
        del stage
        return ((predictions - batch.target) ** 2).mean()

    def targets(self, batch, stage):
        del stage
        return batch.target


class RecordingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([0.0]))
        self.seen_batch_devices = []

    def forward(self, batch):
        self.seen_batch_devices.append(batch.target.device.type)
        return self.weight.repeat(batch.target.size(0))


class NodeBatchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)
        self.seen_graph_device = None
        self.seen_seed_device = None

    def forward(self, batch):
        self.seen_graph_device = batch.graph.x.device.type
        self.seen_seed_device = batch.seed_index.device.type
        return self.linear(batch.graph.x)[batch.seed_index]


class NodeTask(Task):
    def loss(self, batch, predictions, stage):
        del batch, stage
        return predictions.sum() * 0.0

    def targets(self, batch, stage):
        del stage
        return torch.zeros(batch.seed_index.size(0), dtype=torch.long)


def test_trainer_device_moves_model_and_standard_batch():
    model = RecordingModel()
    trainer = Trainer(model=model, task=ToyTask(), optimizer=torch.optim.SGD, lr=0.1, max_epochs=1, device="cpu")
    trainer.fit([ToyBatch(1.0)])
    assert next(trainer.model.parameters()).device.type == "cpu"
    assert model.seen_batch_devices == ["cpu"]


def test_trainer_device_moves_vgl_node_batch():
    graph = Graph.homo(edge_index=torch.tensor([[0, 1], [1, 0]]), x=torch.randn(2, 4), y=torch.tensor([0, 1]))
    batch = NodeBatch.from_samples([SampleRecord(graph=graph, metadata={"node_type": "node"}, subgraph_seed=0)])
    model = NodeBatchModel()
    trainer = Trainer(model=model, task=NodeTask(), optimizer=torch.optim.SGD, lr=0.1, max_epochs=1, device="cpu")
    trainer.fit([batch])
    assert model.seen_graph_device == "cpu"
    assert model.seen_seed_device == "cpu"


def test_trainer_move_batch_to_device_false_skips_automatic_batch_movement():
    model = RecordingModel()
    trainer = Trainer(
        model=model,
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=0.1,
        max_epochs=1,
        device="cpu",
        move_batch_to_device=False,
    )
    trainer.fit([ToyBatch(1.0)])
    assert next(trainer.model.parameters()).device.type == "cpu"


def test_trainer_rejects_unsupported_batch_for_auto_move():
    trainer = Trainer(model=RecordingModel(), task=ToyTask(), optimizer=torch.optim.SGD, lr=0.1, max_epochs=1, device="cpu")
    with pytest.raises(TypeError, match="automatic batch movement"):
        trainer.fit([object()])
```

Update `tests/train/test_mixed_precision.py`:

```python
def test_trainer_uses_configured_device_for_autocast(monkeypatch):
    calls = []

    @contextmanager
    def fake_autocast(device_type, dtype=None, enabled=True):
        calls.append((device_type, dtype, enabled))
        yield

    monkeypatch.setattr(torch, "autocast", fake_autocast)
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=1,
        precision="bf16-mixed",
        device="cpu",
    )
    trainer.fit([ToyBatch(1.0)])
    assert calls == [("cpu", torch.bfloat16, True)]
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/train/test_trainer_device_transfer.py tests/train/test_mixed_precision.py -v`
Expected: `FAIL` because `Trainer` does not yet expose explicit device-placement controls.

**Step 3: Write minimal implementation**

Update `vgl/engine/trainer.py` constructor to accept:

```python
device=None,
move_batch_to_device=True,
non_blocking=None,
```

Store:

```python
self.device = None if device is None else torch.device(device)
self.move_batch_to_device = bool(move_batch_to_device)
self.non_blocking = non_blocking
if self.device is not None:
    self.model = self.model.to(self.device)
```

Add batch preparation helpers:

```python
def _resolved_device_type(self):
    if self.device is not None:
        return self.device.type
    return self._model_device_type()

def _resolved_non_blocking(self, batch):
    if self.non_blocking is not None:
        return bool(self.non_blocking)
    if self.device is None or self.device.type != "cuda":
        return False
    return self._is_pinned_batch(batch)

def _move_batch_to_device(self, batch):
    if self.device is None or not self.move_batch_to_device:
        return batch
    non_blocking = self._resolved_non_blocking(batch)
    if isinstance(batch, torch.Tensor):
        return batch.to(device=self.device, non_blocking=non_blocking)
    if isinstance(batch, dict):
        return {key: self._move_batch_to_device(value) for key, value in batch.items()}
    if isinstance(batch, list):
        return [self._move_batch_to_device(value) for value in batch]
    if isinstance(batch, tuple):
        return tuple(self._move_batch_to_device(value) for value in batch)
    if hasattr(batch, "to"):
        return batch.to(device=self.device, non_blocking=non_blocking)
    raise TypeError("Trainer automatic batch movement supports VGL graph/batch types, tensors, dicts, lists, and tuples")
```

Route all train and eval paths through:

```python
def _prepare_batch(self, batch):
    return self._move_batch_to_device(batch)
```

Update mixed-precision validation to use `_resolved_device_type()` instead of only the model's current placement.

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/train/test_trainer_device_transfer.py tests/train/test_mixed_precision.py -v`
Expected: `PASS`

**Step 5: Commit**

```bash
git add vgl/engine/trainer.py tests/train/test_trainer_device_transfer.py tests/train/test_mixed_precision.py
git commit -m "feat: add trainer device placement controls"
```

### Task 5: Surface The New Controls In Logging And Docs

**Files:**
- Modify: `vgl/engine/trainer.py`
- Modify: `vgl/engine/logging.py`
- Modify: `tests/train/test_trainer_plus.py`
- Modify: `tests/train/test_logging.py`
- Modify: `README.md`
- Modify: `docs/quickstart.md`

**Step 1: Write the failing test**

Update `tests/train/test_trainer_plus.py`:

```python
def test_fit_start_records_include_device_controls():
    logger = RecordingLogger()
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=0.1,
        max_epochs=1,
        device="cpu",
        move_batch_to_device=True,
        non_blocking=False,
        loggers=[logger],
        enable_console_logging=False,
    )
    trainer.fit([ToyBatch(1.0)])
    fit_start = logger.records[0]
    assert fit_start["device"] == "cpu"
    assert fit_start["move_batch_to_device"] is True
    assert fit_start["non_blocking"] is False
```

Update `tests/train/test_logging.py`:

```python
def test_console_logger_run_banner_includes_device_controls(capsys):
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=0.1,
        max_epochs=1,
        loggers=[ConsoleLogger(enable_progress_bar=False)],
        enable_console_logging=False,
        device="cpu",
        move_batch_to_device=True,
        non_blocking=False,
    )
    trainer.fit([ToyBatch(1.0)])
    output = capsys.readouterr().out
    assert "device=cpu" in output
    assert "move_batch_to_device=True" in output
    assert "non_blocking=False" in output
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/train/test_trainer_plus.py tests/train/test_logging.py -v`
Expected: `FAIL` because run metadata and console output do not yet expose the new controls.

**Step 3: Write minimal implementation**

Update `vgl/engine/trainer.py` run records so they include:

```python
record["device"] = None if self.device is None else str(self.device)
record["move_batch_to_device"] = self.move_batch_to_device
record["non_blocking"] = self.non_blocking
```

Update `vgl/engine/logging.py` detailed run banner so it can show:

- `device=<...>`
- `move_batch_to_device=<...>`
- `non_blocking=<...>`

Update documentation with one short snippet each:

- `README.md` for `Loader(..., num_workers=..., pin_memory=...)`
- `docs/quickstart.md` for `Trainer(device="cuda")`

Do not rewrite all examples in this task.

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/train/test_trainer_plus.py tests/train/test_logging.py -v`
Expected: `PASS`

**Step 5: Commit**

```bash
git add vgl/engine/trainer.py vgl/engine/logging.py tests/train/test_trainer_plus.py tests/train/test_logging.py README.md docs/quickstart.md
git commit -m "docs: surface throughput and device controls"
```

### Task 6: Run Full Regression Verification

**Files:**
- Verify only

**Step 1: Run the focused feature tests**

Run: `python -m pytest tests/core/test_graph_transfer.py tests/core/test_batch_transfer.py tests/data/test_loader.py tests/data/test_node_neighbor_sampler.py tests/data/test_link_prediction_loader.py tests/train/test_trainer_device_transfer.py tests/train/test_mixed_precision.py tests/train/test_trainer_plus.py tests/train/test_logging.py -v`
Expected: `PASS`

**Step 2: Run the full test suite**

Run: `python -m pytest -q`
Expected: `PASS`

**Step 3: Run lint and type checking**

Run: `python -m ruff check .`
Expected: `All checks passed`

Run: `python -m mypy vgl`
Expected: `Success: no issues found`

**Step 4: Run example smoke checks**

Run: `python examples/homo/node_classification.py`
Expected: prints a training result dictionary

Run: `python examples/homo/graph_classification.py`
Expected: prints a training result dictionary

Run: `python examples/homo/link_prediction.py`
Expected: prints a training result dictionary

Run: `python examples/temporal/event_prediction.py`
Expected: prints a training result dictionary

**Step 5: Commit final polish only if verification requires follow-up edits**

```bash
git add README.md docs/quickstart.md vgl tests
git commit -m "chore: verify throughput and device transfer flow"
```

Do not create an empty commit. If verification passes without any follow-up edits, stop after recording the evidence.
