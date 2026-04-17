# Migration Guide

## Package Layout Migration

The preferred import layout is now domain-based:

- `vgl.graph` for `Graph`, `GraphBatch`, `GraphView`, and `GraphSchema`
- `vgl.dataloading` for `DataLoader`, datasets, samplers, and sample records
- `vgl.tasks` for task definitions
- `vgl.engine` for `Trainer`, callbacks, checkpoints, and `TrainingHistory`
- `vgl.metrics` for metric implementations and `build_metric`
- `vgl.transforms` for graph transforms

The older `vgl.core`, `vgl.data`, and `vgl.train` modules still work as compatibility layers, but new code should avoid them.

On first import, each legacy namespace now emits a single `FutureWarning` that points back to this guide. The warning is intentionally brief: it tells you which modern module to prefer, then uses this page for concrete rewrite examples.

Common import rewrites:

- `from vgl.data.loader import Loader` -> `from vgl.dataloading import DataLoader`
- `from vgl.data import Sampler` -> `from vgl.dataloading import Sampler`
- `from vgl.data.dataset import ListDataset` -> `from vgl.dataloading import ListDataset`
- `from vgl.data.sample import SampleRecord` -> `from vgl.dataloading import SampleRecord`
- `from vgl.train.tasks import NodeClassificationTask` -> `from vgl.tasks import NodeClassificationTask`
- `from vgl.train.trainer import Trainer` -> `from vgl.engine import Trainer`
- `from vgl.train.evaluator import Evaluator` -> `from vgl.engine import Evaluator`
- `from vgl.train.metrics import build_metric` -> `from vgl.metrics import build_metric`
- `from vgl.train.checkpoints import save_checkpoint` -> `from vgl.engine import save_checkpoint`
- `from vgl.train.history import TrainingHistory` -> `from vgl.engine import TrainingHistory`
- `from vgl.core.graph import Graph` -> `from vgl.graph import Graph`
- `from vgl.core import SchemaError` -> `from vgl.graph import SchemaError`

## From PyG

- `Data(x=..., edge_index=..., y=...)` maps to `Graph.homo(edge_index=..., x=..., y=...)`
- `Graph.from_pyg(data)` imports PyG-style graph data
- `graph.to_pyg()` exports back to a PyG-style `Data` object
- many-graph graph classification maps cleanly to `ListDataset + DataLoader + GraphClassificationTask`

## From DGL

- `graph.ndata[...]` style access remains available on homogeneous graphs
- `Graph.from_dgl(dgl_graph)` is the graph-only import path for DGL homogeneous graphs and heterographs
- DGL message-flow blocks use `Block.from_dgl(dgl_block)` or `vgl.compat.block_from_dgl(dgl_block)` instead of `Graph.from_dgl(...)`
- `graph.to_dgl()` keeps simple homogeneous graphs on `dgl.graph(...)` and exports typed or temporal graphs through `dgl.heterograph(...)`
- canonical edge types survive heterograph round-trips instead of collapsing to VGL's default relation
- external DGL `dgl.NID` / `dgl.EID` metadata imports as VGL `n_id` / `e_id`
- featureless external DGL graphs preserve declared node counts by importing node-id metadata when needed, so isolated nodes survive round-trips
- temporal graphs preserve `graph.schema.time_attr` through the adapter-owned `vgl_time_attr` DGL graph attribute
- sampled graph-classification records can attach labels to sample metadata instead of mutating the graph object

## Mental Model Shift

The package keeps one internal `Graph` abstraction and treats homogeneous, heterogeneous, and temporal graphs as variations of the same core model instead of separate top-level object families.

For training, the important shift is similar: graph classification does not get its own parallel data model. Many-small-graph samples and sampled subgraphs both converge into the same `GraphBatch` contract before they reach the model or task.
