# gnn

Unified graph learning framework with a stable core abstraction for homogeneous, heterogeneous, and temporal graphs.

## Current Scope

- One canonical `Graph` abstraction
- Homogeneous, heterogeneous, and temporal graph constructors
- Schema validation, graph views, and graph batching
- Minimal data pipeline with dataset, sampler, and loader contracts
- `MessagePassing` plus `GCNConv`, `SAGEConv`, and `GATConv`
- Minimal training loop with `NodeClassificationTask` and `Trainer`
- Compatibility adapters for PyG-style and DGL-style objects

## Examples

- `python examples/homo/node_classification.py`
- `python examples/hetero/node_classification.py`
- `python examples/temporal/event_prediction.py`

## Verification

- `python -m pytest -v`
- `python -m ruff check .`
- `python -m mypy src`
