# GNN Framework Design

**Date:** 2026-03-14
**Status:** Approved for planning

## Goal

Build a PyTorch-based graph learning framework that feels familiar to users coming from DGL and PyG, while keeping the framework's core data model under our control. The first phase should prioritize a stable public API and a unified abstraction across homogeneous, heterogeneous, and temporal graphs.

## Scope Decisions

- Build from scratch in `F:\gnn\gnn`
- Use `PyTorch` as the only backend in phase 1
- Support `homogeneous`, `heterogeneous`, and `temporal` graphs through one core abstraction
- Prioritize a `unified abstraction` over maximum operator count
- Keep the external UX close to `PyG/DGL` where it helps migration
- Prioritize `API stability` over shipping a large surface area quickly

## Chosen Direction

Three directions were considered:

1. A single normalized graph core with a compatibility layer
2. Separate implementations for homogeneous, heterogeneous, and temporal graphs behind a shared protocol
3. A compatibility-first facade that mirrors PyG and DGL internally

The chosen direction is:

> Use a single normalized graph core and absorb the best parts of the PyG/DGL user experience at the API boundary.

This is the most difficult route in the short term, but it is the only one that aligns with the requirement for a stable API and a long-lived framework core. Separate internal implementations would drift over time, and a facade-first design would let outside libraries define the package's semantics.

## Architecture

The package should be split into five primary layers:

- `gnn.core`
  - Defines the canonical graph model, schema, stores, views, validation, and batch objects.
- `gnn.data`
  - Owns datasets, transforms, samplers, loaders, and collation rules.
- `gnn.nn`
  - Owns `MessagePassing`, common convolution layers, hetero wrappers, and temporal encoders.
- `gnn.train`
  - Owns tasks, metrics, evaluators, trainer lifecycle, callbacks, and checkpoints.
- `gnn.compat`
  - Owns import/export adapters and migration helpers for PyG and DGL users.

This layering keeps the core graph abstraction independent of the training stack. Users should be able to use `gnn.core` and `gnn.nn` directly without going through a trainer.

## Core Abstractions

### Graph

`Graph` is the only canonical graph object exposed publicly.

- A homogeneous graph is represented as a graph with one node type and one edge type.
- A heterogeneous graph is represented by multiple node stores and edge stores.
- A temporal graph is represented by attaching time-aware metadata to nodes, edges, or events and exposing temporal query views.

The framework should not create separate top-level objects such as `HomoGraph`, `HeteroGraph`, and `TemporalGraph`. That would make the public API look unified while keeping the semantics fragmented internally.

### Stores

The graph should use normalized stores internally:

- `NodeStore`
- `EdgeStore`
- `EventStore`

Each store owns typed features and metadata. This structure gives the framework enough flexibility to support typed relations and temporal data without creating parallel graph systems.

### Schema

Every graph should carry an explicit schema object. The schema is a stability mechanism, not just metadata.

It should record:

- node types
- edge types
- required feature names
- target names
- time field names
- index conventions

Any graph constructor, converter, sampler, or trainer entrypoint should validate against the schema before running.

### GraphView and GraphBatch

The design should separate storage from views.

- `Graph` owns normalized storage.
- `GraphView` represents lightweight projections such as subgraphs, edge-type slices, snapshots, and time windows.
- `GraphBatch` represents the unit that flows into model execution and training steps.

`GraphView` should default to a lightweight wrapper over underlying tensors instead of copying eagerly. That is especially important for heterogeneous and temporal workloads, where repeated slicing is common.

## Public API Shape

The stable public API should be intentionally small:

- `Graph`
- `GraphBatch`
- `Dataset`
- `Sampler`
- `Loader`
- `MessagePassing`
- `Task`
- `Trainer`
- `Metric`
- `compat.from_pyg`
- `compat.to_pyg`
- `compat.from_dgl`
- `compat.to_dgl`

Everything else should stay behind module boundaries until it proves it belongs in the public surface area.

## API Style

The package should blend PyG-like convenience with DGL-like graph-centric access patterns.

### Homogeneous graph convenience

```python
g = Graph.homo(edge_index=edge_index, x=x, y=y)
g.x
g.edge_index
g.y
g.train_mask
```

### DGL-like attribute aliases

```python
g.ndata["x"]
g.edata["weight"]
```

### Heterogeneous access

```python
g = Graph.hetero(
    nodes={
        "paper": {"x": paper_x},
        "author": {"x": author_x},
    },
    edges={
        ("author", "writes", "paper"): {"edge_index": writes_index},
    },
)

g.nodes["paper"].x
g.edges[("author", "writes", "paper")].edge_index
```

### Temporal access

```python
g = Graph.temporal(...)
g.snapshot(t)
g.window(start=t1, end=t2)
g.events(node_type="user")
```

### Compatibility entrypoints

```python
Graph.from_pyg(data)
Graph.from_dgl(graph)
graph.to_pyg()
graph.to_dgl()
```

The compatibility layer should help users enter and leave the framework. It should not define the framework's internal semantics.

## Data Flow

The data path should be fixed and predictable:

`Dataset -> Transform -> Sampler -> Loader -> GraphBatch -> Model -> Task -> Metric`

Each layer should do one job:

- `Dataset` supplies graphs or large-graph handles.
- `Transform` normalizes and enriches graphs.
- `Sampler` produces views and sampled neighborhoods.
- `Loader` batches sampled items into `GraphBatch`.
- `Task` extracts supervision targets and loss definitions.
- `Metric` evaluates outputs in a task-aware but reusable way.

This arrangement makes single-graph, multi-graph, heterogeneous, and temporal training flows look structurally similar.

## Training Layer

The training layer should be thin compared to the graph core.

### Public training objects

- `Task`
- `Trainer`
- `Metric`
- `Evaluator`

The training layer should make common workflows easy without forcing all users into a single lifecycle. Users should still be able to run their own PyTorch loops with `Graph`, `GraphBatch`, and `MessagePassing` modules.

### Training experience

```python
graph = Graph.from_pyg(data)

task = NodeClassificationTask(
    target="y",
    split=("train_mask", "val_mask", "test_mask"),
    loss="cross_entropy",
    metrics=["accuracy", "f1"],
)

model = GCN(
    in_channels=graph.x.size(-1),
    hidden_channels=128,
    out_channels=task.num_classes,
)

trainer = Trainer(
    model=model,
    task=task,
    optimizer=torch.optim.Adam,
    lr=1e-3,
    max_epochs=200,
)

trainer.fit(graph)
result = trainer.test(graph)
```

The UX should feel familiar to users of existing graph libraries, but all internal dispatch should route through the framework's own `Graph`, `Task`, and `Trainer` protocols.

## Model Extension Strategy

There should be two extension levels:

1. Low-level `MessagePassing`
2. Higher-level ready-to-use modules

Phase 1 should include:

- `GCNConv`
- `SAGEConv`
- `GATConv`
- `HeteroConv`
- a small temporal encoding module

`MessagePassing` should accept normalized graph views internally, not just a raw `edge_index` tensor. Convenience overloads can still support direct calls like `conv(x, edge_index)` for migration ergonomics.

## Phase 1 Deliverables

Phase 1 should ship:

- a stable `Graph` abstraction
- constructors for homogeneous, heterogeneous, and temporal graphs
- schema validation
- `GraphView` and `GraphBatch`
- minimal dataset, sampler, and loader infrastructure
- `MessagePassing`
- `GCNConv`, `SAGEConv`, and `GATConv`
- basic hetero and temporal extensions
- task interfaces for node classification, graph classification, link prediction, and temporal event prediction
- a generic `Trainer`
- import/export adapters for PyG and DGL
- representative examples and integration tests

## Explicit Non-Goals for Phase 1

Phase 1 should not attempt to match the breadth of PyG or DGL.

Do not include:

- a large operator zoo
- distributed training
- graph database connectors
- graph compilation and advanced kernel optimization
- AutoML or experiment management platforms
- every sampling strategy at once

These are valid future extensions, but they should not distort the core API before it is stable.

## Repository Structure

The repository should start with a long-lived layout:

```text
gnn/
  pyproject.toml
  README.md
  docs/
    plans/
  src/
    gnn/
      core/
      data/
      nn/
      train/
      compat/
      utils/
  tests/
    core/
    data/
    nn/
    train/
    compat/
  examples/
    homo/
    hetero/
    temporal/
```

This structure makes the intended module boundaries visible from the beginning and reduces the cost of future refactors.

## Testing Strategy

Because API stability is a phase 1 priority, tests should emphasize contracts instead of only happy-path execution.

The suite should include:

- contract tests for graph constructors and graph schema behavior
- behavior tests for message passing, sampling, batching, and trainer logic
- compatibility tests for PyG and DGL adapters
- integration tests for homogeneous, heterogeneous, and temporal end-to-end flows

TDD should be used for every public API addition. The public contract needs to be pinned down with failing tests before implementation.

## Evolution Constraints

The project should follow these rules in phase 1:

1. Every public API change requires contract tests first.
2. The compatibility layer must not define or leak the core data model.
3. Homogeneous graph features must remain a special case of the unified graph semantics, not a forked API path.
4. Temporal features should extend the view/query model instead of introducing a parallel data structure.

These constraints keep the framework from turning into three similar systems that only share names.

## Main Risks and Mitigations

### Risk: Temporal support overcomplicates the core early

Mitigation:

- keep temporal semantics view-driven
- model time as validated metadata plus query helpers
- delay advanced temporal samplers until the core view model is stable

### Risk: Compatibility pressure bends the core toward PyG/DGL internals

Mitigation:

- limit compatibility work to adapters and ergonomic aliases
- avoid depending on third-party graph object semantics internally

### Risk: API stability slows down progress too much

Mitigation:

- keep the initial public surface area small
- expose only core objects and converters
- allow internal modules to move until they prove stable

## Next Step

The next step is to turn this design into an implementation plan with exact files, tests, commands, and staged milestones.
