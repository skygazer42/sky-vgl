# VGL Package Rename and Layout Migration Design

**Date:** 2026-03-15
**Status:** Approved for planning

## Goal

Rename the Python package and project identity from `gnn` to `vgl`, remove the `src/` layout, expose a broader root-package API, and align the codebase, examples, tests, and documentation around the `VGL` name.

## Scope Decisions

- The canonical package name becomes `vgl`
- The repository should no longer use the `src/` layout
- The root package should expose a broad convenience API
- No `gnn` compatibility shim should remain after migration
- Current user-facing docs and historical planning docs should both be updated to use `vgl`
- Existing node classification, graph classification, and temporal event prediction flows must continue to work

## Chosen Direction

Three directions were considered:

1. A hard cut to `vgl` with a top-level `vgl/` package and no compatibility layer
2. A two-step migration that first changes the package path and later broadens the root-package exports
3. A packaging-only rename that keeps the old internal layout and naming around

The chosen direction is:

> Perform a single hard cut to a top-level `vgl/` package, remove the `src/` layout, and expose common framework objects from `vgl.__init__`.

This is the only direction that fully matches the project goal that the repository itself is `VGL`. Keeping `gnn` or `src/` around as transitional structure would preserve ambiguity in both the codebase and the public API.

## Architecture / Packaging Surface

After migration, the package layout should be:

- `vgl/`
- `vgl/core/`
- `vgl/data/`
- `vgl/nn/`
- `vgl/train/`
- `vgl/compat/`

The internal module structure should otherwise stay the same. This is a package and layout migration, not a redesign of the framework internals.

`pyproject.toml` should be updated so that:

- `[project].name = "vgl"`
- pytest no longer injects `src` into `pythonpath`
- tooling runs against the top-level `vgl/` package directly

The root package should become a broad convenience entrypoint. `vgl.__init__` should re-export at least:

- `Graph`
- `GraphBatch`
- `TemporalEventBatch`
- `GraphView`
- `GraphSchema`
- `ListDataset`
- `Loader`
- `FullGraphSampler`
- `NodeSeedSubgraphSampler`
- `SampleRecord`
- `TemporalEventRecord`
- `MessagePassing`
- `global_mean_pool`
- `global_sum_pool`
- `global_max_pool`
- `Task`
- `Trainer`
- `Metric`
- `NodeClassificationTask`
- `GraphClassificationTask`
- `TemporalEventPredictionTask`
- `__version__`

This root-package surface should complement the subpackages rather than replace them. `from vgl import Graph, Loader, Trainer` should work, while `from vgl.train.tasks import GraphClassificationTask` should remain valid.

## Code Migration Rules

This migration should be repository-wide and explicit.

The codebase should follow four rules:

1. Move `src/gnn/` to `vgl/` in one migration instead of keeping a duplicated tree
2. Replace current `gnn` absolute imports with `vgl` absolute imports
3. Remove example-time `sys.path` hacks that point at `src`
4. Keep subpackage imports working while broadening the root-package exports

This phase should not introduce:

- a `gnn` compatibility package
- relative-import refactors unrelated to the rename
- partial dual-package support
- extra naming cleanups beyond the package and project rename

The migration is complete only when current production code, tests, examples, and active docs no longer rely on `gnn` or `src/gnn` references.

## Documentation Strategy

The repository should present a single project identity after this migration.

That means updating:

- `README.md`
- `docs/quickstart.md`
- `docs/core-concepts.md`
- existing examples
- current tests
- historical documents under `docs/plans/`

Historical plan files should keep their dates and filenames, but their content should be updated so that project names, import examples, and package paths refer to `vgl`. The purpose is not to rewrite history, but to keep the repository internally consistent for current readers.

## Error Handling and Compatibility Boundaries

This phase should fail fast on incomplete migration states during development.

The migration should be considered incorrect if:

- root-package re-exports no longer match working subpackage objects
- examples still depend on `src` path injection
- tooling still assumes the old layout
- tests pass only because of stale `pythonpath` configuration

There should be no runtime compatibility promise for `gnn`. Any remaining `gnn` imports in the active codebase count as migration bugs, not supported behavior.

## Testing Strategy

The migration should be validated at four layers:

### Packaging tests

- importing common objects directly from `vgl`
- importing existing module paths from `vgl.*`
- version string remains exposed correctly

### Regression tests

- node classification tests stay green
- graph classification tests stay green
- temporal event prediction tests stay green

### Example tests

- homo node classification example still runs
- homo graph classification example still runs
- hetero node classification example still runs
- hetero graph classification example still runs
- temporal event prediction example still runs

### Repository consistency checks

- no active code or docs references `import gnn` or `from gnn`
- no active code or docs references `src/gnn`

## Deliverables

This phase should ship:

- a top-level `vgl/` package
- an updated `pyproject.toml` with the `vgl` project name
- broad root-package re-exports
- updated tests, examples, and docs
- updated historical planning docs
- a fully green verification run on the renamed package

## Explicit Non-Goals

Do not include:

- link prediction
- additional training tasks
- API redesigns unrelated to package identity
- new compatibility shims for `gnn`
- restructuring the framework internals beyond the rename and layout move

## Repository Touchpoints

This migration will mainly affect:

- `pyproject.toml`
- `vgl/` after the move from `src/gnn/`
- `tests/`
- `examples/`
- `README.md`
- `docs/quickstart.md`
- `docs/core-concepts.md`
- `docs/plans/`

## Stability Constraints

This phase should follow four rules:

1. `vgl` is the only active package name
2. the repository no longer depends on the `src/` layout
3. root-package convenience imports must work for common objects
4. existing framework behavior must survive the rename unchanged

## Acceptance Criteria

The phase is complete when:

1. `pyproject.toml` publishes the project as `vgl`
2. the package directory is `vgl/` at the repository root
3. current code, tests, examples, and docs use `vgl`
4. historical planning docs also refer to `vgl`
5. root-package imports for common objects succeed
6. full test, lint, type-check, and example verification passes after the rename

## Next Step

The next step is to create a detailed implementation plan with exact files, tests, commands, and commit checkpoints.
