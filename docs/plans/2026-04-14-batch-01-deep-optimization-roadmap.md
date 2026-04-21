# Batch 01 Deep Optimization Roadmap

> **For Codex:** Execute this roadmap with 4 parallel worktrees and 4 agents. Merge back to `main` only after lane-level verification and one final integration pass.

**Goal:** Use one 40-task batch to materially improve VGL across API stability, data/sampling throughput, trainer quality, docs, and release confidence.

**Architecture:** Batch 01 is organized as four parallel lanes so each agent owns a coherent subsystem with minimal file overlap. The batch favors convergence over expansion: reduce legacy surface drag, harden contracts, add performance evidence, and improve release trust before adding more features.

**Tech Stack:** Python 3.10+, PyTorch 2.4+, Hatchling packaging, pytest, mypy, ruff, MkDocs Material, GitHub Actions.

---

## External Reference Baseline

- **PyTorch Geometric**
  - Unified graph/data surface plus heterogeneous support, storage abstractions, and scalable sampling are key strengths.
  - Official references:
    - https://github.com/pyg-team/pytorch_geometric
    - https://pytorch-geometric.readthedocs.io/
- **DGL / GraphBolt**
  - Stage-based dataloading, cacheable pipelines, and distributed graph handling are key strengths.
  - Official references:
    - https://github.com/dmlc/dgl
    - https://www.dgl.ai/

## Current Repository Signals

- The repo already carries legacy compatibility layers and tests that preserve them:
  - `vgl/core`
  - `vgl/data`
  - `vgl/train`
  - `docs/migration-guide.md`
  - `tests/test_preferred_import_paths.py`
  - `tests/test_public_surface_scan.py`
  - `scripts/contracts.py`
- The root package surface is very broad:
  - `vgl/__init__.py`
  - `tests/test_package_exports.py`
- The repo already has benchmark and release scaffolding that should be strengthened instead of replaced:
  - `scripts/benchmark_hotpaths.py`
  - `.github/workflows/ci.yml`
  - `.github/workflows/interop-smoke.yml`
  - `.github/workflows/publish.yml`
- The repo already contains seed plans that Batch 01 should reuse where applicable:
  - `docs/plans/2026-03-17-package-layout-refactor.md`
  - `docs/plans/2026-03-17-training-state-checkpoint-resume.md`
  - `docs/plans/2026-03-21-data-throughput-device-transfer.md`
  - `docs/plans/2026-03-25-dgl-compat-expansion.md`

## Lane And Worktree Map

| Lane | Branch | Worktree | Ownership |
| --- | --- | --- | --- |
| Lane A | `batch-01-core-api` | `.worktrees/batch-01-core-api` | Public API, graph/core convergence, compatibility policy |
| Lane B | `batch-01-data-sampling` | `.worktrees/batch-01-data-sampling` | Data, sampling, distributed dataloading, throughput |
| Lane C | `batch-01-engine-quality` | `.worktrees/batch-01-engine-quality` | Trainer, callbacks, metrics, typing, runtime quality |
| Lane D | `batch-01-docs-release` | `.worktrees/batch-01-docs-release` | Docs, benchmarks, packaging, CI/release UX |

## Batch Execution Rules

1. Every lane starts by locking behavior with tests before structural cleanup.
2. Every task must land with one verification artifact: test, benchmark JSON, docs build, or CI check.
3. Prefer reusing existing plans and scripts over creating new infrastructure.
4. Avoid shared-file collisions across lanes unless the task explicitly requires an integration checkpoint.
5. Merge order for Batch 01: Lane A -> Lane B -> Lane C -> Lane D -> final integration pass on `main`.
6. After all 40 tasks merge, run the full repo gate:
   - `python -m ruff check .`
   - `python -m mypy vgl`
   - `python -m pytest -q`
   - `python scripts/public_surface_scan.py`
   - `python scripts/release_contract_scan.py --artifact-dir dist`
7. Only then start Batch 02.

## Task Backlog

### Lane A: Core API And Compatibility Convergence

**Task 1: Freeze the canonical public API tiers**

- **Files:** `vgl/__init__.py`, `scripts/contracts.py`, `docs/public-surface-contract.md`, `tests/test_package_exports.py`
- **Goal:** Split public API into stable, compatibility, and internal tiers so export growth becomes intentional.
- **Acceptance:** Root exports are documented and guarded by tests; compatibility-only exports are explicitly labeled.
- **Dependency:** None.

**Task 2: Reduce root export sprawl with generated ordering rules**

- **Files:** `vgl/__init__.py`, `scripts/public_surface_scan.py`, `tests/test_public_surface_scan.py`
- **Goal:** Replace hand-grown export ordering with a deterministic convention to lower maintenance cost.
- **Acceptance:** Export ordering becomes stable and scan tests detect drift.
- **Dependency:** Task 1.

**Task 3: Add deprecation warnings for legacy namespaces**

- **Files:** `vgl/core/__init__.py`, `vgl/data/__init__.py`, `vgl/train/__init__.py`, `tests/test_runtime_compat.py`, `docs/migration-guide.md`
- **Goal:** Keep compatibility but make the migration path visible at runtime.
- **Acceptance:** First-touch legacy imports warn once with a clear preferred replacement path.
- **Dependency:** Task 1.

**Task 4: Collapse duplicate graph/core contract tests**

- **Files:** `tests/core/test_graph_homo.py`, `tests/core/test_graph_multi_type.py`, `tests/core/test_schema.py`, `tests/test_preferred_import_paths.py`
- **Goal:** Remove redundant coverage while keeping behavior locked.
- **Acceptance:** Coverage remains stable and duplicated assertions move into shared helpers.
- **Dependency:** None.

**Task 5: Harden graph construction invariants**

- **Files:** `vgl/graph/graph.py`, `vgl/graph/schema.py`, `vgl/graph/errors.py`, `tests/core/test_graph_homo.py`, `tests/core/test_schema.py`
- **Goal:** Make invalid graph states fail early with consistent error types and messages.
- **Acceptance:** Edge shape, feature shape, and schema mismatch failures are explicit and regression-tested.
- **Dependency:** None.

**Task 6: Standardize batch/view/store naming**

- **Files:** `vgl/graph/__init__.py`, `vgl/core/__init__.py`, `docs/core-concepts.md`, `tests/test_package_exports.py`
- **Goal:** Remove naming ambiguity around `GraphBatch`, views, and store-related symbols.
- **Acceptance:** Public docs and exports use one naming story; compatibility aliases remain documented.
- **Dependency:** Tasks 1 and 3.

**Task 7: Snapshot the public API surface**

- **Files:** `scripts/contracts.py`, `tests/test_public_surface_scan.py`, `tests/test_release_contract_scan.py`
- **Goal:** Add a golden surface snapshot so accidental API drift is visible in review.
- **Acceptance:** CI fails on unexpected reexport additions/removals.
- **Dependency:** Tasks 1 and 2.

**Task 8: Unify serialization/version metadata for graph objects and checkpoints**

- **Files:** `vgl/graph/graph.py`, `vgl/engine/checkpoints.py`, `docs/public-surface-contract.md`, `tests/test_release_packaging.py`
- **Goal:** Ensure format metadata is explicit and versioned across saved artifacts.
- **Acceptance:** Serialized artifact version fields are documented and tested.
- **Dependency:** Task 5.

**Task 9: Publish the compatibility sunset policy**

- **Files:** `docs/migration-guide.md`, `docs/changelog.md`, `README.md`
- **Goal:** Define when `vgl.core`, `vgl.data`, and `vgl.train` stop being preferred and how breaking changes are announced.
- **Acceptance:** README and docs carry the same migration guidance.
- **Dependency:** Tasks 1 and 3.

**Task 10: Add API-focused integration smoke suites**

- **Files:** `tests/test_package_exports.py`, `tests/test_release_smoke.py`, `tests/test_interop_smoke.py`
- **Goal:** Verify root imports, legacy imports, and preferred imports all behave consistently after convergence.
- **Acceptance:** One smoke suite covers stable, preferred, and compatibility import paths.
- **Dependency:** Tasks 2, 3, and 7.

### Lane B: Data, Sampling, And Throughput

**Task 11: Add stage-level profiling for sampling plans**

- **Files:** `vgl/dataloading/plan.py`, `vgl/dataloading/executor.py`, `vgl/dataloading/materialize.py`, `tests/data/test_sampling_executor.py`
- **Goal:** Record per-stage timing so slow sampling paths can be localized quickly.
- **Acceptance:** Sampling plan execution emits stable stage timing fields.
- **Dependency:** None.

**Task 12: Remove compatibility duplication between `vgl.data` and `vgl.dataloading`**

- **Files:** `vgl/data/__init__.py`, `vgl/data/plan.py`, `vgl/data/executor.py`, `vgl/data/requests.py`, `vgl/dataloading/__init__.py`
- **Goal:** Keep compatibility wrappers thin and declarative instead of partially reimplementing surface logic.
- **Acceptance:** Legacy reexports resolve through one clear path and related tests stay green.
- **Dependency:** Lane A Task 3.

**Task 13: Normalize sample record contracts**

- **Files:** `vgl/dataloading/records.py`, `vgl/dataloading/requests.py`, `tests/data/test_seed_requests.py`, `tests/data/test_loader.py`
- **Goal:** Align node, link, and temporal record semantics to reduce branching in loaders and tasks.
- **Acceptance:** Required and optional fields are uniform and documented.
- **Dependency:** None.

**Task 14: Make negative sampling deterministic under seeded execution**

- **Files:** `vgl/dataloading/sampler.py`, `vgl/dataloading/advanced.py`, `tests/data/test_link_neighbor_sampler.py`, `tests/data/test_sampler_compatibility_extensions.py`
- **Goal:** Guarantee deterministic results for link-level tests and reproducible experiments.
- **Acceptance:** Seeded sampling tests pass across repeated runs.
- **Dependency:** Task 13.

**Task 15: Add reusable sampling-materialization caches**

- **Files:** `vgl/dataloading/materialize.py`, `vgl/dataloading/executor.py`, `vgl/_neighbor_sampling.py`, `tests/data/test_batch_materialize.py`
- **Goal:** Reuse stable intermediate tensors when repeated requests hit the same graph slice.
- **Acceptance:** Benchmarks or tests show fewer repeated materialization allocations.
- **Dependency:** Task 11.

**Task 16: Strengthen large-graph memory budget tests**

- **Files:** `tests/integration/test_foundation_large_graph_flow.py`, `tests/integration/test_foundation_ondisk_sampling.py`, `scripts/benchmark_hotpaths.py`
- **Goal:** Add memory-sensitive coverage so throughput work does not regress practical graph sizes.
- **Acceptance:** CI smoke presets stay lightweight, while local/nightly presets cover larger memory footprints.
- **Dependency:** Task 11.

**Task 17: Clarify prefetch and backpressure semantics in loaders**

- **Files:** `vgl/data/loader.py`, `vgl/dataloading/loader.py`, `tests/data/test_loader_prefetch.py`, `docs/guide/sampling.md`
- **Goal:** Make loader buffering behavior explicit for both performance tuning and correctness.
- **Acceptance:** Loader docs, API behavior, and tests agree on prefetch guarantees.
- **Dependency:** Tasks 11 and 13.

**Task 18: Add local-vs-distributed parity tests for samplers**

- **Files:** `tests/distributed/test_sampling_coordinator.py`, `tests/data/test_node_neighbor_sampler.py`, `tests/data/test_temporal_neighbor_sampler.py`
- **Goal:** Prove distributed routing does not silently diverge from single-machine sampling semantics.
- **Acceptance:** Shared fixtures compare identical seed inputs across local and distributed paths.
- **Dependency:** Tasks 13 and 14.

**Task 19: Expand benchmark coverage for sampling and routing**

- **Files:** `scripts/benchmark_hotpaths.py`, `.github/workflows/ci.yml`, `docs/guide/sampling.md`
- **Goal:** Move benchmarks beyond query ops and include representative sampling and routing hotpaths with persisted JSON artifacts.
- **Acceptance:** CI uploads benchmark JSON and schema changes are versioned.
- **Dependency:** Tasks 11 and 16.

**Task 20: Add a sampling strategy decision guide**

- **Files:** `docs/guide/sampling.md`, `docs/getting-started/quickstart.md`, `README.md`
- **Goal:** Help users choose `FullGraphSampler`, neighbor sampling, GraphSAINT, cluster loading, and temporal sampling correctly.
- **Acceptance:** Docs include a decision matrix keyed by graph size, task type, and memory budget.
- **Dependency:** Tasks 17 and 19.

### Lane C: Trainer, Metrics, And Runtime Quality

**Task 21: Split `Trainer` configuration validation from execution**

- **Files:** `vgl/engine/trainer.py`, `tests/train/test_trainer.py`, `tests/train/test_trainer_plus.py`
- **Goal:** Reduce constructor complexity and make invalid config handling easier to test.
- **Acceptance:** Validation helpers or config objects isolate parameter normalization from runtime logic.
- **Dependency:** None.

**Task 22: Introduce typed protocols for model/task/batch contracts**

- **Files:** `vgl/engine/trainer.py`, `vgl/tasks/base.py`, `vgl/metrics/base.py`, `tests/train/test_task_metric_contract.py`
- **Goal:** Improve mypy coverage and make trainer expectations explicit.
- **Acceptance:** Core trainer/task paths use protocol-friendly type hints and mypy coverage improves.
- **Dependency:** Task 21.

**Task 23: Stabilize metric naming and monitor resolution**

- **Files:** `vgl/engine/monitoring.py`, `vgl/metrics/__init__.py`, `tests/train/test_metrics.py`, `tests/train/test_monitoring.py`
- **Goal:** Eliminate ambiguous metric keys across train/val/test loops and callbacks.
- **Acceptance:** Monitor selection rules are documented and covered by conflict tests.
- **Dependency:** None.

**Task 24: Strengthen checkpoint resume compatibility**

- **Files:** `vgl/engine/checkpoints.py`, `vgl/engine/history.py`, `tests/train/test_checkpoints.py`, `tests/train/test_history.py`
- **Goal:** Ensure resumed runs preserve monitor state, epoch counters, and optimizer/scheduler metadata.
- **Acceptance:** Resume tests cover forward compatibility and corrupted metadata failure paths.
- **Dependency:** Tasks 21 and 23.

**Task 25: Formalize callback and logger event contracts**

- **Files:** `vgl/engine/callbacks.py`, `vgl/engine/logging.py`, `docs/guide/training.md`, `tests/train/test_callbacks.py`, `tests/train/test_logging.py`
- **Goal:** Make lifecycle hooks, event payloads, and structured logs consistent and stable.
- **Acceptance:** Event fields are documented and regression-tested.
- **Dependency:** Task 21.

**Task 26: Expand runtime compile and mixed-precision readiness**

- **Files:** `tests/test_runtime_compile_smoke.py`, `tests/train/test_mixed_precision.py`, `vgl/engine/trainer.py`
- **Goal:** Prove the trainer works cleanly with `torch.compile`, AMP, and device transfer settings.
- **Acceptance:** Smoke tests cover compile, bf16/fp16, and failure-mode skips intentionally.
- **Dependency:** Task 21.

**Task 27: Add profiler schema and retention rules**

- **Files:** `vgl/engine/trainer.py`, `vgl/engine/history.py`, `docs/guide/training.md`, `tests/train/test_monitoring.py`
- **Goal:** Standardize the `"profile"` payload so timing data stays stable across releases.
- **Acceptance:** Docs and tests assert required profile keys.
- **Dependency:** Tasks 21 and 25.

**Task 28: Add failure-injection tests for early stopping, checkpointing, and loggers**

- **Files:** `tests/train/test_callbacks.py`, `tests/train/test_checkpoints.py`, `tests/train/test_logging.py`
- **Goal:** Ensure callbacks and loggers fail loudly and predictably instead of corrupting training state.
- **Acceptance:** Simulated I/O and callback failures are covered.
- **Dependency:** Tasks 24 and 25.

**Task 29: Add recipe-level training smoke suites**

- **Files:** `tests/integration/test_end_to_end_homo.py`, `tests/integration/test_end_to_end_hetero.py`, `tests/integration/test_end_to_end_temporal.py`, `examples/`
- **Goal:** Treat common user workflows as executable recipes instead of only low-level unit coverage.
- **Acceptance:** At least one compact smoke path exists for node, link, graph, and temporal training.
- **Dependency:** Tasks 23, 25, and 26.

**Task 30: Document the trainer quality bar**

- **Files:** `docs/guide/training.md`, `README.md`, `.github/workflows/ci.yml`
- **Goal:** Make the repo’s runtime quality gates explicit to contributors and users.
- **Acceptance:** README and training guide explain which checks prove trainer safety.
- **Dependency:** Tasks 24 through 29.

### Lane D: Docs, Release, Benchmarks, And Ecosystem Trust

**Task 31: Publish a VGL vs PyG vs DGL positioning page**

- **Files:** `docs/index.md`, `docs/faq.md`, `README.md`
- **Goal:** Explain where VGL is stronger, where it is thinner, and why users should choose it.
- **Acceptance:** Comparison is factual, cites official project docs, and avoids unsupported benchmark claims.
- **Dependency:** None.

**Task 32: Add benchmark artifact docs and schema guarantees**

- **Files:** `scripts/benchmark_hotpaths.py`, `docs/changelog.md`, `docs/guide/training.md`, `.github/workflows/ci.yml`
- **Goal:** Treat benchmark JSON as a versioned artifact instead of an opaque CI side output.
- **Acceptance:** Schema versioning and consumer expectations are documented.
- **Dependency:** Lane B Task 19 and Lane C Task 27.

**Task 33: Tighten package and import-time checks**

- **Files:** `tests/test_release_packaging.py`, `tests/test_release_smoke.py`, `scripts/release_smoke.py`, `pyproject.toml`
- **Goal:** Catch oversized wheels, missing package data, and import regressions earlier.
- **Acceptance:** Packaging tests cover import-time smoke and artifact metadata consistently.
- **Dependency:** None.

**Task 34: Expand optional dependency matrix coverage**

- **Files:** `.github/workflows/ci.yml`, `.github/workflows/interop-smoke.yml`, `pyproject.toml`, `docs/getting-started/installation.md`
- **Goal:** Make extras behavior predictable across `networkx`, `scipy`, `tensorboard`, `pyg`, and `dgl`.
- **Acceptance:** Docs match CI matrix and unsupported combinations fail clearly.
- **Dependency:** Task 33.

**Task 35: Expand interop round-trip docs and smoke coverage**

- **Files:** `docs/api/compat.md`, `tests/test_interop_smoke.py`, `tests/compat/`, `scripts/interop_smoke.py`
- **Goal:** Turn interop claims into explicit round-trip guarantees and documented caveats.
- **Acceptance:** Each supported backend has a clear smoke path and caveat list.
- **Dependency:** Task 34.

**Task 36: Rewrite quickstart paths around user jobs-to-be-done**

- **Files:** `README.md`, `docs/getting-started/quickstart.md`, `docs/guide/node-classification.md`, `docs/guide/graph-classification.md`, `docs/guide/link-prediction.md`, `docs/guide/temporal.md`
- **Goal:** Replace package-tour docs with task-oriented first-success flows.
- **Acceptance:** Each common workflow reaches first train/test result with one canonical path.
- **Dependency:** Lane B Task 20 and Lane C Task 29.

**Task 37: Add a release checklist and failure triage guide**

- **Files:** `docs/releasing.md`, `.github/workflows/publish.yml`, `.github/workflows/ci.yml`
- **Goal:** Reduce release-time guesswork and make failure handling repeatable.
- **Acceptance:** Release docs map directly to workflow jobs and expected artifacts.
- **Dependency:** Tasks 33 and 34.

**Task 38: Add contribution templates for performance, interop, and dataset bugs**

- **Files:** `.github/ISSUE_TEMPLATE/`, `README.md`, `docs/faq.md`
- **Goal:** Improve incoming issue quality for the repo’s hardest-to-debug problem classes.
- **Acceptance:** Issue templates ask for the exact artifacts already produced by CI and smoke scripts.
- **Dependency:** Tasks 32, 35, and 37.

**Task 39: Add changelog discipline for compatibility and benchmarks**

- **Files:** `docs/changelog.md`, `docs/migration-guide.md`, `README.md`
- **Goal:** Make compatibility changes, performance shifts, and benchmark methodology visible release over release.
- **Acceptance:** Changelog format reserves sections for API, performance, interop, and migration.
- **Dependency:** Lane A Task 9 and Task 32.

**Task 40: Define the post-batch merge protocol**

- **Files:** `docs/plans/2026-04-14-batch-01-deep-optimization-roadmap.md`, `README.md`
- **Goal:** Standardize how completed lane branches merge into `main`, how integration is verified, and how Batch 02 intake starts.
- **Acceptance:** The repo has one documented rule for merge order, verification, and next-batch kickoff.
- **Dependency:** All prior tasks.

## Merge Protocol After The 40 Tasks

1. Rebase each lane branch on current `main`.
2. Run lane-local verification in each worktree.
3. Merge lanes into `main` one by one in this order:
   - `batch-01-core-api`
   - `batch-01-data-sampling`
   - `batch-01-engine-quality`
   - `batch-01-docs-release`
4. Run the full integration gate on `main`.
5. Write a batch summary covering:
   - changed files
   - simplifications made
   - benchmark deltas
   - compatibility/deprecation changes
   - remaining risks
6. Tag the repository state for Batch 02 planning.
7. Create the next four branches and worktrees only after Batch 01 is green on `main`.

## Immediate Agent Prompts

- **Agent A / Lane A:** Converge public API and legacy namespace policy without breaking preferred import paths.
- **Agent B / Lane B:** Improve dataloading observability, deterministic sampling, and local/distributed parity.
- **Agent C / Lane C:** Refactor `Trainer` for clearer validation, stronger typing, and better runtime guarantees.
- **Agent D / Lane D:** Strengthen docs, release confidence, benchmark visibility, and ecosystem positioning.
