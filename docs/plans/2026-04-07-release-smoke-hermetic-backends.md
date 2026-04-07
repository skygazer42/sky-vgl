# Release Smoke Hermetic Backend Overrides Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make artifact-level release smoke success paths locally testable without requiring real host PyG installs by adding a controlled extra-site-dir override and hermetic fake-backend packaging tests.

**Architecture:** Extend `scripts/release_smoke.py` so it can prepend optional extra site-package directories from an environment override while preserving the default host-assisted behavior. Reuse the fake-backend pattern already proven in `tests/test_interop_smoke.py` to add built-artifact packaging tests for `--interop-backend pyg` and `--interop-backend all` success paths against installed wheel artifacts.

**Tech Stack:** Python 3.10+, `os.environ`, `site`, `subprocess`, pytest, existing release smoke/build tooling

---

### Task 1: Add Failing Helper Tests For Extra Site Directories

**Files:**
- Modify: `tests/test_release_smoke.py`

**Step 1: Write the failing test**

Add tests that expect:
- `release_smoke.py` to read a stable environment variable for extra site directories
- the resolved dependency paths to prepend the override directories before the default outer site-packages
- empty / unset overrides to preserve the current default behavior

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/test_release_smoke.py`
Expected: FAIL because the override helper does not exist yet.

**Step 3: Write minimal implementation**

Implement only the helper seam needed for tests:
- parse the environment variable into `Path` entries
- prepend valid directories to the existing outer site-package list
- keep default behavior unchanged when the variable is absent

**Step 4: Run targeted verification**

Run: `python -m pytest -q tests/test_release_smoke.py`
Expected: PASS

### Task 2: Add Hermetic Artifact Interop Packaging Coverage

**Files:**
- Modify: `tests/test_release_packaging.py`
- Modify: `scripts/release_smoke.py`

**Step 1: Write the failing test**

Add packaging-level tests that:
- create fake `torch_geometric` and fake `dgl` packages in a temp directory
- pass that directory into `release_smoke.py` through the new override
- expect `python scripts/release_smoke.py --artifact-dir <dist> --kind wheel --interop-backend pyg` to pass without real host PyG
- expect `python scripts/release_smoke.py --artifact-dir <dist> --kind wheel --interop-backend all` to pass with fake PyG + fake DGL available

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/test_release_packaging.py tests/test_release_smoke.py`
Expected: FAIL until the override is wired into release smoke.

**Step 3: Write minimal implementation**

Update `release_smoke.py` so it:
- resolves dependency paths through the new override-aware helper
- uses the same resolved path list for import checks, backend preflight, and interop execution
- keeps default host-assisted behavior and CLI flags unchanged

**Step 4: Run targeted verification**

Run: `python -m pytest -q tests/test_release_packaging.py tests/test_release_smoke.py`
Expected: PASS

### Task 3: Lock The Maintainer Contract

**Files:**
- Modify: `docs/releasing.md`
- Modify: `tests/test_release_packaging.py`

**Step 1: Write the failing assertion**

Add docs assertions that expect the release doc to mention:
- artifact interop defaults to host-assisted dependency discovery
- hermetic test coverage also exists for fake-backend success paths

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/test_release_packaging.py`
Expected: FAIL until the docs mention the maintained contract.

**Step 3: Write minimal implementation**

Document the contract narrowly:
- real release use remains host-assisted
- local tests additionally exercise fake-backend success paths for coverage

**Step 4: Run targeted verification**

Run: `python -m pytest -q tests/test_release_packaging.py`
Expected: PASS

## Final Verification

Run these commands before merging:

- `python -m pytest -q tests/test_release_smoke.py tests/test_release_packaging.py`
- `python scripts/full_scan.py`
- `python -m ruff check .`
- `python -m mypy vgl`
- `python -m mkdocs build --strict`
- `python -m pytest -q`
