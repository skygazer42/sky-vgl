# Release Artifact Interop Smoke Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend built-artifact release smoke coverage so maintainers can validate DGL/PyG interoperability against installed wheel/sdist artifacts instead of only against the repo checkout.

**Architecture:** Extend `scripts/release_smoke.py` with an optional backend-selection path that reuses the existing isolated install flow, then runs public interop round-trips against the installed artifact while borrowing already-installed optional backends from the outer environment. Keep the default CLI behavior unchanged so CI cost stays flat, and document the artifact-level interop path as an explicit release-time/manual verification step.

**Tech Stack:** Python 3.10+, `argparse`, `subprocess`, `site`, pytest, existing VGL release/interop smoke scripts, GitHub Actions docs/contracts

---

### Task 1: Extend Release Smoke for Backend-Selectable Artifact Interop

**Files:**
- Modify: `scripts/release_smoke.py`
- Create: `tests/test_release_smoke.py`

**Step 1: Write the failing test**

Add tests that expect:
- `release_smoke.py` to accept `--interop-backend {dgl,pyg,all}` without changing the default path.
- the backend selection logic to expand `all` into the stable backend catalog.
- the install runner to invoke artifact-level interop checks only when explicitly requested.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/test_release_smoke.py`
Expected: FAIL because `release_smoke.py` does not expose artifact interop options yet.

**Step 3: Write minimal implementation**

Extend `release_smoke.py` so it:
- keeps the current default `--kind` behavior untouched
- adds optional `--interop-backend` flags
- reuses the installed artifact environment for public interop round-trips
- prints stable success lines per artifact/backend combination

**Step 4: Run targeted verification**

Run: `python -m pytest -q tests/test_release_smoke.py`
Expected: PASS

### Task 2: Release Packaging Integration and Real-Env DGL Check

**Files:**
- Modify: `tests/test_release_packaging.py`

**Step 1: Write the failing test**

Add packaging-level assertions that expect:
- the release smoke CLI to expose artifact interop options in a usable way
- a DGL-backed artifact smoke run to pass when `dgl` is actually available in the outer environment

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/test_release_packaging.py tests/test_release_smoke.py`
Expected: FAIL until the new CLI path exists.

**Step 3: Write minimal implementation**

Keep the runtime path additive:
- default release smoke remains base-import only
- explicit `--interop-backend dgl` or `pyg` activates optional artifact interop smoke
- DGL integration test should skip cleanly when `dgl` is unavailable

**Step 4: Run targeted verification**

Run: `python -m pytest -q tests/test_release_packaging.py tests/test_release_smoke.py`
Expected: PASS

### Task 3: Release Docs and Local Automation

**Files:**
- Modify: `docs/releasing.md`
- Modify: `Makefile`
- Modify: `tests/test_exec_scan.py`
- Modify: `tests/test_release_packaging.py`

**Step 1: Write the failing assertions**

Add checks that expect:
- release docs to mention artifact-level interop smoke commands
- local automation to expose a dedicated artifact interop smoke target

**Step 2: Run tests to verify they fail**

Run: `python -m pytest -q tests/test_exec_scan.py tests/test_release_packaging.py tests/test_release_smoke.py`
Expected: FAIL until docs and Makefile are updated.

**Step 3: Write minimal implementation**

Document and expose:
- `python scripts/release_smoke.py --artifact-dir dist --kind wheel --interop-backend dgl`
- `python scripts/release_smoke.py --artifact-dir dist --kind wheel --interop-backend pyg`
- a Make target that wraps the artifact interop smoke flow without affecting default CI

**Step 4: Run targeted verification**

Run: `python -m pytest -q tests/test_exec_scan.py tests/test_release_packaging.py tests/test_release_smoke.py`
Expected: PASS

## Final Verification

Run these commands before merging:

- `python -m pytest -q tests/test_release_smoke.py tests/test_release_packaging.py tests/test_exec_scan.py`
- `python -m build`
- `python scripts/release_smoke.py --artifact-dir dist --kind all`
- `python scripts/release_smoke.py --artifact-dir dist --kind wheel --interop-backend dgl`
- `python -m ruff check .`
- `python -m pytest -q`
