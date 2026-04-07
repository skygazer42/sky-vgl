# Artifact Interop Preflight And Sdist Coverage Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make artifact-level interop smoke fail clearly when required host backends are unavailable, and add real release coverage for the sdist + DGL interop path.

**Architecture:** Extend `scripts/release_smoke.py` with small, testable backend-preflight helpers that probe importability through the same outer `site-packages` bootstrap used by artifact smoke. Keep the default `--interop-backend none` behavior unchanged, surface a clear `SystemExit` message for missing host-assisted backends, and add packaging-level coverage for the sdist + DGL path plus the missing-backend failure path.

**Tech Stack:** Python 3.10+, `argparse`, `subprocess`, `site`, pytest, existing release smoke/build tooling

---

### Task 1: Add Failing Tests For Backend Preflight Helpers

**Files:**
- Modify: `tests/test_release_smoke.py`

**Step 1: Write the failing test**

Add tests that expect:
- a backend-to-import-module mapping for `dgl` and `pyg`
- a helper that reports missing interop backends for `all`
- a clear `SystemExit` when requested host-assisted backends are unavailable

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/test_release_smoke.py`
Expected: FAIL because the preflight helper path does not exist yet.

**Step 3: Write minimal implementation**

Implement only the helper seams needed by the tests:
- backend import-module lookup
- backend availability probe through bootstrapped `site-packages`
- missing-backend aggregation
- explicit preflight validation for selected interop backends

**Step 4: Run targeted verification**

Run: `python -m pytest -q tests/test_release_smoke.py`
Expected: PASS

### Task 2: Wire Backend Preflight Into Release Smoke Runtime

**Files:**
- Modify: `scripts/release_smoke.py`
- Modify: `tests/test_release_packaging.py`

**Step 1: Write the failing test**

Add packaging-level assertions that expect:
- `python scripts/release_smoke.py --artifact-dir <dist> --kind sdist --interop-backend dgl` to pass when DGL is available to artifact smoke
- `python scripts/release_smoke.py --artifact-dir <dist> --kind wheel --interop-backend all` to fail with a clear missing-backend message when DGL is available but PyG is not

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/test_release_packaging.py tests/test_release_smoke.py`
Expected: FAIL until runtime preflight and new packaging coverage exist.

**Step 3: Write minimal implementation**

Update `release_smoke.py` so it:
- validates requested interop backends before running the inline round-trip script
- uses the same outer `site-packages` bootstrap for probing and for smoke execution
- raises a stable `SystemExit` message listing unavailable backends/modules
- keeps default release smoke and single-backend success behavior unchanged

**Step 4: Run targeted verification**

Run: `python -m pytest -q tests/test_release_packaging.py tests/test_release_smoke.py`
Expected: PASS

### Task 3: Document The `all`/Host-Assisted Contract

**Files:**
- Modify: `docs/releasing.md`
- Modify: `docs/support-matrix.md`
- Modify: `tests/test_release_packaging.py`

**Step 1: Write the failing assertion**

Add docs assertions that expect release docs/support guidance to mention:
- artifact interop is host-assisted
- `--interop-backend all` requires both host backends to be importable
- missing host backends fail early instead of silently falling through to a traceback

**Step 2: Run tests to verify they fail**

Run: `python -m pytest -q tests/test_release_packaging.py`
Expected: FAIL until docs are updated.

**Step 3: Write minimal implementation**

Document the release contract without changing default CI cost:
- keep wheel-level artifact interop as the manual release path
- explain that `all` is only for environments with both host backends available
- note that missing host backends now produce an explicit preflight failure

**Step 4: Run targeted verification**

Run: `python -m pytest -q tests/test_release_packaging.py`
Expected: PASS

## Final Verification

Run these commands before merging:

- `python -m pytest -q tests/test_release_smoke.py tests/test_release_packaging.py`
- `python -m ruff check .`
- `python -m mypy vgl`
- `python -m mkdocs build --strict`
- `python -m pytest -q`
- `python -m build`
- `python scripts/release_contract_scan.py --artifact-dir dist`
- `python -m twine check dist/*.whl dist/*.tar.gz`
- `python scripts/release_smoke.py --artifact-dir dist --kind all`
- `python scripts/release_smoke.py --artifact-dir dist --kind sdist --interop-backend dgl`
