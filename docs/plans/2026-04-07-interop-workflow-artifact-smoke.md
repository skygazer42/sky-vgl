# Interop Workflow Artifact Smoke Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend the manual/nightly interop workflow so repository automation validates built-wheel artifact interop success paths, including the combined `--backend all` path in a prepared environment.

**Architecture:** Keep the existing checkout-level `pyg` and `dgl` smoke jobs, then add a combined workflow job that installs both optional backends, builds the distribution artifacts, and runs both checkout-level and artifact-level `all` interop smoke. Update scan/tests/docs so the new automation contract stays visible and enforceable.

**Tech Stack:** GitHub Actions, Python 3.11, pytest, existing interop/release smoke scripts, full-scan contracts

---

### Task 1: Add Failing Workflow Contract Tests

**Files:**
- Modify: `tests/test_exec_scan.py`
- Modify: `tests/test_full_scan.py`

**Step 1: Write the failing test**

Add assertions that expect:
- `.github/workflows/interop-smoke.yml` to build artifacts
- the workflow to run `python scripts/interop_smoke.py --backend all`
- the workflow to run `python scripts/release_smoke.py --artifact-dir dist --kind wheel --interop-backend all`
- `full_scan.py` task descriptions to include the new interop-workflow automation checks

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/test_exec_scan.py tests/test_full_scan.py`
Expected: FAIL until the workflow and scan catalog are updated.

**Step 3: Write minimal implementation**

Only add the assertions needed to lock the new workflow contract.

**Step 4: Run targeted verification**

Run: `python -m pytest -q tests/test_exec_scan.py tests/test_full_scan.py`
Expected: PASS

### Task 2: Extend The Interop Workflow And Scan Catalog

**Files:**
- Modify: `.github/workflows/interop-smoke.yml`
- Modify: `scripts/full_scan.py`
- Modify: `scripts/contracts.py`

**Step 1: Write the failing test**

Ensure the new workflow/scan assertions from Task 1 are in place first.

**Step 2: Run tests to verify they fail**

Run: `python -m pytest -q tests/test_exec_scan.py tests/test_full_scan.py`
Expected: FAIL

**Step 3: Write minimal implementation**

Update the interop workflow so it:
- keeps the existing `pyg` and `dgl` checkout smoke jobs
- adds a combined job with both extras installed
- runs `python -m build`
- runs `python scripts/interop_smoke.py --backend all`
- runs `python scripts/release_smoke.py --artifact-dir dist --kind wheel --interop-backend all`

Update the scan catalog/contracts so the repository tracks the new automation commands.

**Step 4: Run targeted verification**

Run: `python -m pytest -q tests/test_exec_scan.py tests/test_full_scan.py`
Expected: PASS

### Task 3: Document Automated Artifact Interop Coverage

**Files:**
- Modify: `docs/releasing.md`
- Modify: `docs/support-matrix.md`
- Modify: `tests/test_release_packaging.py`

**Step 1: Write the failing assertion**

Add docs assertions that expect release docs/support guidance to mention:
- the interop workflow now covers artifact-level wheel smoke
- the combined workflow environment exercises the `all` success path

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/test_release_packaging.py`
Expected: FAIL until docs mention the new automation contract.

**Step 3: Write minimal implementation**

Document the automation without changing default CI cost:
- keep the workflow manual/nightly
- explain that workflow coverage now includes built-wheel artifact interop with both backends available
- preserve local/manual artifact guidance for maintainers

**Step 4: Run targeted verification**

Run: `python -m pytest -q tests/test_release_packaging.py`
Expected: PASS

## Final Verification

Run these commands before merging:

- `python -m pytest -q tests/test_exec_scan.py tests/test_full_scan.py tests/test_release_packaging.py`
- `python scripts/full_scan.py`
- `python -m ruff check .`
- `python -m mypy vgl`
- `python -m mkdocs build --strict`
- `python -m pytest -q`
