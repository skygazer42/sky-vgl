# Workflow Contract Utils Consolidation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Consolidate workflow job extraction into one shared helper so release-gate contract checks in tests and `full_scan` cannot drift apart.

**Architecture:** Move the indentation-based workflow parsing logic into a small reusable module under `scripts/`, then update `scripts/full_scan.py` and the workflow-oriented tests to import that shared helper instead of maintaining duplicate parsing code. Preserve the current behavior and synthetic regression coverage while shrinking the maintenance surface.

**Tech Stack:** Python 3.10+, pytest, existing `scripts/` utilities, workflow text parsing

---

### Task 1: Add Failing Shared-Helper Expectations

**Files:**
- Modify: `tests/test_exec_scan.py`
- Modify: `tests/test_full_scan.py`
- Modify: `tests/test_release_packaging.py`

**Step 1: Write the failing expectations**

Add assertions that expect:
- workflow job extraction to come from a shared non-test helper module
- release packaging / exec-scan tests to keep passing through that shared helper
- `full_scan` to continue exposing the same job-scoped contract descriptions after the refactor

**Step 2: Run tests to verify they fail**

Run: `python -m pytest -q tests/test_release_packaging.py tests/test_exec_scan.py tests/test_full_scan.py`
Expected: FAIL until the shared helper exists and both call sites are migrated.

### Task 2: Extract The Shared Workflow Contract Helper

**Files:**
- Create: `scripts/workflow_contracts.py`
- Modify: `scripts/full_scan.py`
- Modify: `tests/test_exec_scan.py`
- Modify: `tests/test_release_packaging.py`
- Delete: `tests/workflow_helpers.py`

**Step 1: Write minimal implementation**

Implement a shared helper module that provides:
- workflow job text extraction anchored to `jobs:`
- stable errors for missing `jobs:` or missing job names
- a small containment helper that `full_scan` can reuse directly

Update both test modules and `full_scan.py` to import that shared helper and remove the duplicated logic from `ScanContext` / test-only helpers.

**Step 2: Run targeted verification**

Run: `python -m pytest -q tests/test_release_packaging.py tests/test_exec_scan.py tests/test_full_scan.py`
Expected: PASS

### Task 3: Re-verify Full Repository Contracts

**Files:**
- Modify: `scripts/full_scan.py`
- Modify: `tests/test_full_scan.py`

**Step 1: Run scan verification**

Run: `python scripts/full_scan.py`
Expected: PASS with the same 116-task contract surface intact.

**Step 2: Run broader verification**

Run:
- `python -m ruff check .`
- `python -m mypy vgl`
- `python -m mkdocs build --strict`
- `python -m pytest -q`

Expected: PASS

## Final Verification

Run these commands before merging:

- `python -m pytest -q tests/test_release_packaging.py tests/test_exec_scan.py tests/test_full_scan.py`
- `python scripts/full_scan.py`
- `python -m ruff check .`
- `python -m mypy vgl`
- `python -m mkdocs build --strict`
- `python -m pytest -q`
