# Full Scan Loader Adoption Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `scripts/full_scan.py` adopt the shared repo script loader introduced in round16 instead of maintaining its own bespoke workflow-contract import bootstrap.

**Architecture:** Add one focused regression test proving `full_scan.py` reuses `scripts.repo_script_imports.load_repo_module`, then replace the current inline `_load_workflow_contracts_module()` implementation with the shared loader while preserving direct script execution and existing shadow-module protection.

**Tech Stack:** Python 3.10+, `importlib`, `sys.path`, pytest, repository scan scripts

---

### Task 1: Add Failing Shared-Loader Adoption Test

**Files:**
- Modify: `tests/test_full_scan.py`

**Step 1: Write the failing test**

Add a test that:
- loads `scripts/full_scan.py` via the existing file-loader helper
- imports `scripts.repo_script_imports`
- asserts the loaded `full_scan` module exposes and reuses the shared `load_repo_module` function

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/test_full_scan.py::test_full_scan_reuses_shared_repo_script_loader`
Expected: FAIL because `full_scan.py` still owns a bespoke loader.

### Task 2: Switch Full Scan To The Shared Loader

**Files:**
- Modify: `scripts/full_scan.py`

**Step 1: Write minimal implementation**

Update `scripts/full_scan.py` to:
- keep the minimal bootstrap needed for direct `python scripts/full_scan.py` execution
- import `load_repo_module` from `scripts.repo_script_imports`
- load `scripts.workflow_contracts` through that shared helper

**Step 2: Run focused verification**

Run: `python -m pytest -q tests/test_full_scan.py tests/test_exec_scan.py`
Expected: PASS

### Task 3: Re-verify Full Scan Entry Points

**Files:**
- Verify only: `scripts/full_scan.py`
- Verify only: `tests/test_release_packaging.py`

**Step 1: Run broader verification**

Run:
- `python -m pytest -q tests/test_full_scan.py tests/test_exec_scan.py tests/test_release_packaging.py`
- `python scripts/full_scan.py`

Expected: PASS

## Final Verification

Run these commands before merging:

- `python -m pytest -q`
- `python scripts/full_scan.py`
- `python -m ruff check .`
