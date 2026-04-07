# Contract Script Import Hardening Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make contract-oriented repository scripts load `scripts.contracts` from the checkout reliably even when an unrelated `contracts.py` appears earlier on `sys.path`.

**Architecture:** Add a tiny per-script repo-module loader that normalizes repo-root import resolution and returns repo-local `scripts.*` modules even when the script is loaded by file path. Update the affected scripts to consume shared constants through that loader instead of fragile top-level imports, then lock the behavior in with regression tests that inject a shadow `contracts.py`.

**Tech Stack:** Python 3.10+, `importlib`, `sys.path`, pytest, repository CLI scripts

---

### Task 1: Add Failing Shadow-Import Regression Tests

**Files:**
- Modify: `tests/test_release_contract_scan.py`
- Modify: `tests/test_metadata_consistency.py`
- Modify: `tests/test_public_surface_scan.py`

**Step 1: Write the failing tests**

Add tests that:
- create a temporary shadow `contracts.py` with obviously wrong values
- prepend that temporary directory to `sys.path`
- load each target script as a module from file
- assert the script still exposes constants/specs from `scripts.contracts`, not the shadow module

**Step 2: Run tests to verify they fail**

Run: `python -m pytest -q tests/test_release_contract_scan.py tests/test_metadata_consistency.py tests/test_public_surface_scan.py`
Expected: FAIL while the scripts still resolve top-level `contracts` from ambient `sys.path`.

### Task 2: Inline Repo-Module Loaders And Update Scripts

**Files:**
- Modify: `scripts/release_contract_scan.py`
- Modify: `scripts/metadata_consistency.py`
- Modify: `scripts/public_surface_scan.py`

**Step 1: Write minimal implementation**

Add a tiny loader in each target script that:
- resolves the repo root from the current script location
- inserts that root at the front of `sys.path` deterministically
- imports and returns the needed `scripts.*` module

Update each target script to source its shared constants/spec builders through the loader while preserving all existing CLI behavior.

**Step 2: Run focused verification**

Run: `python -m pytest -q tests/test_release_contract_scan.py tests/test_metadata_consistency.py tests/test_public_surface_scan.py`
Expected: PASS

### Task 3: Re-verify Contract Scan Entry Points

**Files:**
- Verify only: `scripts/release_contract_scan.py`
- Verify only: `scripts/metadata_consistency.py`
- Verify only: `scripts/public_surface_scan.py`
- Verify only: `tests/test_release_packaging.py`
- Verify only: `tests/test_exec_scan.py`
- Verify only: `tests/test_full_scan.py`

**Step 1: Run targeted verification**

Run: `python -m pytest -q tests/test_release_contract_scan.py tests/test_metadata_consistency.py tests/test_public_surface_scan.py tests/test_release_packaging.py tests/test_exec_scan.py tests/test_full_scan.py`
Expected: PASS

**Step 2: Run repository scanners**

Run:
- `python scripts/metadata_consistency.py`
- `python scripts/public_surface_scan.py`

Expected: PASS

## Final Verification

Run these commands before merging:

- `python -m pytest -q tests/test_release_contract_scan.py tests/test_metadata_consistency.py tests/test_public_surface_scan.py tests/test_release_packaging.py tests/test_exec_scan.py tests/test_full_scan.py`
- `python -m ruff check .`
- `python scripts/full_scan.py`
