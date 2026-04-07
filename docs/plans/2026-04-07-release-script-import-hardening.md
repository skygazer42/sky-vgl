# Release Script Import Hardening Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make release-oriented scripts load repo-local `scripts.contracts` and `scripts.release_artifact_metadata` reliably even when a shadow module appears earlier on `sys.path`.

**Architecture:** Add shadow-import regression coverage for `install_release_extras.py`, `interop_smoke.py`, and `release_smoke.py`, then harden those scripts with the same repo-root insertion pattern already used in `full_scan.py` and the round13 contract scanners. Keep the implementation local to these scripts to avoid broader packaging churn this round.

**Tech Stack:** Python 3.10+, `importlib`, `sys.path`, pytest, release CLI scripts

---

### Task 1: Add Failing Shadow-Import Regression Tests

**Files:**
- Modify: `tests/test_interop_smoke.py`
- Modify: `tests/test_release_smoke.py`
- Modify: `tests/test_release_packaging.py`

**Step 1: Write the failing tests**

Add tests that:
- load `scripts/interop_smoke.py` from file with a fake `contracts.py` earlier on `sys.path`
- load `scripts/release_smoke.py` from file with a fake `contracts.py` earlier on `sys.path`
- load `scripts/install_release_extras.py` from file with a fake `release_artifact_metadata.py` earlier on `sys.path`
- assert each module still binds repo-local values/functions from `scripts.*`

**Step 2: Run tests to verify they fail**

Run: `python -m pytest -q tests/test_interop_smoke.py tests/test_release_smoke.py tests/test_release_packaging.py`
Expected: FAIL while the scripts still rely on shadowable top-level imports.

### Task 2: Harden Release Script Repo-Local Imports

**Files:**
- Modify: `scripts/install_release_extras.py`
- Modify: `scripts/interop_smoke.py`
- Modify: `scripts/release_smoke.py`

**Step 1: Write minimal implementation**

Update each target script to:
- resolve the repo root from `__file__`
- place that repo root at the front of `sys.path`
- import the needed `scripts.*` module through a deterministic repo-local path

Keep existing CLI behavior unchanged.

**Step 2: Run focused verification**

Run: `python -m pytest -q tests/test_interop_smoke.py tests/test_release_smoke.py tests/test_release_packaging.py`
Expected: PASS

### Task 3: Re-verify Release Script Entry Points

**Files:**
- Verify only: `scripts/install_release_extras.py`
- Verify only: `scripts/interop_smoke.py`
- Verify only: `scripts/release_smoke.py`

**Step 1: Run broader verification**

Run: `python -m pytest -q tests/test_interop_smoke.py tests/test_release_smoke.py tests/test_release_packaging.py tests/test_release_contract_scan.py`
Expected: PASS

**Step 2: Run full repository verification**

Run:
- `python -m pytest -q`
- `python scripts/full_scan.py`
- `python -m ruff check .`

Expected: PASS

## Final Verification

Run these commands before merging:

- `python -m pytest -q`
- `python scripts/full_scan.py`
- `python -m ruff check .`
