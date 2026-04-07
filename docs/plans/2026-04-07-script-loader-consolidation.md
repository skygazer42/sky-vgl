# Script Loader Consolidation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace duplicated repo-local script import bootstrapping with a shared helper while preserving direct script execution, file-loader tests, and release sdist completeness.

**Architecture:** Add a small `scripts/repo_script_imports.py` helper that owns repo-root insertion and `scripts.*` module loading. Update the current consumers to use that helper through a tiny bootstrap that keeps direct `python scripts/<name>.py` execution working, then extend packaging expectations so the helper ships anywhere release scripts depend on it.

**Tech Stack:** Python 3.10+, `importlib`, `sys.path`, pytest, Hatch sdist packaging

---

### Task 1: Add Failing Consolidation Coverage

**Files:**
- Modify: `tests/test_release_packaging.py`
- Modify: `tests/test_release_contract_scan.py`

**Step 1: Write the failing tests**

Add assertions that expect:
- a shared `scripts.repo_script_imports` module exists and can load repo-local modules
- built sdists include `/scripts/repo_script_imports.py`
- release contract scan catalog includes the new required sdist helper path

**Step 2: Run tests to verify they fail**

Run: `python -m pytest -q tests/test_release_packaging.py tests/test_release_contract_scan.py`
Expected: FAIL until the helper exists and packaging expectations are updated.

### Task 2: Extract Shared Script Loader And Update Consumers

**Files:**
- Create: `scripts/repo_script_imports.py`
- Modify: `scripts/metadata_consistency.py`
- Modify: `scripts/public_surface_scan.py`
- Modify: `scripts/release_contract_scan.py`
- Modify: `scripts/install_release_extras.py`
- Modify: `scripts/interop_smoke.py`
- Modify: `scripts/release_smoke.py`

**Step 1: Write minimal implementation**

Add a helper that:
- computes the repo root from the helper module location
- inserts that root at the front of `sys.path`
- imports and returns repo-local `scripts.*` modules

Update each current `_load_repo_module()` consumer to:
- bootstrap repo-root access only enough to import the shared helper
- delegate target-module loading to `scripts.repo_script_imports.load_repo_module`

**Step 2: Run focused verification**

Run: `python -m pytest -q tests/test_release_packaging.py tests/test_release_contract_scan.py tests/test_metadata_consistency.py tests/test_public_surface_scan.py tests/test_interop_smoke.py tests/test_release_smoke.py`
Expected: PASS

### Task 3: Re-verify Release Packaging And Script Contracts

**Files:**
- Modify: `pyproject.toml`
- Modify: `scripts/contracts.py`
- Verify only: `scripts/full_scan.py`

**Step 1: Update packaging contract**

Add the shared helper to:
- `tool.hatch.build.targets.sdist.include`
- `SDIST_REQUIRED_SUFFIXES`

**Step 2: Run broader verification**

Run:
- `python -m pytest -q tests/test_release_packaging.py tests/test_release_contract_scan.py tests/test_metadata_consistency.py tests/test_public_surface_scan.py tests/test_full_scan.py tests/test_interop_smoke.py tests/test_release_smoke.py`
- `python scripts/full_scan.py`

Expected: PASS

## Final Verification

Run these commands before merging:

- `python -m pytest -q`
- `python scripts/full_scan.py`
- `python -m ruff check .`
