# Release Artifact Metadata Helpers Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove duplicated wheel `METADATA` parsing logic from release scripts by introducing a shared helper module used by both `install_release_extras.py` and `release_contract_scan.py`.

**Architecture:** Add a small `scripts/release_artifact_metadata.py` helper that owns wheel `METADATA` extraction and detail reporting. Keep script-level behavior unchanged by wrapping the shared helper where each CLI needs different failure semantics, then cover the helper directly in tests while relying on existing release-contract and packaging tests for regression protection.

**Tech Stack:** Python 3.10+, wheel zip metadata parsing, pytest, Hatch build artifacts

---

### Task 1: Add Failing Shared-Helper Tests

**Files:**
- Modify: `tests/test_release_packaging.py`
- Modify: `tests/test_release_contract_scan.py`

**Step 1: Write the failing tests**

Add assertions that expect:
- a shared `scripts.release_artifact_metadata` module exists and can read wheel `METADATA`
- helper output includes the same `Provides-Extra` and `Requires-Dist` information already asserted by packaging tests
- helper returns a stable failure detail when a wheel-like archive lacks `METADATA`

**Step 2: Run tests to verify they fail**

Run: `python -m pytest -q tests/test_release_contract_scan.py tests/test_release_packaging.py`
Expected: FAIL until the shared helper module exists and the scripts are updated.

### Task 2: Extract Shared Wheel Metadata Logic

**Files:**
- Create: `scripts/release_artifact_metadata.py`
- Modify: `scripts/install_release_extras.py`
- Modify: `scripts/release_contract_scan.py`

**Step 1: Write minimal implementation**

Add a shared helper that:
- reads wheel `METADATA` from a built wheel archive
- returns `(metadata, detail)` so scanners can keep rich failure output

Update `install_release_extras.py` to reuse the helper while preserving current CLI exit behavior.

Update `release_contract_scan.py` to reuse the same helper inside `ArtifactContext.wheel_metadata()`.

**Step 2: Run focused verification**

Run: `python -m pytest -q tests/test_release_contract_scan.py tests/test_release_packaging.py`
Expected: PASS

### Task 3: Re-verify Release Contract Flows

**Files:**
- Verify only: `scripts/release_contract_scan.py`
- Verify only: `scripts/install_release_extras.py`
- Verify only: `tests/test_exec_scan.py`
- Verify only: `tests/test_full_scan.py`

**Step 1: Run targeted verification**

Run: `python -m pytest -q tests/test_release_contract_scan.py tests/test_release_packaging.py tests/test_exec_scan.py tests/test_full_scan.py`
Expected: PASS

**Step 2: Run scanner verification**

Run:
- `tmpdir=$(mktemp -d /tmp/vgl-round12-XXXXXX)`
- `python -m build --outdir "$tmpdir"`
- `python scripts/release_contract_scan.py --artifact-dir "$tmpdir"`

Expected: PASS

## Final Verification

Run these commands before merging:

- `python -m pytest -q tests/test_release_contract_scan.py tests/test_release_packaging.py tests/test_exec_scan.py tests/test_full_scan.py`
- `python -m ruff check .`
- `python scripts/full_scan.py`
