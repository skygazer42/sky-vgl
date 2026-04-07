# Release Gate Job-Scoped Contract Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Harden release gate verification so CI/publish artifact interop checks are validated inside the intended workflow jobs instead of by whole-file substring matches.

**Architecture:** Add small workflow-text helpers for both tests and `scripts/full_scan.py` that extract job-specific blocks using YAML-style job anchors and indentation boundaries. Then migrate the new round7 release gate assertions to those helpers so the contract only passes when the commands remain in `package-check` and publish `build`.

**Tech Stack:** Python 3.10+, pytest, string parsing over GitHub Actions YAML, existing full-scan task catalog

---

### Task 1: Add Failing Job-Scoped Assertions

**Files:**
- Modify: `tests/test_release_packaging.py`
- Modify: `tests/test_exec_scan.py`
- Modify: `tests/test_full_scan.py`

**Step 1: Write the failing assertions**

Add tests that expect:
- the CI `package-check` job block, not just the whole file, to contain the `.[pyg,dgl]` install and `--interop-backend all` release smoke command
- the publish `build` job block, not just the whole file, to contain the same two commands
- `full_scan` to expose descriptions that explicitly mention the job-scoped contract

**Step 2: Run tests to verify they fail**

Run: `python -m pytest -q tests/test_release_packaging.py tests/test_exec_scan.py tests/test_full_scan.py`
Expected: FAIL because the tests and scan helpers still use whole-file substring checks.

### Task 2: Add Reusable Workflow Job Helpers And Migrate The Contract

**Files:**
- Modify: `scripts/full_scan.py`
- Modify: `tests/test_release_packaging.py`
- Modify: `tests/test_exec_scan.py`
- Modify: `tests/test_full_scan.py`

**Step 1: Write minimal implementation**

Implement the smallest useful helper seam:
- in `scripts/full_scan.py`, add a `ScanContext` helper that returns a named workflow job block and a task builder that checks for a snippet within that block
- in tests, add a local helper that extracts a workflow job block and reuse it for the round7 release-gate assertions
- replace the round7 all-backend release gate checks with the new job-scoped helpers while keeping unrelated workflow assertions unchanged

**Step 2: Run targeted verification**

Run: `python -m pytest -q tests/test_release_packaging.py tests/test_exec_scan.py tests/test_full_scan.py`
Expected: PASS

### Task 3: Re-verify Repository Contracts

**Files:**
- Modify: `scripts/full_scan.py`
- Modify: `tests/test_full_scan.py`

**Step 1: Run scan verification**

Run: `python scripts/full_scan.py`
Expected: PASS with the new job-scoped task descriptions listed and passing.

**Step 2: Run broader verification**

Run:
- `python -m ruff check .`
- `python -m mypy vgl`
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
