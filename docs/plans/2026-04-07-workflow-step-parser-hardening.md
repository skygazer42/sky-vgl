# Workflow Step Parser Hardening Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make workflow step parsing resilient to quoted step names, inline comments, and trailing spaces, while ensuring negative step checks fail when the target step is missing.

**Architecture:** Extend `scripts/workflow_contracts.py` with step-header matching logic parallel to the existing job-header helpers, then tighten `scripts/full_scan.py` negative step checks to distinguish "step missing" from "snippet absent". Cover both behaviors with focused parser and scan tests before merging.

**Tech Stack:** Python 3.10+, regex-based workflow text parsing, pytest

---

### Task 1: Add Failing Step Parser Tests

**Files:**
- Modify: `tests/test_exec_scan.py`
- Modify: `tests/test_full_scan.py`

**Step 1: Write the failing tests**

Add tests that expect:
- `workflow_step_text()` can find a named step when the `- name:` line uses quotes, trailing spaces, or an inline comment
- `_workflow_step_lacks_task`-backed checks fail when the named step is missing instead of reporting success

**Step 2: Run tests to verify they fail**

Run: `python -m pytest -q tests/test_exec_scan.py tests/test_full_scan.py`
Expected: FAIL until the parser and negative helper are tightened.

### Task 2: Harden Step Matching And Negative Semantics

**Files:**
- Modify: `scripts/workflow_contracts.py`
- Modify: `scripts/full_scan.py`

**Step 1: Write minimal implementation**

Update `scripts/workflow_contracts.py` to:
- match named workflow steps with quoted or unquoted step names
- tolerate inline comments and trailing whitespace on step name headers
- expose a negative helper that fails when the named step is missing

Update `scripts/full_scan.py` to:
- route negative step checks through the shared negative helper
- keep positive step checks unchanged

**Step 2: Run focused verification**

Run: `python -m pytest -q tests/test_exec_scan.py tests/test_full_scan.py`
Expected: PASS

### Task 3: Re-verify Workflow Contract Scans

**Files:**
- Verify only: `scripts/full_scan.py`
- Verify only: `tests/test_release_packaging.py`

**Step 1: Run targeted verification**

Run: `python -m pytest -q tests/test_release_packaging.py tests/test_exec_scan.py tests/test_full_scan.py`
Expected: PASS

**Step 2: Run scanner verification**

Run: `python scripts/full_scan.py`
Expected: PASS

## Final Verification

Run these commands before merging:

- `python -m pytest -q tests/test_release_packaging.py tests/test_exec_scan.py tests/test_full_scan.py`
- `python -m ruff check .`
- `python scripts/full_scan.py`
