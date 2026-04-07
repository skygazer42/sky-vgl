# Release Contract Extra Metadata Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend artifact release contract checks so built wheel metadata must declare the optional backend requirements consumed by `scripts/install_release_extras.py`.

**Architecture:** Keep the source of truth in `scripts/contracts.py`, then teach `scripts/release_contract_scan.py` to assert those expected `Requires-Dist` metadata lines on built wheels. Lock the behavior with packaging-level tests so release verification fails before CI workflow smoke reaches an under-declared artifact.

**Tech Stack:** Python 3.10+, stdlib `email` metadata parsing, pytest, existing release contract scan helpers

---

### Task 1: Add failing metadata contract tests

**Files:**
- Modify: `tests/test_release_packaging.py`
- Modify: `tests/test_release_contract_scan.py`

**Step 1: Write the failing packaging test**

Add a test that builds release artifacts, loads wheel `METADATA`, and asserts the exact `Requires-Dist` lines for the release interop extras exist:
- `torch-geometric>=2.5; extra == 'pyg'`
- `dgl>=2.1; extra == 'dgl'`

**Step 2: Run the targeted test to verify it fails**

Run: `python -m pytest -q tests/test_release_packaging.py -k metadata_declares_release_interop_extra_requirements`
Expected: FAIL until the new assertion exists or the release metadata contract is wired.

**Step 3: Write the failing release-contract-scan test**

Extend the expected scan catalog count to include the new metadata tasks and keep the pass-on-built-artifacts assertion aligned with the expanded catalog.

**Step 4: Run the scan tests to verify they fail**

Run: `python -m pytest -q tests/test_release_contract_scan.py`
Expected: FAIL because `release_contract_scan.py` does not yet validate those metadata lines.

### Task 2: Teach release contract scan about extra metadata

**Files:**
- Modify: `scripts/contracts.py`
- Modify: `scripts/release_contract_scan.py`
- Test: `tests/test_release_contract_scan.py`

**Step 1: Add shared contract data**

Define one shared constant in `scripts/contracts.py` for the exact wheel `Requires-Dist` lines that must exist for release interop extras.

**Step 2: Implement minimal scan task support**

Add a helper in `scripts/release_contract_scan.py` that checks whether a specific `Requires-Dist` line is present in wheel metadata, then append tasks for each required extra metadata line using the shared contract constant.

**Step 3: Run focused tests**

Run: `python -m pytest -q tests/test_release_contract_scan.py tests/test_release_packaging.py`
Expected: PASS with the new metadata tasks included.

### Task 3: Verify the round

**Files:**
- Modify: none

**Step 1: Run focused release verification**

Run: `python -m pytest -q tests/test_release_contract_scan.py tests/test_release_packaging.py tests/test_exec_scan.py tests/test_full_scan.py`
Expected: PASS

**Step 2: Run repo scans**

Run: `python scripts/release_contract_scan.py --artifact-dir <fresh-dist-dir>`
Expected: PASS on a freshly built artifact directory

Run: `python scripts/full_scan.py`
Expected: PASS

**Step 3: Commit**

```bash
git add scripts/contracts.py scripts/release_contract_scan.py tests/test_release_contract_scan.py tests/test_release_packaging.py docs/plans/2026-04-07-release-contract-extra-metadata.md
git commit -m "Harden release contract extra metadata"
```
