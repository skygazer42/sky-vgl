# Release Gates All-Backend Artifact Interop Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Promote combined optional-backend artifact interop smoke into the real CI and publish build gates so tagged releases prove `--interop-backend all` before upload.

**Architecture:** Keep the existing base package smoke steps, then add a narrowly scoped follow-up install of the optional interop extras in the workflow jobs that already build artifacts. Extend the release workflow/docs scans so the stronger gate is contractually enforced and documented.

**Tech Stack:** GitHub Actions YAML, pytest, repo scan helpers, release docs

---

### Task 1: Add Failing Assertions For The Stronger Release Gate

**Files:**
- Modify: `tests/test_release_packaging.py`
- Modify: `tests/test_exec_scan.py`
- Modify: `tests/test_full_scan.py`

**Step 1: Write the failing assertions**

Add assertions that expect:
- CI `package-check` to install optional interop extras before running the new artifact interop smoke
- Publish `build` to install optional interop extras before running the new artifact interop smoke
- the stronger `python scripts/release_smoke.py --artifact-dir dist --kind all --interop-backend all` command to be present in the protected workflow/doc scan surface

**Step 2: Run tests to verify they fail**

Run: `python -m pytest -q tests/test_release_packaging.py tests/test_exec_scan.py tests/test_full_scan.py`
Expected: FAIL until the workflows and scan catalog are updated.

### Task 2: Wire The Gate Into CI, Publish, And Repository Scans

**Files:**
- Modify: `.github/workflows/ci.yml`
- Modify: `.github/workflows/publish.yml`
- Modify: `scripts/full_scan.py`

**Step 1: Write minimal implementation**

Update the artifact-building jobs so they:
- keep the existing build / contract / metadata / baseline smoke flow
- add a dedicated optional-backend install step for `.[pyg,dgl]`
- run `python scripts/release_smoke.py --artifact-dir dist --kind all --interop-backend all`

Update `full_scan.py` so the workflow contract now checks for:
- the new optional-backend install steps in CI and publish build jobs
- the new all-backend artifact smoke command in CI and publish build jobs

**Step 2: Run targeted verification**

Run: `python -m pytest -q tests/test_release_packaging.py tests/test_exec_scan.py tests/test_full_scan.py`
Expected: PASS

### Task 3: Lock The Maintainer Contract In Release Docs

**Files:**
- Modify: `docs/releasing.md`
- Modify: `tests/test_release_packaging.py`

**Step 1: Write the failing assertion**

Add a docs assertion that expects the release doc to mention:
- CI / publish build jobs now run the combined all-backend artifact smoke gate
- that gate installs both optional interop extras before validating artifacts

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/test_release_packaging.py`
Expected: FAIL until the release doc describes the maintained workflow contract.

**Step 3: Write minimal implementation**

Document the contract narrowly:
- the package-check / publish build jobs still build and smoke-test the artifacts
- they now also install both optional interop extras and run the combined all-backend artifact smoke command before publish

**Step 4: Run targeted verification**

Run: `python -m pytest -q tests/test_release_packaging.py`
Expected: PASS

## Final Verification

Run these commands before merging:

- `python -m pytest -q tests/test_release_packaging.py tests/test_exec_scan.py tests/test_full_scan.py`
- `python scripts/full_scan.py`
- `python -m ruff check .`
- `python -m mypy vgl`
- `python -m mkdocs build --strict`
- `python -m pytest -q`
