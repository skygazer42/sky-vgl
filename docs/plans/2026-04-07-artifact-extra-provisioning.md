# Artifact Extra Provisioning From Built Metadata Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the all-backend release gate provision optional backend dependencies from built artifact metadata instead of editable checkout extras.

**Architecture:** Add a small script that reads wheel metadata, resolves the concrete `Requires-Dist` entries activated by selected extras in the current environment, and optionally installs those requirements with `pip`. Update CI/publish release gates and their contract tests to use this artifact-derived provisioning path before running `release_smoke.py --interop-backend all`.

**Tech Stack:** Python 3.10+, wheel metadata parsing, `packaging.requirements`, pytest, GitHub Actions

---

### Task 1: Add Failing Contract Tests For Artifact-Derived Extra Provisioning

**Files:**
- Modify: `tests/test_release_packaging.py`
- Modify: `tests/test_exec_scan.py`
- Modify: `tests/test_full_scan.py`

**Step 1: Write the failing assertions**

Add assertions that expect:
- CI `package-check` and publish `build` to call a new artifact extra provisioning script instead of `python -m pip install -e ".[pyg,dgl]"`
- the release-gate contract descriptions in `full_scan` to reflect artifact-derived provisioning

**Step 2: Run tests to verify they fail**

Run: `python -m pytest -q tests/test_release_packaging.py tests/test_exec_scan.py tests/test_full_scan.py`
Expected: FAIL until the provisioning script and workflow changes exist.

### Task 2: Add The Artifact Extra Provisioning Script

**Files:**
- Create: `scripts/install_release_extras.py`
- Modify: `tests/test_release_packaging.py`

**Step 1: Write the failing test**

Add tests that expect the script to:
- read a built wheel
- resolve the `pyg` and `dgl` extra requirements that apply in the current environment
- print those requirements in pip-installable form

**Step 2: Run the focused test to verify it fails**

Run: `python -m pytest -q tests/test_release_packaging.py`
Expected: FAIL until the script exists.

**Step 3: Write minimal implementation**

Implement a script that:
- accepts `--artifact-dir <dir>` and one or more `--extras`
- finds the built wheel and parses `Requires-Dist` entries from the wheel metadata
- evaluates markers for the requested extras in the current environment
- supports `--print-only` for deterministic tests
- otherwise installs the resolved third-party requirements through `pip`

**Step 4: Run targeted verification**

Run: `python -m pytest -q tests/test_release_packaging.py`
Expected: PASS

### Task 3: Wire The Release Gates To Artifact Metadata Provisioning

**Files:**
- Modify: `.github/workflows/ci.yml`
- Modify: `.github/workflows/publish.yml`
- Modify: `scripts/full_scan.py`
- Modify: `docs/releasing.md`
- Modify: `tests/test_release_packaging.py`
- Modify: `tests/test_exec_scan.py`
- Modify: `tests/test_full_scan.py`

**Step 1: Write minimal implementation**

Update the release-gate jobs so they:
- keep building and scanning artifacts as before
- run `python scripts/install_release_extras.py --artifact-dir dist --extras pyg dgl`
- then run the all-backend artifact smoke gate

Update docs and `full_scan` descriptions to match the maintained contract.

**Step 2: Run targeted verification**

Run: `python -m pytest -q tests/test_release_packaging.py tests/test_exec_scan.py tests/test_full_scan.py`
Expected: PASS

## Final Verification

Run these commands before merging:

- `python -m pytest -q tests/test_release_packaging.py tests/test_exec_scan.py tests/test_full_scan.py`
- `python scripts/full_scan.py`
- `python -m ruff check .`
- `python -m mypy vgl`
- `python -m mkdocs build --strict`
- `python -m pytest -q`
