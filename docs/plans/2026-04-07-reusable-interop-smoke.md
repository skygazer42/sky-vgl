# Reusable Interop Smoke Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a reusable optional-backend smoke script for DGL and PyG, then wire workflows, release contracts, docs, and tests to keep that path stable.

**Architecture:** Introduce one small CLI in `scripts/interop_smoke.py` that exercises the public `vgl.compat` and `Graph.to_*` / `Graph.from_*` interop paths against an installed backend. Keep the runtime logic self-contained and lightweight, then point the manual/nightly workflow, Makefile, release docs, and repository scans at that single script so there is exactly one maintained smoke path.

**Tech Stack:** Python 3.10+, pytest, GitHub Actions, MkDocs, existing VGL compat adapters, existing release-quality scan scripts.

---

### Task 1: Add the reusable interop smoke CLI

**Files:**
- Create: `scripts/interop_smoke.py`
- Modify: `scripts/contracts.py`
- Test: `tests/test_interop_smoke.py`

**Step 1: Write the failing test**

Add CLI coverage for:
- `python scripts/interop_smoke.py --list-backends`
- direct module-level round-trip helpers using fake PyG and fake DGL modules

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/test_interop_smoke.py`
Expected: FAIL because `scripts/interop_smoke.py` does not exist yet.

**Step 3: Write minimal implementation**

Create a script that:
- exposes `--backend {pyg,dgl,all}` and `--list-backends`
- builds one tiny homogeneous `Graph`
- runs round-trip checks through the public compat API
- prints a stable success line per backend
- uses constants from `scripts/contracts.py` for supported interop backends

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/test_interop_smoke.py`
Expected: PASS

### Task 2: Rewire workflow, release contract, and local automation

**Files:**
- Modify: `.github/workflows/interop-smoke.yml`
- Modify: `Makefile`
- Modify: `scripts/contracts.py`
- Modify: `scripts/full_scan.py`
- Test: `tests/test_exec_scan.py`
- Test: `tests/test_release_packaging.py`

**Step 1: Write/update failing tests**

Add checks that:
- `interop-smoke.yml` runs `python scripts/interop_smoke.py --backend pyg` and `--backend dgl`
- `Makefile` exposes an `interop-smoke` target
- release packaging/scan expectations include `scripts/interop_smoke.py`

**Step 2: Run focused tests to verify failure**

Run: `python -m pytest -q tests/test_exec_scan.py tests/test_release_packaging.py`
Expected: FAIL on missing script/target/scan expectations.

**Step 3: Write minimal implementation**

Update the workflow, Makefile, and scan contract so that the reusable CLI becomes the single maintained interop smoke entry point.

**Step 4: Run focused tests to verify success**

Run: `python -m pytest -q tests/test_exec_scan.py tests/test_release_packaging.py`
Expected: PASS

### Task 3: Align docs with the reusable smoke path

**Files:**
- Modify: `docs/releasing.md`
- Modify: `docs/support-matrix.md`
- Modify: `docs/getting-started/installation.md`
- Test: `tests/test_release_packaging.py`
- Test: `tests/test_exec_scan.py`

**Step 1: Write/update failing tests**

Add assertions that the docs mention the reusable interop smoke command where appropriate.

**Step 2: Run focused tests to verify failure**

Run: `python -m pytest -q tests/test_exec_scan.py tests/test_release_packaging.py`
Expected: FAIL until the docs are updated.

**Step 3: Write minimal implementation**

Document:
- the reusable script name
- how maintainers run backend smoke locally
- that support matrix verification comes from the scheduled/manual workflow invoking the script

**Step 4: Run focused tests to verify success**

Run: `python -m pytest -q tests/test_exec_scan.py tests/test_release_packaging.py`
Expected: PASS

### Task 4: Verify the Round 2 slice end-to-end

**Files:**
- No new files

**Step 1: Run targeted verification**

Run:
- `python -m pytest -q tests/test_interop_smoke.py tests/test_exec_scan.py tests/test_release_packaging.py`
- `python scripts/full_scan.py`
- `python -m ruff check .`

Expected: PASS

**Step 2: Run broader verification**

Run:
- `python -m pytest -q`
- `python -m mkdocs build --strict`
- `python -m mypy vgl`

Expected: PASS

**Step 3: Commit**

```bash
git add scripts/interop_smoke.py scripts/contracts.py .github/workflows/interop-smoke.yml Makefile docs/releasing.md docs/support-matrix.md docs/getting-started/installation.md tests/test_interop_smoke.py tests/test_exec_scan.py tests/test_release_packaging.py scripts/full_scan.py docs/plans/2026-04-07-reusable-interop-smoke.md
git commit -m "Add reusable interop smoke coverage"
```
