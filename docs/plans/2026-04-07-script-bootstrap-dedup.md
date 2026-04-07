## Round 18 Plan: Script bootstrap dedup

### Goal

Centralize repo-root bootstrap behavior in `scripts/repo_script_imports.py` so script entrypoints stop repeating inline `sys.path` manipulation.

### Scope

Update these scripts to use a shared loader/bootstrap import path:

- `scripts/full_scan.py`
- `scripts/release_contract_scan.py`
- `scripts/public_surface_scan.py`
- `scripts/install_release_extras.py`
- `scripts/metadata_consistency.py`
- `scripts/release_smoke.py`
- `scripts/interop_smoke.py`

### Tasks

1. Add a structural regression test proving bootstrap logic is centralized in `scripts/repo_script_imports.py`.
2. Replace inline repo-root bootstrap blocks with a shared `repo_script_imports` module binding strategy that works for both direct script execution and in-process test loading.
3. Rebind `interop_smoke.py` to the shared `ensure_repo_root_on_path()` helper instead of maintaining a local copy.
4. Run targeted script tests, `python scripts/full_scan.py`, and `python -m ruff check .`.

### Notes

- Keep direct `python scripts/<name>.py` execution working.
- Avoid broad behavior changes beyond loader/bootstrap consolidation.
