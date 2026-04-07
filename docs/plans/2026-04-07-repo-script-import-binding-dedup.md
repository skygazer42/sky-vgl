## Round 21 Plan: Repo script import binding dedup

### Goal

Remove the repeated `importlib.import_module(...repo_script_imports...)` binding boilerplate from the script entrypoints while preserving direct execution and shared helper identity.

### Scope

Update:

- `repo_script_imports.py`
- `scripts/full_scan.py`
- `scripts/install_release_extras.py`
- `scripts/interop_smoke.py`
- `scripts/metadata_consistency.py`
- `scripts/public_surface_scan.py`
- `scripts/release_contract_scan.py`
- `scripts/release_smoke.py`
- `tests/test_repo_script_bootstrap.py`

### Tasks

1. Add a failing structural regression test that forbids the repeated `importlib.import_module(...)` binding boilerplate.
2. Introduce a repo-root shim import path so scripts can bind helpers via a direct `import repo_script_imports`.
3. Rebind the affected scripts to the new shared import path without changing behavior.
4. Re-run script-focused pytest coverage, `python scripts/full_scan.py`, and `python -m ruff check`.

### Notes

- Preserve direct execution from outside the repository root.
- Preserve shared helper object identity verified by the existing bootstrap tests.
