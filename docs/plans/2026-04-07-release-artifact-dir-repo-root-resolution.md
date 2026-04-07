## Round 19 Plan: Repo-relative artifact-dir resolution

### Goal

Make release helper scripts resolve relative `--artifact-dir` values from the repository root so direct execution outside the repo root remains usable beyond `--help`.

### Scope

Update these scripts and their tests:

- `scripts/release_contract_scan.py`
- `scripts/install_release_extras.py`
- `scripts/release_smoke.py`
- `scripts/repo_script_imports.py`
- `tests/test_release_contract_scan.py`
- `tests/test_release_packaging.py`
- `tests/test_direct_script_execution.py`

### Tasks

1. Add failing subprocess tests that run the release scripts from outside the repo root while passing repo-relative artifact directories.
2. Centralize repo-relative path resolution in `scripts/repo_script_imports.py`.
3. Rebind the release scripts to the shared path resolver so explicit relative artifact directories are anchored to the repo root instead of the caller's cwd.
4. Run targeted release-script tests, `python scripts/full_scan.py`, and `python -m ruff check .`.

### Notes

- Preserve absolute-path handling.
- Keep `release_contract_scan.py --repo-root` semantics intact by resolving relative artifact paths against the selected repo root.
- Keep the broader outside-repo direct-execution behavior covered for representative scripts.
