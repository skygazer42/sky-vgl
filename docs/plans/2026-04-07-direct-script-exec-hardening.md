## Round 19 Plan: Direct script execution hardening

### Goal

Lock in the centralized bootstrap behavior from round18 with subprocess coverage that runs repo scripts by absolute path from outside the repository root.

### Scope

Add direct-execution regression coverage for these entrypoints:

- `scripts/full_scan.py`
- `scripts/metadata_consistency.py`
- `scripts/public_surface_scan.py`
- `scripts/interop_smoke.py`
- `scripts/release_contract_scan.py`
- `scripts/install_release_extras.py`
- `scripts/release_smoke.py`

### Tasks

1. Add a reusable subprocess helper that executes scripts from a temporary external working directory.
2. Cover list/help style entrypoints that exercise bootstrap imports without requiring heavy fixture setup.
3. Keep assertions narrow to stable CLI markers so the tests protect execution behavior without becoming brittle.
4. Run targeted script-related tests and `python scripts/full_scan.py`.

### Notes

- This round is test-only; it should not change production script behavior.
- Prefer absolute script paths plus external `cwd` to exercise the bootstrap path that normal repo-root subprocess tests miss.
