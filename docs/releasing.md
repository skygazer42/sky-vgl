# Releasing VGL

## Pre-release checklist

1. Confirm the intended public version in `vgl/version.py`.
2. Run the metadata consistency checker before building artifacts:

```bash
python scripts/metadata_consistency.py
```

3. Re-check that the sky-vgl project name is still available or that you control the existing PyPI project.
4. Run the full local verification suite:

```bash
python -m pytest -q
python -m build
python scripts/release_contract_scan.py --artifact-dir dist
python -m twine check dist/*.whl dist/*.tar.gz
python scripts/release_smoke.py --artifact-dir dist --kind all --max-import-seconds 30
```

5. Confirm the smoke script reports both wheel and sdist installs passing from
outside the repository root. The script creates isolated virtual environments,
installs the built artifacts, and verifies `import vgl` plus the golden-path
symbols from the release contract (`Graph`, `Trainer`, `PlanetoidDataset`,
`NodeClassificationTask`) resolve from the installed distribution instead of the
checkout. The smoke output also includes per-module `IMPORT_TIMING` lines and an
`IMPORT_BUDGET_OK` line for the root `vgl` import when the configured import-time
budget passes.
6. Run optional backend interop smoke checks in an environment where the extras
are installed. The reusable checkout-level script is the same one used by the
manual/nightly workflow. Artifact-level interop smoke reuses host-installed
torch and optional backend packages from the outer `site-packages` while still
verifying that `vgl` resolves from the installed wheel instead of the checkout.
Before the inline interop checks run, `scripts/release_smoke.py` probes
the same host `site-packages` and will exit with a clear error if any requested
backend is unavailable. `--interop-backend all` therefore requires both host backends,
PyG and DGL, to be importable in the host environment; missing support causes the
command to fail early with a descriptive message instead of a cryptic import traceback.

```bash
python scripts/interop_smoke.py --list-backends
python scripts/interop_smoke.py --backend pyg
python scripts/interop_smoke.py --backend dgl
python scripts/interop_smoke.py --backend all  # only when both extras are installed
# Optional: verify interop through installed wheel artifacts (not checkout imports)
python scripts/release_smoke.py --artifact-dir dist --kind wheel --interop-backend dgl --max-import-seconds 30
```
The manual/nightly `interop-smoke` workflow now also installs both optional
backends, builds the artifacts, and runs
`python scripts/interop_smoke.py --backend all` followed by
`python scripts/release_smoke.py --artifact-dir dist --kind wheel --interop-backend all`
so automation continuously exercises the artifact-level success path for both
backends.
The CI `package-check` job and the publish `build` job also install
`python scripts/install_release_extras.py --artifact-dir dist --extras pyg dgl` before running
`python scripts/release_smoke.py --artifact-dir dist --kind all --interop-backend all`,
so the real release gate provisions optional backends from built wheel metadata
before proving combined artifact interop.

7. Draft release notes from the changelog and recent commits before tagging:

```bash
git log --oneline --decorate <previous-tag>..HEAD
```

## Workflow map

Use this table when you need to map a failing workflow step back to the exact local command and expected artifact:

| Workflow job | What it proves | Expected artifact/output | Local replay |
| --- | --- | --- | --- |
| `ci.yml -> package-check` | The branch can build releasable wheel/sdist artifacts, validate metadata, and pass combined artifact smoke | `dist/*.whl`, `dist/*.tar.gz` | `python -m build && python scripts/release_contract_scan.py --artifact-dir dist && python -m twine check dist/*.whl dist/*.tar.gz && python scripts/release_smoke.py --artifact-dir dist --kind all --interop-backend all` |
| `publish.yml -> build` | The real publish gate can rebuild artifacts, provision release extras from metadata, and produce the upload bundle | uploaded artifact `release-dists` | `python scripts/metadata_consistency.py --git-ref "${GITHUB_REF}" && python -m build && python scripts/install_release_extras.py --artifact-dir dist --extras pyg dgl && python scripts/release_smoke.py --artifact-dir dist --kind all --interop-backend all` |
| `interop-smoke.yml -> backend-smoke` | A single real backend install can round-trip through the checkout surface | console output ending in `pyg interop smoke passed` or `dgl interop smoke passed` | `python scripts/interop_smoke.py --backend pyg` or `python scripts/interop_smoke.py --backend dgl` |
| `interop-smoke.yml -> all-artifact-smoke` | Both backends can round-trip through built artifacts, not just the checkout | built `dist/*` plus successful all-backend smoke output | `python -m build && python scripts/interop_smoke.py --backend all && python scripts/release_smoke.py --artifact-dir dist --kind wheel --interop-backend all` |
| `ci.yml -> benchmark` | Runtime-sensitive changes still produce a benchmark artifact in CI format | uploaded artifact `benchmark-hotpaths` containing `benchmark-hotpaths.json` | `python scripts/benchmark_hotpaths.py --preset ci --output artifacts/benchmark-hotpaths.json --print` |

## Release failure triage

Use the failing workflow and step name to narrow the fix path before rerunning anything:

- `ci.yml -> package-check` covers artifact build, metadata validation, release smoke, and all-backend artifact interop
- `interop-smoke.yml -> backend-smoke` covers single-backend checkout interop smoke
- `interop-smoke.yml -> all-artifact-smoke` covers combined built-wheel backend validation
- `publish.yml -> build` is the real publishing gate and should only be retried after the equivalent local commands pass

Recommended order:

1. Re-run the exact failing command locally.
2. Preserve the smoke output or benchmark artifact in the PR or issue.
3. If the failure is backend-specific, compare `python scripts/interop_smoke.py --backend <name>` with `python scripts/release_smoke.py --artifact-dir dist --kind wheel --interop-backend <name>`.
4. If the failure is packaging-specific, inspect `python scripts/release_contract_scan.py --artifact-dir dist` and `python -m twine check dist/*.whl dist/*.tar.gz` before rebuilding.

## Issue intake

The repository now ships focused GitHub issue templates under `.github/ISSUE_TEMPLATE/`:

- `Performance Regression` asks for `scripts/benchmark_hotpaths.py` evidence and the generated `benchmark-hotpaths.json`
- `Interop Failure` asks for `scripts/interop_smoke.py` or `scripts/release_smoke.py` output plus the exact extra install command
- `Dataset Or On-Disk Bug` asks for a `Minimal Reproduction` plus any graph payload format/version or `manifest.json` evidence

Keep these templates aligned with the workflow names and smoke commands above so incoming reports already contain the artifacts maintainers need.

## TestPyPI

Use the GitHub `publish` workflow in manual mode to publish to TestPyPI first.

For first-time TestPyPI setup, a pending publisher is fine while the project does
not exist yet. If you prefer token-based publishing instead, create an
environment secret named `TEST_PYPI_API_TOKEN` on the GitHub `testpypi`
environment; the workflow uses that secret when present and falls back to
Trusted Publishing otherwise.

After upload, verify:

- the project page renders correctly
- `pip install --index-url https://test.pypi.org/simple/ sky-vgl` works in a clean environment
- the documented extras install correctly

## PyPI

Create a GitHub release after TestPyPI verification passes. The release workflow publishes with Trusted Publishing and uploads the already-built distribution artifacts to PyPI.

Because `sky-vgl` already exists on PyPI, configure a normal trusted publisher
under `Manage Project -> Publishing`; do not leave PyPI configured only with a
pending publisher. Pending publishers are for first project creation and will
fail for existing projects with an `invalid-pending-publisher` style error.

If you prefer an explicit fallback, create an environment secret named
`PYPI_API_TOKEN` on the GitHub `pypi` environment. The workflow uses that token
when present and falls back to Trusted Publishing otherwise.

## Post-release verification

1. Install from PyPI in a clean environment.
2. Import `vgl` and the golden-path public imports.
3. If the release touched interop adapters, rerun `python scripts/interop_smoke.py --backend dgl` and/or `python scripts/interop_smoke.py --backend pyg` for the extras you have installed. Use `--backend all` only when both extras are present; the host-assisted release smoke script will abort early when either backend is missing. Artifact interop smoke defaults to host-assisted dependency discovery, so `python scripts/release_smoke.py --artifact-dir dist --kind wheel --interop-backend dgl` (or `pyg`) expects those extras to remain importable from the host environment before invoking `--interop-backend all`.
4. Packaging tests also cover hermetic fake-backend success paths so CI can validate `--interop-backend pyg` and `--interop-backend all` without requiring real host PyG or DGL installs.
5. Inspect the benchmark artifact from `scripts/benchmark_hotpaths.py` when runtime-sensitive code changed. Its JSON schema is versioned and includes timestamp, runner metadata, and per-domain timing maps in seconds.
6. Verify the PyPI project page links for Homepage, Repository, Documentation, and Issues.
7. Check that the tagged source and published package versions match.
8. Confirm `refs/tags/vX.Y.Z` matches `vgl.__version__ == "X.Y.Z"` and that README/docs no longer show stale hard-coded version badges.

The published distribution name is `sky-vgl`, but the Python import surface remains `vgl`.
