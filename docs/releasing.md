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
python scripts/release_smoke.py --artifact-dir dist --kind all
```

5. Confirm the smoke script reports both wheel and sdist installs passing from
outside the repository root. The script creates isolated virtual environments,
installs the built artifacts, and verifies `import vgl` plus the golden-path
symbols from the release contract (`Graph`, `Trainer`, `PlanetoidDataset`,
`NodeClassificationTask`) resolve from the installed distribution instead of the
checkout.
6. Draft release notes from the changelog and recent commits before tagging:

```bash
git log --oneline --decorate <previous-tag>..HEAD
```

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
3. Verify the PyPI project page links for Homepage, Repository, Documentation, and Issues.
4. Check that the tagged source and published package versions match.
5. Confirm `refs/tags/vX.Y.Z` matches `vgl.__version__ == "X.Y.Z"` and that README/docs no longer show stale hard-coded version badges.

The published distribution name is `sky-vgl`, but the Python import surface remains `vgl`.
