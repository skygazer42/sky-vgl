# Releasing VGL

## Pre-release checklist

1. Confirm the intended public version in `vgl/version.py`.
2. Re-check that the `vgl` project name is still available or that you control the existing PyPI project.
3. Run the full local verification suite:

```bash
python -m pytest -q
python -m build
python -m twine check dist/*.whl dist/*.tar.gz
```

4. Smoke-test the built wheel from outside the repository root:

```bash
python -m venv /tmp/vgl-release-check
/tmp/vgl-release-check/bin/pip install --upgrade pip
/tmp/vgl-release-check/bin/pip install --no-deps dist/*.whl
cd /tmp
/tmp/vgl-release-check/bin/python -c "import vgl; print(vgl.__version__)"
```

## TestPyPI

Use the GitHub `publish` workflow in manual mode to publish to TestPyPI first.

After upload, verify:

- the project page renders correctly
- `pip install --index-url https://test.pypi.org/simple/ vgl` works in a clean environment
- the documented extras install correctly

## PyPI

Create a GitHub release after TestPyPI verification passes. The release workflow publishes with Trusted Publishing and uploads the already-built distribution artifacts to PyPI.

## Post-release verification

1. Install from PyPI in a clean environment.
2. Import `vgl`, `Graph`, and `Trainer`.
3. Verify the PyPI project page links for Homepage, Repository, Documentation, and Issues.
4. Check that the tagged source and published package versions match.
