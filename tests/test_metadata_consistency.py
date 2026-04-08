import importlib.util
import subprocess
import sys
import textwrap
from pathlib import Path

import scripts.contracts as contracts


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "metadata_consistency.py"


def _load_metadata_consistency(module_name: str = "metadata_consistency"):
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_metadata_consistency_prefers_repo_contracts_module(monkeypatch, tmp_path):
    shadow_module = tmp_path / "contracts.py"
    shadow_module.write_text(
        textwrap.dedent(
            """
            DOCS_INDEX_VERSION_BADGE = "shadow-docs-badge"
            PROJECT_URLS = {}
            README_VERSION_BADGE = "shadow-readme-badge"
            RELEASE_VERSION = "999.0.0"
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.syspath_prepend(str(tmp_path))
    shadowed = sys.modules.pop("contracts", None)
    try:
        metadata_consistency = _load_metadata_consistency("metadata_consistency_shadowed")
    finally:
        sys.modules.pop("metadata_consistency_shadowed", None)
        if shadowed is not None:
            sys.modules["contracts"] = shadowed

    assert metadata_consistency.PROJECT_URLS == contracts.PROJECT_URLS
    assert metadata_consistency.RELEASE_VERSION == contracts.RELEASE_VERSION


def test_metadata_consistency_passes_on_repository():
    completed = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "SUMMARY 6/6 passed" in completed.stdout


def test_metadata_consistency_validates_tag_refs():
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--git-ref", "refs/tags/v999.0.0"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert "tag version" in completed.stdout
