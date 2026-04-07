import importlib.util
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import scripts.contracts as contracts
from scripts.contracts import (
    OPTIONAL_EXTRAS,
    PROJECT_URLS,
    RELEASE_INTEROP_EXTRA_REQUIREMENTS,
    SDIST_EXCLUDED_SUBSTRINGS,
    SDIST_REQUIRED_SUFFIXES,
    WHEEL_EXCLUDED_SUBSTRINGS,
    WHEEL_REQUIRED_FILES,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SCAN_SCRIPT = REPO_ROOT / "scripts" / "release_contract_scan.py"


def _load_release_contract_scan(module_name: str = "release_contract_scan"):
    spec = importlib.util.spec_from_file_location(module_name, SCAN_SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def built_artifact_dir(tmp_path_factory) -> Path:
    output_dir = tmp_path_factory.mktemp("release-contract-dist")
    subprocess.run(
        [sys.executable, "-m", "build", "--outdir", str(output_dir)],
        cwd=REPO_ROOT,
        check=True,
    )
    return output_dir


def test_release_contract_scan_prefers_repo_contracts_module(monkeypatch, tmp_path):
    shadow_module = tmp_path / "contracts.py"
    shadow_module.write_text(
        textwrap.dedent(
            """
            OPTIONAL_EXTRAS = ()
            PROJECT_NAME = "shadow-project"
            PROJECT_URLS = {}
            RELEASE_INTEROP_EXTRA_REQUIREMENTS = {"pyg": "shadow"}
            REQUIRES_PYTHON = ">=9.9"
            SDIST_EXCLUDED_SUBSTRINGS = ()
            SDIST_REQUIRED_SUFFIXES = ()
            WHEEL_EXCLUDED_SUBSTRINGS = ()
            WHEEL_REQUIRED_FILES = ()
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.syspath_prepend(str(tmp_path))
    shadowed = sys.modules.pop("contracts", None)
    try:
        scan = _load_release_contract_scan("release_contract_scan_shadowed")
    finally:
        sys.modules.pop("release_contract_scan_shadowed", None)
        if shadowed is not None:
            sys.modules["contracts"] = shadowed

    assert scan.PROJECT_NAME == contracts.PROJECT_NAME
    assert scan.RELEASE_INTEROP_EXTRA_REQUIREMENTS == contracts.RELEASE_INTEROP_EXTRA_REQUIREMENTS


def test_release_artifact_helper_reads_built_contract_scan_wheel(built_artifact_dir: Path):
    from scripts.release_artifact_metadata import read_wheel_metadata

    wheel_path = next(built_artifact_dir.glob("*.whl"))
    metadata, detail = read_wheel_metadata(wheel_path)

    assert metadata is not None
    assert detail.endswith("METADATA")
    requires_dist = set(metadata.get_all("Requires-Dist", []))
    for requirement in RELEASE_INTEROP_EXTRA_REQUIREMENTS.values():
        assert requirement in requires_dist


def test_release_contract_scan_lists_stable_catalog():
    completed = subprocess.run(
        [sys.executable, str(SCAN_SCRIPT), "--list"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    listed = [line for line in completed.stdout.splitlines() if line.startswith("SCAN ")]
    expected = 5 + len(PROJECT_URLS) + len(OPTIONAL_EXTRAS) + len(WHEEL_REQUIRED_FILES)
    expected += len(WHEEL_EXCLUDED_SUBSTRINGS) + len(SDIST_REQUIRED_SUFFIXES) + len(SDIST_EXCLUDED_SUBSTRINGS)
    expected += len(RELEASE_INTEROP_EXTRA_REQUIREMENTS)
    assert len(listed) == expected
    for extra in RELEASE_INTEROP_EXTRA_REQUIREMENTS:
        assert any(f"wheel metadata exposes {extra} extra requirement line" in line for line in listed)
    assert any("sdist contains /scripts/install_release_extras.py" in line for line in listed)
    assert any("sdist contains /scripts/release_artifact_metadata.py" in line for line in listed)


def test_release_contract_scan_passes_on_built_artifacts(built_artifact_dir: Path):
    completed = subprocess.run(
        [
            sys.executable,
            str(SCAN_SCRIPT),
            "--artifact-dir",
            str(built_artifact_dir),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    expected = 5 + len(PROJECT_URLS) + len(OPTIONAL_EXTRAS) + len(WHEEL_REQUIRED_FILES)
    expected += len(WHEEL_EXCLUDED_SUBSTRINGS) + len(SDIST_REQUIRED_SUFFIXES) + len(SDIST_EXCLUDED_SUBSTRINGS)
    expected += len(RELEASE_INTEROP_EXTRA_REQUIREMENTS)
    assert f"SUMMARY {expected}/{expected} passed" in completed.stdout
    for requirement in RELEASE_INTEROP_EXTRA_REQUIREMENTS.values():
        assert requirement in completed.stdout
