import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

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


def _repo_relative_artifact_dir(name: str, source_dir: Path) -> tuple[Path, str]:
    artifact_dir = REPO_ROOT / ".tmp_test_artifacts" / name
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    artifact_dir.mkdir(parents=True)
    for artifact in source_dir.iterdir():
        if artifact.is_file():
            shutil.copy2(artifact, artifact_dir / artifact.name)
    return artifact_dir, str(artifact_dir.relative_to(REPO_ROOT))


@pytest.fixture(scope="module")
def built_artifact_dir(tmp_path_factory) -> Path:
    output_dir = tmp_path_factory.mktemp("release-contract-dist")
    subprocess.run(
        [sys.executable, "-m", "build", "--outdir", str(output_dir)],
        cwd=REPO_ROOT,
        check=True,
    )
    return output_dir


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
    assert any("sdist contains /scripts/repo_script_imports.py" in line for line in listed)


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


def test_release_contract_scan_resolves_relative_artifact_dir_from_repo_root(
    built_artifact_dir: Path,
    tmp_path: Path,
):
    artifact_dir, relative_artifact_dir = _repo_relative_artifact_dir(
        "release-contract-scan-relative",
        built_artifact_dir,
    )
    try:
        completed = subprocess.run(
            [
                sys.executable,
                str(SCAN_SCRIPT),
                "--artifact-dir",
                relative_artifact_dir,
            ],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )
    finally:
        shutil.rmtree(artifact_dir)

    assert completed.returncode == 0, completed.stderr
    assert "SUMMARY" in completed.stdout
