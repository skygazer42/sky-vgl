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
