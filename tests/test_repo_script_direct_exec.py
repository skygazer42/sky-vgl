import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_script_outside_repo(tmp_path: Path, relative_script: str, *args: str) -> subprocess.CompletedProcess[str]:
    script = REPO_ROOT / relative_script
    return subprocess.run(
        [sys.executable, str(script), *args],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )


def test_full_scan_lists_tasks_outside_repo_root(tmp_path):
    completed = _run_script_outside_repo(tmp_path, "scripts/full_scan.py", "--list")

    assert completed.returncode == 0, completed.stderr
    assert "SCAN 001 [repo] README.md exists" in completed.stdout
    assert "SUMMARY listed " in completed.stdout


def test_metadata_consistency_runs_outside_repo_root(tmp_path):
    completed = _run_script_outside_repo(tmp_path, "scripts/metadata_consistency.py")

    assert completed.returncode == 0, completed.stderr
    assert "SUMMARY 6/6 passed" in completed.stdout


def test_public_surface_scan_lists_tasks_outside_repo_root(tmp_path):
    completed = _run_script_outside_repo(tmp_path, "scripts/public_surface_scan.py", "--list")

    assert completed.returncode == 0, completed.stderr
    assert "SCAN 001" in completed.stdout
    assert "tests/integration avoids legacy import paths" in completed.stdout


def test_interop_smoke_lists_backends_outside_repo_root(tmp_path):
    completed = _run_script_outside_repo(tmp_path, "scripts/interop_smoke.py", "--list-backends")

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == ["pyg", "dgl"]


def test_release_contract_scan_lists_tasks_outside_repo_root(tmp_path):
    completed = _run_script_outside_repo(tmp_path, "scripts/release_contract_scan.py", "--list")

    assert completed.returncode == 0, completed.stderr
    assert "SCAN 001 [artifact] built wheel exists" in completed.stdout
    assert "SCAN 036 [sdist] sdist excludes __pycache__" in completed.stdout


def test_install_release_extras_help_renders_outside_repo_root(tmp_path):
    completed = _run_script_outside_repo(tmp_path, "scripts/install_release_extras.py", "--help")

    assert completed.returncode == 0
    assert "usage: install_release_extras.py" in completed.stdout
    assert "--artifact-dir" in completed.stdout


def test_release_smoke_help_renders_outside_repo_root(tmp_path):
    completed = _run_script_outside_repo(tmp_path, "scripts/release_smoke.py", "--help")

    assert completed.returncode == 0
    assert "usage: release_smoke.py" in completed.stdout
    assert "--interop-backend" in completed.stdout
