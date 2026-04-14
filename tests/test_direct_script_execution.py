import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_script_outside_repo_root(tmp_path: Path, relative_script: str, *args: str) -> subprocess.CompletedProcess[str]:
    script = REPO_ROOT / "scripts" / relative_script
    return subprocess.run(
        [sys.executable, str(script), *args],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )


def test_full_scan_lists_tasks_outside_repo_root(tmp_path):
    completed = _run_script_outside_repo_root(tmp_path, "full_scan.py", "--list")

    assert completed.returncode == 0, completed.stderr
    assert "SCAN 001 [repo] README.md exists" in completed.stdout
    assert "SUMMARY listed " in completed.stdout


def test_public_surface_scan_lists_tasks_outside_repo_root(tmp_path):
    completed = _run_script_outside_repo_root(tmp_path, "public_surface_scan.py", "--list")

    assert completed.returncode == 0, completed.stderr
    assert "SCAN 001 [root] vgl exports Graph from vgl.graph" in completed.stdout
    assert "[imports] examples avoids legacy import paths" in completed.stdout
    assert "[imports] tests/integration avoids legacy import paths" in completed.stdout


def test_metadata_consistency_runs_outside_repo_root(tmp_path):
    completed = _run_script_outside_repo_root(tmp_path, "metadata_consistency.py")

    assert completed.returncode == 0, completed.stderr
    assert "SUMMARY 6/6 passed" in completed.stdout


def test_interop_smoke_lists_backends_outside_repo_root(tmp_path):
    completed = _run_script_outside_repo_root(tmp_path, "interop_smoke.py", "--list-backends")

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == ["pyg", "dgl"]
