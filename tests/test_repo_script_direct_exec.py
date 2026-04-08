import json
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


def test_docs_link_scan_lists_tasks_outside_repo_root(tmp_path):
    completed = _run_script_outside_repo(tmp_path, "scripts/docs_link_scan.py", "--list")

    assert completed.returncode == 0, completed.stderr
    assert "README.md link LICENSE resolves" in completed.stdout
    assert "docs/getting-started/installation.md link ../support-matrix.md resolves" in completed.stdout


def test_dependency_audit_prints_requirements_outside_repo_root(tmp_path):
    completed = _run_script_outside_repo(tmp_path, "scripts/dependency_audit.py", "--print-requirements")

    assert completed.returncode == 0, completed.stderr
    assert "torch>=2.4" in completed.stdout.splitlines()
    assert "typing_extensions>=4.12" in completed.stdout.splitlines()


def test_extras_smoke_lists_defaults_outside_repo_root(tmp_path):
    completed = _run_script_outside_repo(tmp_path, "scripts/extras_smoke.py", "--list-defaults")

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == ["networkx", "scipy", "tensorboard"]


def test_benchmark_hotpaths_writes_json_outside_repo_root(tmp_path):
    output = tmp_path / "benchmarks" / "outside-repo.json"
    completed = _run_script_outside_repo(
        tmp_path,
        "scripts/benchmark_hotpaths.py",
        "--num-nodes",
        "100",
        "--num-edges",
        "500",
        "--num-queries",
        "25",
        "--num-partitions",
        "2",
        "--warmup",
        "1",
        "--repeats",
        "1",
        "--seed",
        "0",
        "--output",
        str(output),
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["benchmark"] == "vgl_hotpaths"
    assert payload["config"]["num_nodes"] == 100
