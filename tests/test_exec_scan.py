import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_dependency_audit_prints_runtime_requirements_from_pyproject():
    script = REPO_ROOT / "scripts" / "dependency_audit.py"

    completed = subprocess.run(
        [sys.executable, str(script), "--print-requirements"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    assert "torch>=2.4" in lines
    assert "typing_extensions>=4.12" in lines


def test_extras_smoke_lists_default_lightweight_extras():
    script = REPO_ROOT / "scripts" / "extras_smoke.py"

    completed = subprocess.run(
        [sys.executable, str(script), "--list-defaults"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    assert lines == ["networkx", "scipy", "tensorboard"]


def test_ci_workflow_runs_lint_extras_smoke_and_dependency_audit():
    ci_text = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert "python -m ruff check ." in ci_text
    assert "python scripts/extras_smoke.py --extras networkx scipy tensorboard" in ci_text
    assert "python scripts/dependency_audit.py" in ci_text
    assert "python scripts/metadata_consistency.py" in ci_text


def test_ci_workflow_includes_extended_matrix_and_jobs():
    ci_text = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert 'python-version: ["3.10", "3.11", "3.12"]' in ci_text
    assert "runs-on: macos-latest" in ci_text
    assert "workflow-lint:" in ci_text
    assert "contract-tests:" in ci_text
    assert "benchmark:" in ci_text
    assert "python scripts/benchmark_hotpaths.py" in ci_text
    assert "--cov=vgl" in ci_text
    assert "--cov=scripts" in ci_text


def test_manual_interop_workflow_covers_pyg_and_dgl():
    workflow_text = (REPO_ROOT / ".github" / "workflows" / "interop-smoke.yml").read_text(encoding="utf-8")

    assert "workflow_dispatch:" in workflow_text
    assert "schedule:" in workflow_text
    assert 'python -m pip install -e ".[dev,pyg]"' in workflow_text
    assert 'python -m pip install -e ".[dev,dgl]"' in workflow_text
    assert 'python -m pip install -e ".[dev,pyg,dgl]"' in workflow_text
    assert "python -m build" in workflow_text
    assert "python scripts/interop_smoke.py --backend pyg" in workflow_text
    assert "python scripts/interop_smoke.py --backend dgl" in workflow_text
    assert "python scripts/interop_smoke.py --backend all" in workflow_text
    assert "python scripts/release_smoke.py --artifact-dir dist --kind wheel --interop-backend all" in workflow_text


def test_ci_and_publish_build_jobs_gate_all_backend_artifact_interop():
    ci_text = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    publish_text = (REPO_ROOT / ".github" / "workflows" / "publish.yml").read_text(encoding="utf-8")

    assert 'python -m pip install -e ".[pyg,dgl]"' in ci_text
    assert "python scripts/release_smoke.py --artifact-dir dist --kind all --interop-backend all" in ci_text
    assert 'python -m pip install -e ".[pyg,dgl]"' in publish_text
    assert "python scripts/release_smoke.py --artifact-dir dist --kind all --interop-backend all" in publish_text


def test_makefile_exposes_interop_smoke_target():
    makefile_text = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")

    assert "INTEROP_BACKEND ?= all" in makefile_text
    assert "interop-smoke:" in makefile_text
    assert "scripts/interop_smoke.py --backend=$(INTEROP_BACKEND)" in makefile_text
    assert "RELEASE_INTEROP_BACKEND ?= dgl" in makefile_text
    assert "release-artifact-interop-smoke:" in makefile_text
    assert "scripts/release_smoke.py --artifact-dir dist --kind wheel --interop-backend=$(RELEASE_INTEROP_BACKEND)" in makefile_text


def test_benchmark_hotpaths_writes_quiet_stable_json(tmp_path):
    script = REPO_ROOT / "scripts" / "benchmark_hotpaths.py"
    output_file = tmp_path / "benchmarks" / "benchmark.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
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
            str(output_file),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == ""
    payload = json.loads(output_file.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["benchmark"] == "vgl_hotpaths"
    assert payload["preset"] == "default"
    assert payload["config"]["num_nodes"] == 100
    assert set(payload) >= {"config", "query_ops", "routing", "sampling"}


def test_support_matrix_tracks_live_optional_interop_verification():
    support_matrix = (REPO_ROOT / "docs" / "support-matrix.md").read_text(encoding="utf-8")

    assert "interop-smoke" in support_matrix
    assert "python scripts/interop_smoke.py --backend dgl" in support_matrix
    assert "python scripts/interop_smoke.py --backend pyg" in support_matrix
    assert "python scripts/release_smoke.py --artifact-dir dist --kind wheel --interop-backend dgl" in support_matrix
    assert "Planned real install smoke" not in support_matrix
