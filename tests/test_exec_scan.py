import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

from scripts.workflow_contracts import workflow_job_text, workflow_step_text

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_dependency_audit_prints_selected_requirement_groups_from_pyproject():
    script = REPO_ROOT / "scripts" / "dependency_audit.py"

    completed = subprocess.run(
        [sys.executable, str(script), "--print-requirements", "--groups", "runtime", "networkx", "pyg"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    assert lines == [
        "numpy>=1.26",
        "torch>=2.4",
        "typing_extensions>=4.12",
        "networkx>=3.2",
        "torch-geometric>=2.5",
    ]


def test_dependency_audit_defaults_to_runtime_plus_all_optional_groups():
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
    assert "build>=1.2" in lines
    assert "numpy>=1.26" in lines
    assert "networkx>=3.2" in lines
    assert "tensorboard>=2.14" in lines
    assert "dgl>=1.1.3,<2" in lines
    assert "torch-geometric>=2.5" in lines


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


def test_publish_workflow_pins_all_actions_to_full_commit_shas():
    publish_text = (REPO_ROOT / ".github" / "workflows" / "publish.yml").read_text(encoding="utf-8")

    for action in (
        "actions/checkout",
        "actions/setup-python",
        "actions/upload-artifact",
        "actions/download-artifact",
        "pypa/gh-action-pypi-publish",
    ):
        assert re.search(rf"uses:\s+{re.escape(action)}@[0-9a-f]{{40}}", publish_text), action

    assert "@v4" not in publish_text
    assert "@v5" not in publish_text
    assert "@release/v1" not in publish_text


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
    assert "matrix:" in workflow_text
    assert "backend: pyg" in workflow_text
    assert "backend: dgl" in workflow_text
    assert "extras: pyg" in workflow_text
    assert "extras: dgl" in workflow_text
    assert 'python -m pip install -e ".[dev,${{ matrix.extras }}]"' in workflow_text
    assert 'python -m pip install -e ".[dev,pyg,dgl]"' in workflow_text
    assert "python -m build" in workflow_text
    assert "python scripts/interop_smoke.py --backend ${{ matrix.backend }}" in workflow_text
    assert "python scripts/interop_smoke.py --backend all" in workflow_text
    assert "python scripts/release_smoke.py --artifact-dir dist --kind wheel --interop-backend all" in workflow_text


def test_issue_templates_cover_performance_interop_and_dataset_intake():
    template_root = REPO_ROOT / ".github" / "ISSUE_TEMPLATE"
    config = (template_root / "config.yml").read_text(encoding="utf-8")

    performance = (template_root / "performance-regression.yml").read_text(encoding="utf-8")
    interop = (template_root / "interop-failure.yml").read_text(encoding="utf-8")
    dataset = (template_root / "dataset-bug.yml").read_text(encoding="utf-8")

    assert "blank_issues_enabled: false" in config
    assert "scripts/benchmark_hotpaths.py" in performance
    assert "benchmark-hotpaths.json" in performance
    assert "scripts/interop_smoke.py" in interop
    assert "scripts/release_smoke.py" in interop
    assert "sky-vgl[pyg]" in interop
    assert "sky-vgl[dgl]" in interop
    assert "benchmark-hotpaths" in performance
    assert "release-dists" in interop
    assert "graph payload format/version" in dataset
    assert "Minimal Reproduction" in dataset
    assert "manifest.json" in dataset


def test_ci_and_publish_build_jobs_gate_all_backend_artifact_interop():
    ci_install_step = workflow_step_text(
        REPO_ROOT / ".github" / "workflows" / "ci.yml",
        "package-check",
        "Install release interop extras",
    )
    ci_smoke_step = workflow_step_text(
        REPO_ROOT / ".github" / "workflows" / "ci.yml",
        "package-check",
        "Smoke-test built distributions with all interop backends",
    )
    publish_install_step = workflow_step_text(
        REPO_ROOT / ".github" / "workflows" / "publish.yml",
        "build",
        "Install release interop extras",
    )
    publish_smoke_step = workflow_step_text(
        REPO_ROOT / ".github" / "workflows" / "publish.yml",
        "build",
        "Smoke-test built distributions with all interop backends",
    )

    assert "python scripts/install_release_extras.py --artifact-dir dist --extras pyg dgl" in ci_install_step
    assert "python scripts/release_smoke.py --artifact-dir dist --kind all --interop-backend all" in ci_smoke_step
    assert "python scripts/install_release_extras.py --artifact-dir dist --extras pyg dgl" in publish_install_step
    assert "python scripts/release_smoke.py --artifact-dir dist --kind all --interop-backend all" in publish_smoke_step


def test_workflow_job_helper_comes_from_shared_scripts_module():
    assert workflow_job_text.__module__ == "scripts.workflow_contracts"


def test_workflow_job_text_anchors_to_jobs_section(tmp_path):
    workflow_path = tmp_path / "workflow.yml"
    workflow_path.write_text(
        textwrap.dedent(
            """
            name: demo

            on:
              workflow_dispatch:

            jobs:
              build:
                runs-on: ubuntu-latest
                steps:
                  - run: echo build

            concurrency: ci-${{ github.ref }}

            env:
              SHOULD_NOT_BE_CAPTURED: "1"
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    job_text = workflow_job_text(workflow_path, "build")

    assert "runs-on: ubuntu-latest" in job_text
    assert "echo build" in job_text
    assert "SHOULD_NOT_BE_CAPTURED" not in job_text
    assert "concurrency:" not in job_text


def test_workflow_job_text_tolerates_jobs_comments_and_quoted_keys(tmp_path):
    workflow_path = tmp_path / "workflow.yml"
    workflow_path.write_text(
        textwrap.dedent(
            """
            name: demo

            jobs:  # declared jobs
              "build":  # build contract
                runs-on: ubuntu-latest
                steps:
                  - run: echo build
              "release":
                runs-on: ubuntu-latest
                steps:
                  - run: echo release
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    job_text = workflow_job_text(workflow_path, "build")

    assert "echo build" in job_text
    assert "echo release" not in job_text


def test_workflow_step_text_isolates_single_named_step(tmp_path):
    workflow_path = tmp_path / "workflow.yml"
    workflow_path.write_text(
        textwrap.dedent(
            """
            name: demo

            jobs:
              build:
                runs-on: ubuntu-latest
                steps:
                  - name: Install release interop extras
                    run: python scripts/install_release_extras.py --artifact-dir dist --extras pyg dgl
                  - name: Smoke-test built distributions with all interop backends
                    run: python scripts/release_smoke.py --artifact-dir dist --kind all --interop-backend all
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    step_text = workflow_step_text(workflow_path, "build", "Install release interop extras")

    assert "install_release_extras.py" in step_text
    assert "interop-backend all" not in step_text


def test_workflow_step_text_tolerates_quoted_names_and_inline_comments(tmp_path):
    workflow_path = tmp_path / "workflow.yml"
    workflow_path.write_text(
        textwrap.dedent(
            """
            name: demo

            jobs:  # step parsing should tolerate comments
              "build":  # quoted job key
                runs-on: ubuntu-latest
                steps:
                  - name: "Install release interop extras"   # quoted step name
                    run: python scripts/install_release_extras.py --artifact-dir dist --extras pyg dgl
                  - name: "Smoke-test built distributions with all interop backends"
                    run: python scripts/release_smoke.py --artifact-dir dist --kind all --interop-backend all
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    step_text = workflow_step_text(workflow_path, "build", "Install release interop extras")

    assert "install_release_extras.py" in step_text
    assert "interop-backend all" not in step_text


def test_workflow_step_text_tolerates_quoted_names_comments_and_trailing_spaces(tmp_path):
    workflow_path = tmp_path / "workflow.yml"
    workflow_path.write_text(
        textwrap.dedent(
            """
            name: demo

            jobs:
              build:
                runs-on: ubuntu-latest
                steps:
                  - name: "Install release interop extras"  # quoted with comment
                    run: python scripts/install_release_extras.py --artifact-dir dist --extras pyg dgl
                  - name: 'Smoke-test built distributions with all interop backends'   
                    run: python scripts/release_smoke.py --artifact-dir dist --kind all --interop-backend all
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    step_text = workflow_step_text(workflow_path, "build", "Install release interop extras")

    assert "install_release_extras.py" in step_text
    assert "interop-backend all" not in step_text


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
