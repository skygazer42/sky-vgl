import importlib.util
import subprocess
import sys
import textwrap
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCAN_SCRIPT = REPO_ROOT / "scripts" / "full_scan.py"


def _load_scan_module():
    spec = importlib.util.spec_from_file_location("full_scan", SCAN_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_full_scan_catalog_contains_unique_tasks():
    scan = _load_scan_module()

    tasks = scan.build_tasks(REPO_ROOT)

    assert len(tasks) >= 100
    assert len({task.id for task in tasks}) == len(tasks)
    assert all(task.category for task in tasks)
    assert all(task.description for task in tasks)
    descriptions = {task.description for task in tasks}
    assert "releasing doc includes release artifact interop smoke command" in descriptions
    assert "Makefile includes release artifact interop smoke target" in descriptions
    assert "interop workflow builds artifacts" in descriptions
    assert "interop workflow runs all-backend release artifact smoke" in descriptions
    assert "releasing doc mentions package-check release gate" in descriptions
    assert "releasing doc mentions publish build release gate" in descriptions
    assert "releasing doc mentions release gate interop extras install" in descriptions
    assert "releasing doc mentions all-backend release artifact smoke gate" in descriptions
    assert "CI package-check job installs release interop extras from built wheel" in descriptions
    assert "CI package-check job runs all-backend release artifact smoke" in descriptions
    assert "publish build job installs release interop extras from built wheel" in descriptions
    assert "publish build runs all-backend release artifact smoke" in descriptions


def test_full_scan_lists_every_task():
    expected = len(_load_scan_module().build_tasks(REPO_ROOT))
    completed = subprocess.run(
        [sys.executable, str(SCAN_SCRIPT), "--list"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    listed = [line for line in completed.stdout.splitlines() if line.startswith("SCAN ")]
    assert len(listed) == expected
    assert f"SUMMARY listed {expected} tasks" in completed.stdout


def test_full_scan_passes_on_the_repository():
    expected = len(_load_scan_module().build_tasks(REPO_ROOT))
    completed = subprocess.run(
        [sys.executable, str(SCAN_SCRIPT)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert f"SUMMARY {expected}/{expected} passed" in completed.stdout


def test_scan_context_workflow_job_contains_anchors_to_jobs_section(tmp_path):
    scan = _load_scan_module()
    workflow_path = tmp_path / "workflow.yml"
    workflow_path.write_text(
        textwrap.dedent(
            """
            name: demo

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

    ctx = scan.ScanContext(tmp_path)

    assert ctx.workflow_job_contains("workflow.yml", "build", "echo build") == (
        True,
        "workflow.yml job 'build' contains 'echo build'",
    )
    assert ctx.workflow_job_contains("workflow.yml", "build", "SHOULD_NOT_BE_CAPTURED")[0] is False
    assert ctx.workflow_job_contains("workflow.yml", "build", "concurrency:")[0] is False


def test_full_scan_release_gate_tasks_require_named_ci_steps(tmp_path):
    scan = _load_scan_module()
    workflow_path = tmp_path / ".github" / "workflows" / "ci.yml"
    workflow_path.parent.mkdir(parents=True)
    workflow_path.write_text(
        textwrap.dedent(
            """
            name: ci

            jobs:
              package-check:
                runs-on: ubuntu-latest
                steps:
                  - name: Build distributions
                    run: python -m build
                  - name: Prepare release interop extras
                    run: python scripts/install_release_extras.py --artifact-dir dist --extras pyg dgl
                  - name: Verify all interop backends
                    run: python scripts/release_smoke.py --artifact-dir dist --kind all --interop-backend all
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    tasks = {task.id: task for task in scan.build_tasks(tmp_path)}

    assert tasks["065a"].check()[0] is False
    assert tasks["065b"].check()[0] is False


def test_full_scan_release_gate_tasks_reject_editable_ci_install_step(tmp_path):
    scan = _load_scan_module()
    workflow_path = tmp_path / ".github" / "workflows" / "ci.yml"
    workflow_path.parent.mkdir(parents=True)
    workflow_path.write_text(
        textwrap.dedent(
            """
            name: ci

            jobs:
              package-check:
                runs-on: ubuntu-latest
                steps:
                  - name: Install release interop extras
                    run: |
                      python scripts/install_release_extras.py --artifact-dir dist --extras pyg dgl
                      python -m pip install -e ".[pyg,dgl]"
                  - name: Smoke-test built distributions with all interop backends
                    run: python scripts/release_smoke.py --artifact-dir dist --kind all --interop-backend all
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    tasks = {task.id: task for task in scan.build_tasks(tmp_path)}

    assert tasks["065c"].check()[0] is False


def test_full_scan_release_gate_negative_task_fails_when_named_install_step_is_missing(tmp_path):
    scan = _load_scan_module()
    workflow_path = tmp_path / ".github" / "workflows" / "ci.yml"
    workflow_path.parent.mkdir(parents=True)
    workflow_path.write_text(
        textwrap.dedent(
            """
            name: ci

            jobs:
              package-check:
                runs-on: ubuntu-latest
                steps:
                  - name: Prepare release interop extras
                    run: python -m pip install -e ".[pyg,dgl]"
                  - name: Smoke-test built distributions with all interop backends
                    run: python scripts/release_smoke.py --artifact-dir dist --kind all --interop-backend all
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    tasks = {task.id: task for task in scan.build_tasks(tmp_path)}

    assert tasks["065c"].check()[0] is False


def test_full_scan_release_gate_negative_step_checks_fail_when_step_is_missing(tmp_path):
    scan = _load_scan_module()
    workflow_path = tmp_path / ".github" / "workflows" / "ci.yml"
    workflow_path.parent.mkdir(parents=True)
    workflow_path.write_text(
        textwrap.dedent(
            """
            name: ci

            jobs:
              package-check:
                runs-on: ubuntu-latest
                steps:
                  - name: Build distributions
                    run: python -m build
                  - name: Smoke-test built distributions with all interop backends
                    run: python scripts/release_smoke.py --artifact-dir dist --kind all --interop-backend all
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    tasks = {task.id: task for task in scan.build_tasks(tmp_path)}

    assert tasks["065c"].check()[0] is False


def test_ci_workflow_runs_the_full_scan_script():
    ci_text = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert "python scripts/full_scan.py" in ci_text
    assert "python scripts/public_surface_scan.py" in ci_text
    assert "python scripts/metadata_consistency.py" in ci_text
