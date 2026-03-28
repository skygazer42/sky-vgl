import importlib.util
import subprocess
import sys
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


def test_full_scan_catalog_contains_exactly_100_unique_tasks():
    scan = _load_scan_module()

    tasks = scan.build_tasks(REPO_ROOT)

    assert len(tasks) == 100
    assert len({task.id for task in tasks}) == 100
    assert all(task.category for task in tasks)
    assert all(task.description for task in tasks)


def test_full_scan_lists_every_task():
    completed = subprocess.run(
        [sys.executable, str(SCAN_SCRIPT), "--list"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    listed = [line for line in completed.stdout.splitlines() if line.startswith("SCAN ")]
    assert len(listed) == 100


def test_full_scan_passes_on_the_repository():
    completed = subprocess.run(
        [sys.executable, str(SCAN_SCRIPT)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "SUMMARY 100/100 passed" in completed.stdout


def test_ci_workflow_runs_the_full_scan_script():
    ci_text = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert "python scripts/full_scan.py" in ci_text
