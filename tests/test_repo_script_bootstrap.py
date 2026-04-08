from pathlib import Path
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]

SCRIPT_PATHS = {
    "benchmark_hotpaths": REPO_ROOT / "scripts" / "benchmark_hotpaths.py",
    "full_scan": REPO_ROOT / "scripts" / "full_scan.py",
    "release_contract_scan": REPO_ROOT / "scripts" / "release_contract_scan.py",
    "public_surface_scan": REPO_ROOT / "scripts" / "public_surface_scan.py",
    "install_release_extras": REPO_ROOT / "scripts" / "install_release_extras.py",
    "metadata_consistency": REPO_ROOT / "scripts" / "metadata_consistency.py",
    "release_smoke": REPO_ROOT / "scripts" / "release_smoke.py",
    "interop_smoke": REPO_ROOT / "scripts" / "interop_smoke.py",
}

@pytest.mark.parametrize("script_name", sorted(SCRIPT_PATHS))
def test_scripts_use_shared_repo_script_bootstrap(script_name: str):
    text = SCRIPT_PATHS[script_name].read_text(encoding="utf-8")

    assert "if not __package__:" not in text
    assert "sys.path.insert(0, repo_root_str)" not in text
    assert "import repo_script_imports" in text
    assert 'importlib.import_module("scripts.repo_script_imports")' not in text
    assert 'importlib.import_module("repo_script_imports")' not in text


def test_benchmark_hotpaths_module_load_avoids_eager_torch_import(tmp_path: Path):
    script_path = SCRIPT_PATHS["benchmark_hotpaths"]
    probe = (
        "import importlib.util, sys\n"
        "from pathlib import Path\n"
        "script_path = Path(sys.argv[1])\n"
        "sys.path.insert(0, str(script_path.parent.parent))\n"
        "spec = importlib.util.spec_from_file_location('benchmark_hotpaths_probe', script_path)\n"
        "assert spec is not None\n"
        "assert spec.loader is not None\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "sys.modules['benchmark_hotpaths_probe'] = module\n"
        "spec.loader.exec_module(module)\n"
        "print('torch' in sys.modules)\n"
    )

    completed = subprocess.run(
        [sys.executable, "-c", probe, str(script_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "False"


@pytest.mark.parametrize("script_name", sorted(SCRIPT_PATHS))
def test_scripts_support_direct_execution_outside_repo_root(tmp_path: Path, script_name: str):
    completed = subprocess.run(
        [sys.executable, str(SCRIPT_PATHS[script_name]), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout.lower()
