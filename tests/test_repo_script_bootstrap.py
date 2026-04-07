import importlib
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]

SCRIPT_PATHS = {
    "full_scan": REPO_ROOT / "scripts" / "full_scan.py",
    "release_contract_scan": REPO_ROOT / "scripts" / "release_contract_scan.py",
    "public_surface_scan": REPO_ROOT / "scripts" / "public_surface_scan.py",
    "install_release_extras": REPO_ROOT / "scripts" / "install_release_extras.py",
    "metadata_consistency": REPO_ROOT / "scripts" / "metadata_consistency.py",
    "release_smoke": REPO_ROOT / "scripts" / "release_smoke.py",
    "interop_smoke": REPO_ROOT / "scripts" / "interop_smoke.py",
}


def _load_script_module(script_name: str):
    module_name = f"{script_name}_bootstrap_probe"
    script_path = SCRIPT_PATHS[script_name]
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("script_name", sorted(SCRIPT_PATHS))
def test_script_bootstrap_is_centralized_in_repo_script_imports(script_name: str):
    text = SCRIPT_PATHS[script_name].read_text(encoding="utf-8")

    assert "if not __package__:" not in text
    assert "sys.path.insert(0, repo_root_str)" not in text


@pytest.mark.parametrize("script_name", sorted(SCRIPT_PATHS))
def test_scripts_reuse_shared_repo_script_helpers(script_name: str):
    module = _load_script_module(script_name)
    shared_helpers = importlib.import_module("scripts.repo_script_imports")

    assert module.load_repo_module is shared_helpers.load_repo_module
    if script_name == "interop_smoke":
        assert module.ensure_repo_root_on_path is shared_helpers.ensure_repo_root_on_path


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
