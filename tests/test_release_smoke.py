import importlib.util
import sys
from pathlib import Path

import pytest

from scripts.contracts import REAL_INTEROP_BACKENDS


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "release_smoke.py"


def _load_release_smoke_module():
    spec = importlib.util.spec_from_file_location("release_smoke", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_args_accepts_optional_interop_backend_flag():
    release_smoke = _load_release_smoke_module()

    args = release_smoke._parse_args(["--interop-backend", "dgl"])

    assert args.interop_backend == "dgl"


def test_parse_args_defaults_optional_interop_backend_to_none():
    release_smoke = _load_release_smoke_module()

    args = release_smoke._parse_args([])

    assert args.interop_backend == "none"


def test_parse_args_accepts_all_interop_backend_flag():
    release_smoke = _load_release_smoke_module()

    args = release_smoke._parse_args(["--interop-backend", "all"])

    assert args.interop_backend == "all"


def test_parse_args_rejects_unknown_interop_backend():
    release_smoke = _load_release_smoke_module()

    with pytest.raises(SystemExit):
        release_smoke._parse_args(["--interop-backend", "networkx"])


def test_selected_interop_backends_treat_none_as_disabled():
    release_smoke = _load_release_smoke_module()

    assert release_smoke._selected_interop_backends("none") == ()
    assert release_smoke._selected_interop_backends(None) == ()


def test_selected_interop_backends_expands_all_keyword():
    release_smoke = _load_release_smoke_module()

    assert release_smoke._selected_interop_backends("all") == REAL_INTEROP_BACKENDS


def test_selected_interop_backends_rejects_unknown_backend():
    release_smoke = _load_release_smoke_module()

    with pytest.raises(ValueError, match="unsupported interop backend"):
        release_smoke._selected_interop_backends("networkx")


def test_build_interop_check_script_for_dgl_uses_installed_public_api():
    release_smoke = _load_release_smoke_module()

    script = release_smoke._build_interop_check_script(
        "dgl",
        repo_root=Path("/tmp/repo"),
        dependency_paths=[Path("/opt/site-packages")],
    )

    assert "site.addsitedir('/opt/site-packages')" in script
    assert "Graph.from_dgl(graph.to_dgl())" in script
    assert "repo_root not in module_path.parents" in script


def test_build_interop_check_script_for_pyg_uses_installed_public_api():
    release_smoke = _load_release_smoke_module()

    script = release_smoke._build_interop_check_script(
        "pyg",
        repo_root=Path("/tmp/repo"),
        dependency_paths=[],
    )

    assert "Graph.from_pyg(graph.to_pyg())" in script
    assert "edge_attr" in script


def test_backend_import_module_name_returns_expected_names():
    release_smoke = _load_release_smoke_module()

    assert release_smoke._backend_import_module_name("pyg") == "torch_geometric"
    assert release_smoke._backend_import_module_name("dgl") == "dgl"


def test_preflight_reports_missing_backends(monkeypatch):
    release_smoke = _load_release_smoke_module()

    def fake_check(module_name, dependency_paths):
        return module_name == "dgl"

    monkeypatch.setattr(release_smoke, "_check_backend_availability", fake_check)

    with pytest.raises(SystemExit, match="pyg"):
        release_smoke._preflight_interop_backends(
            release_smoke._selected_interop_backends("all"),
            dependency_paths=[],
        )


def test_preflight_succeeds_when_all_backends_present(monkeypatch):
    release_smoke = _load_release_smoke_module()

    monkeypatch.setattr(release_smoke, "_check_backend_availability", lambda module_name, dependency_paths: True)

    release_smoke._preflight_interop_backends(
        release_smoke._selected_interop_backends("all"),
        dependency_paths=[],
    )
