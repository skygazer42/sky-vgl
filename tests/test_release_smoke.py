import importlib.util
import importlib.abc
import os
import subprocess
import sys
import tarfile
import textwrap
import zipfile
from pathlib import Path

import pytest

import scripts.contracts as contracts
from scripts.contracts import REAL_INTEROP_BACKENDS


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "release_smoke.py"


def _load_release_smoke_module(module_name: str = "release_smoke"):
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _write_fake_wheel(tmp_path: Path, *, metadata_text: str | None = None) -> Path:
    wheel_path = tmp_path / "fake.whl"
    with zipfile.ZipFile(wheel_path, "w") as archive:
        archive.writestr("vgl/__init__.py", "__version__ = '0.0.0'\n")
        if metadata_text is not None:
            archive.writestr("fake-0.0.0.dist-info/METADATA", metadata_text)
    return wheel_path


def _write_fake_sdist(tmp_path: Path, members: dict[str, str]) -> Path:
    sdist_path = tmp_path / "fake.tar.gz"
    with tarfile.open(sdist_path, "w:gz") as archive:
        for relative_path, content in members.items():
            path = tmp_path / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            archive.add(path, arcname=f"fake-0.0.0/{relative_path}")
    return sdist_path


def test_release_smoke_prefers_repo_contracts_module(monkeypatch, tmp_path):
    shadow_module = tmp_path / "contracts.py"
    shadow_module.write_text(
        textwrap.dedent(
            """
            REAL_INTEROP_BACKENDS = ("shadow",)
            WHEEL_IMPORT_SYMBOLS = ("ShadowSymbol",)
            PREFERRED_IMPORT_SMOKES = (("shadow.module", "ShadowPreferred"),)
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.syspath_prepend(str(tmp_path))
    shadowed = sys.modules.pop("contracts", None)
    try:
        release_smoke = _load_release_smoke_module("release_smoke_shadowed")
    finally:
        sys.modules.pop("release_smoke_shadowed", None)
        sys.modules.pop("contracts", None)
        if shadowed is not None:
            sys.modules["contracts"] = shadowed

    assert release_smoke.REAL_INTEROP_BACKENDS == contracts.REAL_INTEROP_BACKENDS
    assert release_smoke.WHEEL_IMPORT_SYMBOLS == contracts.WHEEL_IMPORT_SYMBOLS
    assert release_smoke.PREFERRED_IMPORT_SMOKES == contracts.PREFERRED_IMPORT_SMOKES
    assert release_smoke.INTEROP_BACKENDS == ("none", *contracts.REAL_INTEROP_BACKENDS, "all")


def test_release_smoke_can_fall_back_to_package_repo_script_imports(monkeypatch):
    class _TopLevelRepoScriptImportsBlocker(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "repo_script_imports":
                raise ModuleNotFoundError("blocked import: repo_script_imports")
            return None

    blocker = _TopLevelRepoScriptImportsBlocker()
    shadowed = sys.modules.pop("repo_script_imports", None)
    monkeypatch.setattr(sys, "meta_path", [blocker, *sys.meta_path])
    try:
        release_smoke = _load_release_smoke_module("release_smoke_repo_script_imports_fallback")
    finally:
        sys.modules.pop("release_smoke_repo_script_imports_fallback", None)
        if shadowed is not None:
            sys.modules["repo_script_imports"] = shadowed

    assert release_smoke.INTEROP_BACKENDS == ("none", *contracts.REAL_INTEROP_BACKENDS, "all")


def test_parse_args_accepts_optional_interop_backend_flag():
    release_smoke = _load_release_smoke_module()

    args = release_smoke._parse_args(["--interop-backend", "dgl"])

    assert args.interop_backend == "dgl"


def test_parse_args_accepts_optional_import_time_threshold():
    release_smoke = _load_release_smoke_module()

    args = release_smoke._parse_args(["--max-import-seconds", "2.5"])

    assert args.max_import_seconds == 2.5


def test_parse_args_accepts_optional_artifact_size_thresholds():
    release_smoke = _load_release_smoke_module()

    args = release_smoke._parse_args(["--max-wheel-bytes", "500000", "--max-sdist-bytes", "600000"])

    assert args.max_wheel_bytes == 500000
    assert args.max_sdist_bytes == 600000


def test_parse_args_defaults_optional_interop_backend_to_none():
    release_smoke = _load_release_smoke_module()

    args = release_smoke._parse_args([])

    assert args.interop_backend == "none"
    assert args.max_import_seconds is None
    assert args.max_wheel_bytes is None
    assert args.max_sdist_bytes is None


def test_parse_args_accepts_all_interop_backend_flag():
    release_smoke = _load_release_smoke_module()

    args = release_smoke._parse_args(["--interop-backend", "all"])

    assert args.interop_backend == "all"


def test_parse_args_rejects_unknown_interop_backend():
    release_smoke = _load_release_smoke_module()

    with pytest.raises(SystemExit):
        release_smoke._parse_args(["--interop-backend", "networkx"])


def test_parse_args_rejects_non_positive_import_time_threshold():
    release_smoke = _load_release_smoke_module()

    with pytest.raises(SystemExit):
        release_smoke._parse_args(["--max-import-seconds", "0"])


def test_parse_args_rejects_non_positive_artifact_size_thresholds():
    release_smoke = _load_release_smoke_module()

    with pytest.raises(SystemExit):
        release_smoke._parse_args(["--max-wheel-bytes", "0"])
    with pytest.raises(SystemExit):
        release_smoke._parse_args(["--max-sdist-bytes", "-1"])


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


def test_build_import_check_script_measures_root_import_time_and_preferred_imports():
    release_smoke = _load_release_smoke_module()

    script = release_smoke._build_import_check_script(
        repo_root=Path("/tmp/repo"),
        dependency_paths=[Path("/opt/site-packages")],
    )

    assert "IMPORT_TIMING vgl" in script
    assert "time.perf_counter()" in script
    assert "from vgl.graph import Graph" in script
    assert "site.addsitedir('/opt/site-packages')" in script


def test_build_import_check_script_uses_root_and_preferred_import_paths():
    release_smoke = _load_release_smoke_module()

    script = release_smoke._build_import_check_script(
        repo_root=Path("/tmp/repo"),
        dependency_paths=[Path("/opt/site-packages")],
    )

    assert "import vgl" in script
    assert "from vgl import Graph, Trainer, PlanetoidDataset, NodeClassificationTask" in script
    assert "from vgl.graph import Graph" in script
    assert "from vgl.engine import Trainer" in script
    assert "from vgl.data import PlanetoidDataset" in script
    assert "from vgl.storage import MmapTensorStore" in script
    assert "from vgl.tasks import NodeClassificationTask" in script
    assert "repo_root not in module_path.parents" in script


def test_build_import_check_script_also_uses_legacy_compat_import_paths():
    release_smoke = _load_release_smoke_module()

    script = release_smoke._build_import_check_script(
        repo_root=Path("/tmp/repo"),
        dependency_paths=[],
    )

    assert "from vgl.core import Graph as LegacyCoreGraph" in script
    assert "from vgl.train import Trainer as LegacyTrainer" in script
    assert "from vgl.data import Loader as LegacyLoader" in script


def test_build_import_check_script_asserts_root_preferred_and_legacy_identity():
    release_smoke = _load_release_smoke_module()

    script = release_smoke._build_import_check_script(
        repo_root=Path("/tmp/repo"),
        dependency_paths=[],
    )

    assert "assert Graph is LegacyCoreGraph" in script
    assert "assert Graph is PreferredGraph" in script
    assert "assert Trainer is LegacyTrainer" in script
    assert "assert Trainer is PreferredTrainer" in script
    assert "assert PlanetoidDataset is PreferredPlanetoidDataset" in script
    assert "assert NodeClassificationTask is PreferredNodeClassificationTask" in script


def test_generated_import_smoke_script_executes_root_preferred_and_legacy_paths_together():
    release_smoke = _load_release_smoke_module()

    script = release_smoke._build_import_check_script(
        repo_root=Path("/tmp/non-repo-root"),
        dependency_paths=[],
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "IMPORT_TIMING vgl " in completed.stdout


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


def test_extra_dependency_paths_from_env_parses(tmp_path, monkeypatch):
    release_smoke = _load_release_smoke_module()

    extra_dirs = [tmp_path / "extra_a", tmp_path / "extra_b"]
    for path in extra_dirs:
        path.mkdir()
    monkeypatch.setenv("RELEASE_INTEROP_EXTRA_SITE_DIRS", os.pathsep.join(str(path) for path in extra_dirs))

    assert release_smoke._extra_dependency_paths_from_env() == extra_dirs


def test_resolved_dependency_paths_prefers_override(monkeypatch, tmp_path):
    release_smoke = _load_release_smoke_module()
    override_dir = tmp_path / "aggregated"
    override_dir.mkdir()
    monkeypatch.setenv("RELEASE_INTEROP_EXTRA_SITE_DIRS", str(override_dir))

    outer_paths = [Path("/outer/one"), Path("/outer/two")]
    monkeypatch.setattr(release_smoke, "_outer_site_packages", lambda: outer_paths)

    resolved = release_smoke._resolved_dependency_paths()

    assert resolved[0] == override_dir
    assert resolved[1:] == outer_paths


def test_resolved_dependency_paths_defaults_to_outer(monkeypatch):
    release_smoke = _load_release_smoke_module()
    outer_paths = [Path("/outer/three")]
    monkeypatch.setattr(release_smoke, "_outer_site_packages", lambda: outer_paths)
    monkeypatch.delenv("RELEASE_INTEROP_EXTRA_SITE_DIRS", raising=False)

    assert release_smoke._resolved_dependency_paths() == outer_paths


def test_validate_artifact_size_budget_rejects_oversized_artifact(tmp_path):
    release_smoke = _load_release_smoke_module()
    artifact = tmp_path / "demo.whl"
    artifact.write_bytes(b"12345")

    with pytest.raises(SystemExit, match="exceeds"):
        release_smoke._validate_artifact_size_budget("wheel", artifact, max_bytes=4)


def test_validate_wheel_metadata_rejects_missing_metadata_archive(tmp_path):
    release_smoke = _load_release_smoke_module()
    artifact = tmp_path / "broken.whl"
    with zipfile.ZipFile(artifact, "w") as archive:
        archive.writestr("demo/__init__.py", "")

    with pytest.raises(SystemExit, match="wheel METADATA missing"):
        release_smoke._validate_wheel_metadata(artifact)


def test_preflight_artifact_rejects_wheel_missing_required_package_files(tmp_path):
    release_smoke = _load_release_smoke_module()
    wheel_path = _write_fake_wheel(
        tmp_path,
        metadata_text="\n".join(
            [
                "Metadata-Version: 2.1",
                f"Name: {contracts.PROJECT_NAME}",
                f"Requires-Python: {contracts.REQUIRES_PYTHON}",
                *(f"Provides-Extra: {extra}" for extra in contracts.OPTIONAL_EXTRAS),
                *(
                    f"Requires-Dist: {requirement}"
                    for requirement in contracts.RELEASE_INTEROP_EXTRA_REQUIREMENTS.values()
                ),
            ]
        )
        + "\n",
    )

    with pytest.raises(SystemExit, match="missing required wheel files"):
        release_smoke._preflight_artifact("wheel", wheel_path)


def test_preflight_artifact_rejects_sdist_missing_release_support_files(tmp_path):
    release_smoke = _load_release_smoke_module()
    sdist_path = _write_fake_sdist(
        tmp_path,
        {
            "README.md": "demo\n",
            "LICENSE": "demo\n",
        },
    )

    with pytest.raises(SystemExit, match="missing required sdist files"):
        release_smoke._preflight_artifact("sdist", sdist_path)


def test_validate_artifact_size_rejects_oversized_wheel(tmp_path):
    release_smoke = _load_release_smoke_module()
    wheel_path = tmp_path / "too-large.whl"
    wheel_path.write_bytes(b"x" * (release_smoke.MAX_WHEEL_BYTES + 1))

    with pytest.raises(SystemExit, match="exceeds wheel size budget"):
        release_smoke._validate_artifact_size("wheel", wheel_path)


def test_validate_wheel_metadata_rejects_name_mismatch(tmp_path):
    release_smoke = _load_release_smoke_module()
    metadata_lines = [
        "Metadata-Version: 2.1",
        "Name: wrong-name",
        f"Requires-Python: {contracts.REQUIRES_PYTHON}",
    ]
    metadata_lines.extend(f"Provides-Extra: {extra}" for extra in contracts.OPTIONAL_EXTRAS)
    metadata_lines.extend(
        f"Requires-Dist: {requirement}"
        for requirement in contracts.RELEASE_INTEROP_EXTRA_REQUIREMENTS.values()
    )
    wheel_path = _write_fake_wheel(tmp_path, metadata_text="\n".join(metadata_lines) + "\n")

    with pytest.raises(SystemExit, match="wheel metadata name mismatch"):
        release_smoke._validate_wheel_metadata(wheel_path)
