import email
import importlib
import importlib.util
import os
import shutil
import subprocess
import sys
import tarfile
import textwrap
import zipfile
from pathlib import Path

import pytest

import scripts.release_artifact_metadata as release_artifact_metadata
from scripts.workflow_contracts import workflow_step_text
from scripts.contracts import (
    DOCS_INDEX_VERSION_BADGE,
    PROJECT_NAME,
    PROJECT_URLS,
    RELEASE_INTEROP_EXTRA_REQUIREMENTS,
    README_VERSION_BADGE,
    RELEASE_VERSION,
    WHEEL_IMPORT_SYMBOLS,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
RELEASE_SMOKE_EXTRA_SITE_DIRS_ENV = "RELEASE_INTEROP_EXTRA_SITE_DIRS"


def _load_release_smoke_module():
    script_path = REPO_ROOT / "scripts" / "release_smoke.py"
    spec = importlib.util.spec_from_file_location("release_smoke", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_install_release_extras_module(module_name: str = "install_release_extras"):
    script_path = REPO_ROOT / "scripts" / "install_release_extras.py"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _artifact_smoke_backend_available(name: str) -> bool:
    release_smoke = _load_release_smoke_module()
    bootstrap = "".join(
        f"site.addsitedir({str(path)!r})\n" for path in release_smoke._outer_site_packages()
    )
    script = "import site\n" + bootstrap + f"import {name}\n"
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.returncode == 0


def _build_release_artifacts(tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("release-dist")
    subprocess.run(
        [sys.executable, "-m", "build", "--outdir", str(output_dir)],
        cwd=REPO_ROOT,
        check=True,
    )
    wheel_path = next(output_dir.glob("*.whl"))
    sdist_path = next(output_dir.glob("*.tar.gz"))
    return wheel_path, sdist_path


def _fake_interop_packages_dir(tmp_path: Path) -> Path:
    package_root = tmp_path / "fake_packages"
    fake_packages = {
        "torch_geometric/__init__.py": "from . import data\n",
        "torch_geometric/data.py": """
class Data:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
""",
        "dgl.py": """
import torch

NID = "dgl.NID"
EID = "dgl.EID"


class FakeDGLGraph:
    def __init__(self, edges, num_nodes=None):
        self._src, self._dst = edges
        if num_nodes is None:
            num_nodes = int(torch.max(torch.cat((self._src, self._dst))).item()) + 1
        self._num_nodes = num_nodes
        self.ndata = {}
        self.edata = {}

    def edges(self):
        return self._src, self._dst

    def num_nodes(self, node_type=None):
        return self._num_nodes


def graph(edges, num_nodes=None):
    return FakeDGLGraph(edges, num_nodes=num_nodes)
""",
    }
    for relative_path, content in fake_packages.items():
        path = package_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(content), encoding="utf-8")
    return package_root


def _wheel_metadata(wheel_path: Path):
    with zipfile.ZipFile(wheel_path) as archive:
        metadata_name = next(name for name in archive.namelist() if name.endswith("METADATA"))
        return email.message_from_bytes(archive.read(metadata_name))


def _repo_relative_artifact_dir(name: str, built_release_artifacts) -> tuple[Path, str]:
    artifact_dir = REPO_ROOT / ".tmp_test_artifacts" / name
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    artifact_dir.mkdir(parents=True)
    for artifact in built_release_artifacts:
        shutil.copy2(artifact, artifact_dir / artifact.name)
    return artifact_dir, str(artifact_dir.relative_to(REPO_ROOT))


@pytest.fixture(scope="module")
def built_release_artifacts(tmp_path_factory):
    return _build_release_artifacts(tmp_path_factory)


def test_shared_release_artifact_helper_reads_wheel_metadata(built_release_artifacts):
    from scripts.release_artifact_metadata import read_wheel_metadata

    wheel_path, _ = built_release_artifacts
    metadata, detail = read_wheel_metadata(wheel_path)

    assert metadata is not None
    assert detail.endswith("METADATA")
    assert set(metadata.get_all("Provides-Extra", [])) >= {"dgl", "pyg", "full"}
    requires_dist = set(metadata.get_all("Requires-Dist", []))
    for requirement in RELEASE_INTEROP_EXTRA_REQUIREMENTS.values():
        assert requirement in requires_dist


def test_shared_release_artifact_helper_reports_missing_metadata(tmp_path):
    from scripts.release_artifact_metadata import read_wheel_metadata

    wheel_path = tmp_path / "broken.whl"
    with zipfile.ZipFile(wheel_path, "w") as archive:
        archive.writestr("demo/__init__.py", "")

    metadata, detail = read_wheel_metadata(wheel_path)

    assert metadata is None
    assert detail == "wheel METADATA missing"


def test_shared_repo_script_import_helper_loads_repo_local_modules():
    from scripts.repo_script_imports import load_repo_module

    assert load_repo_module("scripts.contracts") is importlib.import_module("scripts.contracts")


def test_top_level_repo_script_import_aliases_shared_module():
    import repo_script_imports

    shared = importlib.import_module("scripts.repo_script_imports")

    assert repo_script_imports is shared
    assert repo_script_imports.load_repo_module is shared.load_repo_module


def test_shared_repo_script_import_helper_deduplicates_repo_root(monkeypatch):
    from scripts.repo_script_imports import REPO_ROOT, ensure_repo_root_on_path

    repo_root_str = str(REPO_ROOT)
    monkeypatch.setattr(sys, "path", ["alpha", repo_root_str, "beta", repo_root_str])

    returned = ensure_repo_root_on_path()

    assert returned == REPO_ROOT
    assert sys.path[0] == repo_root_str
    assert sys.path[1:] == ["alpha", "beta"]
    assert sys.path.count(repo_root_str) == 1


def test_install_release_extras_prefers_repo_artifact_metadata_helper(monkeypatch, tmp_path):
    shadow_module = tmp_path / "release_artifact_metadata.py"
    shadow_module.write_text(
        textwrap.dedent(
            """
            def read_wheel_metadata(_wheel_path):
                return None, "shadow helper"
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.syspath_prepend(str(tmp_path))
    shadowed = sys.modules.pop("release_artifact_metadata", None)
    try:
        install_release_extras = _load_install_release_extras_module("install_release_extras_shadowed")
    finally:
        sys.modules.pop("install_release_extras_shadowed", None)
        sys.modules.pop("release_artifact_metadata", None)
        if shadowed is not None:
            sys.modules["release_artifact_metadata"] = shadowed

    assert install_release_extras.read_wheel_metadata is release_artifact_metadata.read_wheel_metadata


def test_release_metadata_exposes_public_package_information(built_release_artifacts):
    wheel_path, _ = built_release_artifacts
    metadata = _wheel_metadata(wheel_path)
    project_urls = metadata.get_all("Project-URL", [])
    requires_dist = set(metadata.get_all("Requires-Dist", []))

    assert metadata["Name"] == PROJECT_NAME
    assert metadata["Requires-Python"] == ">=3.10"
    assert metadata["Author-email"]
    assert metadata["Keywords"]
    assert metadata.get_all("Classifier")
    for label, url in PROJECT_URLS.items():
        assert f"{label}, {url}" in project_urls
    assert set(metadata.get_all("Provides-Extra", [])) >= {
        "dev",
        "scipy",
        "networkx",
        "tensorboard",
        "dgl",
        "pyg",
        "full",
    }
    for requirement in RELEASE_INTEROP_EXTRA_REQUIREMENTS.values():
        assert requirement in requires_dist


def test_release_artifacts_exclude_internal_repo_only_content(built_release_artifacts):
    wheel_path, sdist_path = built_release_artifacts

    with zipfile.ZipFile(wheel_path) as archive:
        wheel_names = archive.namelist()
    with tarfile.open(sdist_path) as archive:
        sdist_names = archive.getnames()

    assert not any("/.factory/" in name for name in wheel_names)
    assert not any("/.factory/" in name for name in sdist_names)
    assert not any("/docs/plans/" in name for name in wheel_names)
    assert not any("/docs/plans/" in name for name in sdist_names)
    assert not any("__pycache__" in name for name in wheel_names)
    assert not any("__pycache__" in name for name in sdist_names)
    assert not any(name.startswith("examples/") for name in wheel_names)
    assert any("/examples/homo/node_classification.py" in name for name in sdist_names)
    assert any(name.endswith("/README.md") for name in sdist_names)
    assert any(name.endswith("/LICENSE") for name in sdist_names)
    assert any(name.endswith("/scripts/release_smoke.py") for name in sdist_names)
    assert any(name.endswith("/scripts/interop_smoke.py") for name in sdist_names)
    assert any(name.endswith("/scripts/install_release_extras.py") for name in sdist_names)
    assert any(name.endswith("/scripts/release_artifact_metadata.py") for name in sdist_names)
    assert any(name.endswith("/scripts/repo_script_imports.py") for name in sdist_names)


def test_release_workflows_exist_for_ci_and_pypi_publish():
    ci_path = REPO_ROOT / ".github" / "workflows" / "ci.yml"
    publish_path = REPO_ROOT / ".github" / "workflows" / "publish.yml"
    smoke_script = REPO_ROOT / "scripts" / "release_smoke.py"
    interop_script = REPO_ROOT / "scripts" / "interop_smoke.py"
    install_release_extras_script = REPO_ROOT / "scripts" / "install_release_extras.py"

    assert ci_path.exists()
    assert publish_path.exists()
    assert smoke_script.exists()
    assert interop_script.exists()
    assert install_release_extras_script.exists()

    ci_text = ci_path.read_text(encoding="utf-8")
    publish_text = publish_path.read_text(encoding="utf-8")
    package_check_install_step = workflow_step_text(ci_path, "package-check", "Install release interop extras")
    package_check_smoke_step = workflow_step_text(
        ci_path,
        "package-check",
        "Smoke-test built distributions with all interop backends",
    )
    publish_build_install_step = workflow_step_text(
        publish_path,
        "build",
        "Install release interop extras",
    )
    publish_build_smoke_step = workflow_step_text(
        publish_path,
        "build",
        "Smoke-test built distributions with all interop backends",
    )

    assert "python -m pytest -q" in ci_text
    assert "python -m mypy vgl" in ci_text
    assert "python -m mkdocs build --strict" in ci_text
    assert "python -m build" in ci_text
    assert "python -m twine check" in ci_text
    assert "python scripts/docs_link_scan.py" in ci_text
    assert "python scripts/release_contract_scan.py --artifact-dir dist" in ci_text
    assert "python scripts/release_smoke.py --artifact-dir dist --kind all" in ci_text
    assert "python scripts/install_release_extras.py --artifact-dir dist --extras pyg dgl" in package_check_install_step
    assert 'python -m pip install -e ".[pyg,dgl]"' not in package_check_install_step
    assert "python scripts/release_smoke.py --artifact-dir dist --kind all --interop-backend all" in package_check_smoke_step
    assert 'python -m pip install -e ".[pyg,dgl]"' not in package_check_smoke_step
    assert "python scripts/metadata_consistency.py" in (REPO_ROOT / "docs" / "releasing.md").read_text(encoding="utf-8")
    assert "tags:" in publish_text
    assert "v*" in publish_text
    assert "testpypi" in publish_text.lower()
    assert "pypi" in publish_text.lower()
    assert "id-token: write" in publish_text
    assert "PYPI_API_TOKEN" in publish_text
    assert "TEST_PYPI_API_TOKEN" in publish_text
    assert "probe-publish-auth:" in publish_text
    assert "needs.probe-publish-auth.outputs.has_pypi_api_token" in publish_text
    assert "needs.probe-publish-auth.outputs.has_test_pypi_api_token" in publish_text
    assert "GITHUB_OUTPUT" in publish_text
    assert "python scripts/release_smoke.py --artifact-dir dist --kind all" in publish_text
    assert "python scripts/install_release_extras.py --artifact-dir dist --extras pyg dgl" in publish_build_install_step
    assert 'python -m pip install -e ".[pyg,dgl]"' not in publish_build_install_step
    assert "python scripts/release_smoke.py --artifact-dir dist --kind all --interop-backend all" in publish_build_smoke_step
    assert 'python -m pip install -e ".[pyg,dgl]"' not in publish_build_smoke_step
    assert "Publish to PyPI with API token" in publish_text
    assert "Publish to PyPI with Trusted Publishing" in publish_text
    assert "Publish to TestPyPI with API token" in publish_text
    assert "Publish to TestPyPI with Trusted Publishing" in publish_text


def test_docs_publish_workflow_exists_for_github_pages():
    docs_path = REPO_ROOT / ".github" / "workflows" / "docs.yml"

    assert docs_path.exists()

    docs_text = docs_path.read_text(encoding="utf-8")

    assert "python -m mkdocs build --strict" in docs_text
    assert "actions/configure-pages@v5" in docs_text
    assert "actions/upload-pages-artifact@v3" in docs_text
    assert "actions/deploy-pages@v4" in docs_text
    assert "pages: write" in docs_text
    assert "id-token: write" in docs_text
    assert "path: site" in docs_text


def test_install_release_extras_prints_selected_artifact_requirements(built_release_artifacts):
    wheel_path, _ = built_release_artifacts
    script = REPO_ROOT / "scripts" / "install_release_extras.py"

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--artifact-dir",
            str(wheel_path.parent),
            "--extras",
            "pyg",
            "dgl",
            "--print-only",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    requirements = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    assert requirements == ["torch-geometric>=2.5", "dgl>=2.1"]


def test_install_release_extras_rejects_unknown_extras(built_release_artifacts):
    wheel_path, _ = built_release_artifacts
    script = REPO_ROOT / "scripts" / "install_release_extras.py"

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--artifact-dir",
            str(wheel_path.parent),
            "--extras",
            "does-not-exist",
            "--print-only",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "does-not-exist" in completed.stderr


def test_install_release_extras_resolves_relative_artifact_dir_from_repo_root(
    built_release_artifacts,
    tmp_path: Path,
):
    script = REPO_ROOT / "scripts" / "install_release_extras.py"
    artifact_dir, relative_artifact_dir = _repo_relative_artifact_dir(
        "install-release-extras-relative",
        built_release_artifacts,
    )
    try:
        completed = subprocess.run(
            [
                sys.executable,
                str(script),
                "--artifact-dir",
                relative_artifact_dir,
                "--extras",
                "pyg",
                "dgl",
                "--print-only",
            ],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )
    finally:
        shutil.rmtree(artifact_dir)

    assert completed.returncode == 0, completed.stderr
    requirements = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    assert requirements == ["torch-geometric>=2.5", "dgl>=2.1"]


def test_generated_site_directory_is_ignored_and_not_tracked():
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    tracked_site = subprocess.run(
        ["git", "ls-files", "site"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "site/" in gitignore
    assert tracked_site.stdout.strip() == ""


def test_release_dev_dependencies_include_docs_build_stack():
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'mkdocs>=' in pyproject
    assert 'mkdocs-material>=' in pyproject
    assert 'mkdocstrings[python]>=' in pyproject


def test_release_dev_dependencies_include_packaging_for_artifact_helper():
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'packaging>=' in pyproject


def test_mkdocs_config_marks_internal_docs_as_not_in_nav():
    mkdocs = (REPO_ROOT / "mkdocs.yml").read_text(encoding="utf-8")

    assert "not_in_nav:" in mkdocs
    assert "plans/*" in mkdocs
    assert "public-surface-contract.md" in mkdocs
    assert "releasing.md" in mkdocs


def test_release_smoke_script_can_install_built_wheel(built_release_artifacts):
    wheel_path, _ = built_release_artifacts
    smoke_script = REPO_ROOT / "scripts" / "release_smoke.py"

    completed = subprocess.run(
        [
            sys.executable,
            str(smoke_script),
            "--artifact-dir",
            str(wheel_path.parent),
            "--kind",
            "wheel",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert f"wheel smoke check passed for {wheel_path.name}" in completed.stdout


def test_release_smoke_resolves_relative_artifact_dir_from_repo_root(
    built_release_artifacts,
    tmp_path: Path,
):
    wheel_path, _ = built_release_artifacts
    smoke_script = REPO_ROOT / "scripts" / "release_smoke.py"
    artifact_dir, relative_artifact_dir = _repo_relative_artifact_dir(
        "release-smoke-relative",
        built_release_artifacts,
    )
    try:
        completed = subprocess.run(
            [
                sys.executable,
                str(smoke_script),
                "--artifact-dir",
                relative_artifact_dir,
                "--kind",
                "wheel",
            ],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )
    finally:
        shutil.rmtree(artifact_dir)

    assert completed.returncode == 0, completed.stderr
    assert f"wheel smoke check passed for {wheel_path.name}" in completed.stdout


def test_release_smoke_script_accepts_disabled_interop_backend_flag(built_release_artifacts):
    wheel_path, _ = built_release_artifacts
    smoke_script = REPO_ROOT / "scripts" / "release_smoke.py"

    completed = subprocess.run(
        [
            sys.executable,
            str(smoke_script),
            "--artifact-dir",
            str(wheel_path.parent),
            "--kind",
            "wheel",
            "--interop-backend",
            "none",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert f"wheel smoke check passed for {wheel_path.name}" in completed.stdout


def test_release_smoke_script_supports_artifact_interop_backend_pyg_with_fake_backend(
    built_release_artifacts,
    tmp_path,
):
    wheel_path, _ = built_release_artifacts
    smoke_script = REPO_ROOT / "scripts" / "release_smoke.py"
    env = os.environ.copy()
    env[RELEASE_SMOKE_EXTRA_SITE_DIRS_ENV] = str(_fake_interop_packages_dir(tmp_path))

    completed = subprocess.run(
        [
            sys.executable,
            str(smoke_script),
            "--artifact-dir",
            str(wheel_path.parent),
            "--kind",
            "wheel",
            "--interop-backend",
            "pyg",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode == 0, completed.stderr
    assert "pyg interop smoke check passed" in completed.stdout
    assert f"wheel smoke check passed for {wheel_path.name}" in completed.stdout


def test_release_smoke_script_supports_artifact_interop_backend_all_with_fake_backends(
    built_release_artifacts,
    tmp_path,
):
    wheel_path, _ = built_release_artifacts
    smoke_script = REPO_ROOT / "scripts" / "release_smoke.py"
    env = os.environ.copy()
    env[RELEASE_SMOKE_EXTRA_SITE_DIRS_ENV] = str(_fake_interop_packages_dir(tmp_path))

    completed = subprocess.run(
        [
            sys.executable,
            str(smoke_script),
            "--artifact-dir",
            str(wheel_path.parent),
            "--kind",
            "wheel",
            "--interop-backend",
            "all",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode == 0, completed.stderr
    assert "pyg interop smoke check passed" in completed.stdout
    assert "dgl interop smoke check passed" in completed.stdout
    assert f"wheel smoke check passed for {wheel_path.name}" in completed.stdout


@pytest.mark.skipif(
    not _artifact_smoke_backend_available("dgl"),
    reason="dgl is not available to release_smoke artifact checks",
)
def test_release_smoke_script_supports_artifact_interop_backend_dgl(built_release_artifacts):
    wheel_path, _ = built_release_artifacts
    smoke_script = REPO_ROOT / "scripts" / "release_smoke.py"

    completed = subprocess.run(
        [
            sys.executable,
            str(smoke_script),
            "--artifact-dir",
            str(wheel_path.parent),
            "--kind",
            "wheel",
            "--interop-backend",
            "dgl",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "dgl interop smoke check passed" in completed.stdout
    assert f"wheel smoke check passed for {wheel_path.name}" in completed.stdout


@pytest.mark.skipif(
    not _artifact_smoke_backend_available("dgl"),
    reason="dgl is not available to release_smoke artifact checks",
)
def test_release_smoke_script_supports_sdist_artifact_interop_backend_dgl(built_release_artifacts):
    _, sdist_path = built_release_artifacts
    smoke_script = REPO_ROOT / "scripts" / "release_smoke.py"

    completed = subprocess.run(
        [
            sys.executable,
            str(smoke_script),
            "--artifact-dir",
            str(sdist_path.parent),
            "--kind",
            "sdist",
            "--interop-backend",
            "dgl",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "dgl interop smoke check passed" in completed.stdout
    assert f"sdist smoke check passed for {sdist_path.name}" in completed.stdout


@pytest.mark.skipif(
    not _artifact_smoke_backend_available("dgl") or _artifact_smoke_backend_available("pyg"),
    reason="requires dgl available and pyg unavailable for artifact smoke",
)
def test_release_smoke_script_reports_missing_backend_for_all_artifact_interop(
    built_release_artifacts,
):
    wheel_path, _ = built_release_artifacts
    smoke_script = REPO_ROOT / "scripts" / "release_smoke.py"

    completed = subprocess.run(
        [
            sys.executable,
            str(smoke_script),
            "--artifact-dir",
            str(wheel_path.parent),
            "--kind",
            "wheel",
            "--interop-backend",
            "all",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "artifact interop backend(s) unavailable from outer site-packages" in completed.stderr
    assert "pyg (torch_geometric)" in completed.stderr


def test_release_readme_documents_public_install_paths():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    docs_index = (REPO_ROOT / "docs" / "index.md").read_text(encoding="utf-8")
    installation = (REPO_ROOT / "docs" / "getting-started" / "installation.md").read_text(encoding="utf-8")
    quickstart = (REPO_ROOT / "docs" / "public-surface-contract.md").read_text(encoding="utf-8")
    releasing = (REPO_ROOT / "docs" / "releasing.md").read_text(encoding="utf-8")
    support_matrix = (REPO_ROOT / "docs" / "support-matrix.md").read_text(encoding="utf-8")

    assert README_VERSION_BADGE in readme
    assert DOCS_INDEX_VERSION_BADGE in docs_index
    assert f"version-{RELEASE_VERSION}" not in readme
    assert f"version-{RELEASE_VERSION}" not in docs_index
    assert 'pip install sky-vgl' in readme
    assert 'pip install "sky-vgl[full]"' in readme
    assert 'pip install "sky-vgl[networkx]"' in readme
    assert 'pip install "sky-vgl[pyg]"' in readme
    assert 'pip install "sky-vgl[dgl]"' in readme
    assert "git clone https://github.com/skygazer42/sky-vgl.git" in readme
    assert "cd sky-vgl" in readme
    assert 'src="assets/logo.svg"' in readme
    assert 'src="assets/graph-types.svg"' in readme
    assert 'src="assets/architecture.svg"' in readme
    assert 'src="assets/pipeline.svg"' in readme
    assert 'src="assets/conv-layers.svg"' in readme
    assert "raw.githubusercontent.com/skygazer42/sky-vgl/main/assets/" not in readme
    assert "pip install sky-vgl" in quickstart
    assert 'pip install "sky-vgl[full]"' in quickstart
    assert "installed release version" in installation
    assert "sky-vgl project name" in releasing
    assert "sky-vgl` works in a clean environment" in releasing
    assert "pending publisher" in releasing
    assert "Manage Project -> Publishing" in releasing
    assert "PYPI_API_TOKEN" in releasing
    assert "TEST_PYPI_API_TOKEN" in releasing
    assert "host-installed" in releasing
    assert "host-assisted dependency discovery" in releasing
    assert "--interop-backend all" in releasing
    assert "both host backends" in releasing
    assert "fail early" in releasing
    assert "hermetic fake-backend success paths" in releasing
    assert "package-check" in releasing
    assert "publish `build` job" in releasing
    assert "python scripts/install_release_extras.py --artifact-dir dist --extras pyg dgl" in releasing
    assert "python scripts/release_smoke.py --artifact-dir dist --kind all --interop-backend all" in releasing
    assert "python scripts/release_smoke.py --artifact-dir dist --kind all" in releasing
    assert "python scripts/release_smoke.py --artifact-dir dist --kind wheel --interop-backend dgl" in releasing
    assert "python scripts/release_smoke.py --artifact-dir dist --kind wheel --interop-backend all" in releasing
    assert "interop-smoke" in releasing
    assert "builds the artifacts" in releasing
    assert "python scripts/interop_smoke.py --backend all" in releasing
    assert "providing automated artifact-level verification for the combined backend scenario" in support_matrix
    for symbol in WHEEL_IMPORT_SYMBOLS:
        assert symbol in releasing
