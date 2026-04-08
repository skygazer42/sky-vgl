import importlib.util
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import scripts.contracts as contracts
from scripts.contracts import (
    OPTIONAL_EXTRAS,
    PROJECT_URLS,
    RELEASE_INTEROP_EXTRA_REQUIREMENTS,
    SDIST_EXCLUDED_SUBSTRINGS,
    SDIST_REQUIRED_SUFFIXES,
    WHEEL_EXCLUDED_SUBSTRINGS,
    WHEEL_REQUIRED_FILES,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SCAN_SCRIPT = REPO_ROOT / "scripts" / "release_contract_scan.py"


def _load_release_contract_scan(module_name: str = "release_contract_scan"):
    spec = importlib.util.spec_from_file_location(module_name, SCAN_SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _repo_relative_artifact_dir(name: str, source_dir: Path) -> tuple[Path, str]:
    artifact_dir = REPO_ROOT / ".tmp_test_artifacts" / name
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    artifact_dir.mkdir(parents=True)
    for artifact in source_dir.iterdir():
        if artifact.is_file():
            shutil.copy2(artifact, artifact_dir / artifact.name)
    return artifact_dir, str(artifact_dir.relative_to(REPO_ROOT))


@pytest.fixture(scope="module")
def built_artifact_dir(tmp_path_factory) -> Path:
    output_dir = tmp_path_factory.mktemp("release-contract-dist")
    subprocess.run(
        [sys.executable, "-m", "build", "--outdir", str(output_dir)],
        cwd=REPO_ROOT,
        check=True,
    )
    return output_dir


def test_release_contract_scan_prefers_repo_contracts_module(monkeypatch, tmp_path):
    shadow_module = tmp_path / "contracts.py"
    shadow_module.write_text(
        textwrap.dedent(
            """
            OPTIONAL_EXTRAS = ()
            PROJECT_NAME = "shadow-project"
            PROJECT_URLS = {}
            RELEASE_INTEROP_EXTRA_REQUIREMENTS = {"pyg": "shadow"}
            REQUIRES_PYTHON = ">=9.9"
            SDIST_EXCLUDED_SUBSTRINGS = ()
            SDIST_REQUIRED_SUFFIXES = ()
            WHEEL_EXCLUDED_SUBSTRINGS = ()
            WHEEL_REQUIRED_FILES = ()
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.syspath_prepend(str(tmp_path))
    shadowed = sys.modules.pop("contracts", None)
    try:
        scan = _load_release_contract_scan("release_contract_scan_shadowed")
    finally:
        sys.modules.pop("release_contract_scan_shadowed", None)
        if shadowed is not None:
            sys.modules["contracts"] = shadowed

    assert scan.PROJECT_NAME == contracts.PROJECT_NAME
    assert scan.RELEASE_INTEROP_EXTRA_REQUIREMENTS == contracts.RELEASE_INTEROP_EXTRA_REQUIREMENTS


def test_release_artifact_helper_reads_built_contract_scan_wheel(built_artifact_dir: Path):
    from scripts.release_artifact_metadata import read_wheel_metadata

    wheel_path = next(built_artifact_dir.glob("*.whl"))
    metadata, detail = read_wheel_metadata(wheel_path)

    assert metadata is not None
    assert detail.endswith("METADATA")
    requires_dist = set(metadata.get_all("Requires-Dist", []))
    for requirement in RELEASE_INTEROP_EXTRA_REQUIREMENTS.values():
        assert requirement in requires_dist


def test_release_contract_scan_lists_stable_catalog():
    completed = subprocess.run(
        [sys.executable, str(SCAN_SCRIPT), "--list"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    listed = [line for line in completed.stdout.splitlines() if line.startswith("SCAN ")]
    expected = 5 + len(PROJECT_URLS) + len(OPTIONAL_EXTRAS) + len(WHEEL_REQUIRED_FILES)
    expected += len(WHEEL_EXCLUDED_SUBSTRINGS) + len(SDIST_REQUIRED_SUFFIXES) + len(SDIST_EXCLUDED_SUBSTRINGS)
    expected += len(RELEASE_INTEROP_EXTRA_REQUIREMENTS)
    assert len(listed) == expected
    for extra in RELEASE_INTEROP_EXTRA_REQUIREMENTS:
        assert any(f"wheel metadata exposes {extra} extra requirement line" in line for line in listed)
    assert any("sdist contains /scripts/install_release_extras.py" in line for line in listed)
    assert any("sdist contains /scripts/release_artifact_metadata.py" in line for line in listed)
    assert any("sdist contains /scripts/repo_script_imports.py" in line for line in listed)


def test_release_contract_scan_task_ids_follow_contract_cardinality(monkeypatch, tmp_path: Path):
    scan = _load_release_contract_scan("release_contract_scan_id_probe")
    try:
        fake_contracts = type(
            "FakeContracts",
            (),
            {
                "OPTIONAL_EXTRAS": ("lite",),
                "PROJECT_NAME": "sky-vgl",
                "PROJECT_URLS": {"Homepage": "https://example.invalid/project"},
                "RELEASE_INTEROP_EXTRA_REQUIREMENTS": {"lite": "lite-dep>=1.0"},
                "REQUIRES_PYTHON": ">=3.10",
                "SDIST_EXCLUDED_SUBSTRINGS": ("/internal/",),
                "SDIST_REQUIRED_SUFFIXES": ("/README.md",),
                "WHEEL_EXCLUDED_SUBSTRINGS": ("tests/",),
                "WHEEL_REQUIRED_FILES": ("vgl/__init__.py",),
            },
        )()

        monkeypatch.setattr(scan, "_contracts_module", lambda: fake_contracts)
        monkeypatch.setattr(
            scan,
            "_list_contract_values",
            lambda _repo_root: {
                "OPTIONAL_EXTRAS": fake_contracts.OPTIONAL_EXTRAS,
                "PROJECT_URLS": fake_contracts.PROJECT_URLS,
                "RELEASE_INTEROP_EXTRA_REQUIREMENTS": fake_contracts.RELEASE_INTEROP_EXTRA_REQUIREMENTS,
                "SDIST_EXCLUDED_SUBSTRINGS": fake_contracts.SDIST_EXCLUDED_SUBSTRINGS,
                "SDIST_REQUIRED_SUFFIXES": fake_contracts.SDIST_REQUIRED_SUFFIXES,
                "WHEEL_EXCLUDED_SUBSTRINGS": fake_contracts.WHEEL_EXCLUDED_SUBSTRINGS,
                "WHEEL_REQUIRED_FILES": fake_contracts.WHEEL_REQUIRED_FILES,
            },
        )

        built_tasks = scan.build_tasks(REPO_ROOT, tmp_path, validate_repo_contracts=False)
        listed_tasks = scan.list_task_specs(REPO_ROOT)
    finally:
        sys.modules.pop("release_contract_scan_id_probe", None)

    expected_ids = [f"{index:03d}" for index in range(1, 13)]
    assert [task.id for task in built_tasks] == expected_ids
    assert [task.id for task in listed_tasks] == expected_ids


def test_release_contract_scan_list_catalog_follows_named_tuple_composition(tmp_path: Path):
    repo_root = tmp_path / "repo"
    contracts_path = repo_root / "scripts" / "contracts.py"
    contracts_path.parent.mkdir(parents=True)
    contracts_path.write_text(
        textwrap.dedent(
            """
            BASE_EXTRAS = ("lite",)
            OPTIONAL_EXTRAS = BASE_EXTRAS + ("full",)
            PROJECT_URLS = {"Homepage": "https://example.invalid/project"}
            RELEASE_INTEROP_EXTRA_REQUIREMENTS = {"lite": "lite-dep>=1.0"}
            SDIST_EXCLUDED_SUBSTRINGS = ("/internal/",)
            SDIST_REQUIRED_SUFFIXES = ("/README.md",)
            WHEEL_EXCLUDED_SUBSTRINGS = ("tests/",)
            CORE_WHEEL_FILES = ("vgl/__init__.py",)
            WHEEL_REQUIRED_FILES = CORE_WHEEL_FILES + ("vgl/version.py",)
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    scan = _load_release_contract_scan("release_contract_scan_composition_probe")
    try:
        listed_tasks = scan.list_task_specs(repo_root)
    finally:
        sys.modules.pop("release_contract_scan_composition_probe", None)

    assert [task.description for task in listed_tasks] == [
        "built wheel exists",
        "built sdist exists",
        "wheel name matches project",
        "wheel version matches repo version",
        "wheel Requires-Python matches pyproject",
        "wheel exposes Homepage project URL",
        "wheel provides extra lite",
        "wheel provides extra full",
        "wheel metadata exposes lite extra requirement line lite-dep>=1.0",
        "wheel contains vgl/__init__.py",
        "wheel contains vgl/version.py",
        "wheel excludes tests/",
        "sdist contains /README.md",
        "sdist excludes /internal/",
    ]


def test_release_contract_scan_list_catalog_follows_named_dict_union(tmp_path: Path):
    repo_root = tmp_path / "repo"
    contracts_path = repo_root / "scripts" / "contracts.py"
    contracts_path.parent.mkdir(parents=True)
    contracts_path.write_text(
        textwrap.dedent(
            """
            BASE_URLS = {"Homepage": "https://example.invalid/project"}
            PROJECT_URLS = BASE_URLS | {"Documentation": "https://example.invalid/docs"}
            OPTIONAL_EXTRAS = ("lite",)
            RELEASE_INTEROP_EXTRA_REQUIREMENTS = {"lite": "lite-dep>=1.0"}
            SDIST_EXCLUDED_SUBSTRINGS = ("/internal/",)
            SDIST_REQUIRED_SUFFIXES = ("/README.md",)
            WHEEL_EXCLUDED_SUBSTRINGS = ("tests/",)
            WHEEL_REQUIRED_FILES = ("vgl/__init__.py",)
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    scan = _load_release_contract_scan("release_contract_scan_dict_union_probe")
    try:
        listed_tasks = scan.list_task_specs(repo_root)
    finally:
        sys.modules.pop("release_contract_scan_dict_union_probe", None)

    assert [task.description for task in listed_tasks] == [
        "built wheel exists",
        "built sdist exists",
        "wheel name matches project",
        "wheel version matches repo version",
        "wheel Requires-Python matches pyproject",
        "wheel exposes Homepage project URL",
        "wheel exposes Documentation project URL",
        "wheel provides extra lite",
        "wheel metadata exposes lite extra requirement line lite-dep>=1.0",
        "wheel contains vgl/__init__.py",
        "wheel excludes tests/",
        "sdist contains /README.md",
        "sdist excludes /internal/",
    ]


def test_release_contract_scan_list_catalog_supports_annotated_assignments(tmp_path: Path):
    repo_root = tmp_path / "repo"
    contracts_path = repo_root / "scripts" / "contracts.py"
    contracts_path.parent.mkdir(parents=True)
    contracts_path.write_text(
        textwrap.dedent(
            """
            BASE_EXTRAS: tuple[str, ...] = ("lite",)
            OPTIONAL_EXTRAS: tuple[str, ...] = BASE_EXTRAS + ("full",)
            PROJECT_URLS: dict[str, str] = {"Homepage": "https://example.invalid/project"}
            RELEASE_INTEROP_EXTRA_REQUIREMENTS: dict[str, str] = {"lite": "lite-dep>=1.0"}
            SDIST_EXCLUDED_SUBSTRINGS: tuple[str, ...] = ("/internal/",)
            SDIST_REQUIRED_SUFFIXES: tuple[str, ...] = ("/README.md",)
            WHEEL_EXCLUDED_SUBSTRINGS: tuple[str, ...] = ("tests/",)
            WHEEL_REQUIRED_FILES: tuple[str, ...] = ("vgl/__init__.py",)
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    scan = _load_release_contract_scan("release_contract_scan_annotated_probe")
    try:
        listed_tasks = scan.list_task_specs(repo_root)
    finally:
        sys.modules.pop("release_contract_scan_annotated_probe", None)

    assert [task.description for task in listed_tasks] == [
        "built wheel exists",
        "built sdist exists",
        "wheel name matches project",
        "wheel version matches repo version",
        "wheel Requires-Python matches pyproject",
        "wheel exposes Homepage project URL",
        "wheel provides extra lite",
        "wheel provides extra full",
        "wheel metadata exposes lite extra requirement line lite-dep>=1.0",
        "wheel contains vgl/__init__.py",
        "wheel excludes tests/",
        "sdist contains /README.md",
        "sdist excludes /internal/",
    ]


def test_release_contract_scan_passes_on_built_artifacts(built_artifact_dir: Path):
    completed = subprocess.run(
        [
            sys.executable,
            str(SCAN_SCRIPT),
            "--artifact-dir",
            str(built_artifact_dir),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    expected = 5 + len(PROJECT_URLS) + len(OPTIONAL_EXTRAS) + len(WHEEL_REQUIRED_FILES)
    expected += len(WHEEL_EXCLUDED_SUBSTRINGS) + len(SDIST_REQUIRED_SUFFIXES) + len(SDIST_EXCLUDED_SUBSTRINGS)
    expected += len(RELEASE_INTEROP_EXTRA_REQUIREMENTS)
    assert f"SUMMARY {expected}/{expected} passed" in completed.stdout
    for requirement in RELEASE_INTEROP_EXTRA_REQUIREMENTS.values():
        assert requirement in completed.stdout


def test_release_contract_scan_resolves_relative_artifact_dir_from_repo_root(
    built_artifact_dir: Path,
    tmp_path: Path,
):
    artifact_dir, relative_artifact_dir = _repo_relative_artifact_dir(
        "release-contract-scan-relative",
        built_artifact_dir,
    )
    try:
        completed = subprocess.run(
            [
                sys.executable,
                str(SCAN_SCRIPT),
                "--artifact-dir",
                relative_artifact_dir,
            ],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )
    finally:
        shutil.rmtree(artifact_dir)

    assert completed.returncode == 0, completed.stderr
    assert "SUMMARY" in completed.stdout
