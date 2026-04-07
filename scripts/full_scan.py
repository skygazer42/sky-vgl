#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

try:
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib  # type: ignore[no-redef]


CheckFn = Callable[[], tuple[bool, str]]


def _load_workflow_contracts_module():
    repo_root = Path(__file__).resolve().parent.parent
    repo_root_str = str(repo_root)
    if repo_root_str in sys.path:
        sys.path.remove(repo_root_str)
    sys.path.insert(0, repo_root_str)
    return importlib.import_module("scripts.workflow_contracts")


workflow_contracts = _load_workflow_contracts_module()
workflow_job_contains_text = workflow_contracts.workflow_job_contains_text
workflow_step_contains_text = workflow_contracts.workflow_step_contains_text
workflow_step_lacks_text = workflow_contracts.workflow_step_lacks_text


@dataclass(frozen=True)
class ScanTask:
    id: str
    category: str
    description: str
    check: CheckFn


@dataclass(frozen=True)
class ScanResult:
    task: ScanTask
    passed: bool
    details: str


class ScanContext:
    def __init__(self, repo_root: Path):
        self.repo_root = Path(repo_root).resolve()
        self._text_cache: dict[Path, str] = {}
        self._pyproject_cache: dict | None = None

    def resolve(self, relative_path: str) -> Path:
        return self.repo_root / relative_path

    def exists(self, relative_path: str) -> tuple[bool, str]:
        path = self.resolve(relative_path)
        return path.exists(), relative_path

    def is_dir(self, relative_path: str) -> tuple[bool, str]:
        path = self.resolve(relative_path)
        return path.is_dir(), relative_path

    def contains(self, relative_path: str, snippet: str) -> tuple[bool, str]:
        text = self._read_text(relative_path)
        return snippet in text, f"{relative_path} contains {snippet!r}"

    def workflow_job_contains(self, relative_path: str, job_name: str, snippet: str) -> tuple[bool, str]:
        return workflow_job_contains_text(
            self._read_text(relative_path),
            job_name,
            snippet,
            source=relative_path,
        )

    def workflow_step_contains(
        self,
        relative_path: str,
        job_name: str,
        step_name: str,
        snippet: str,
    ) -> tuple[bool, str]:
        return workflow_step_contains_text(
            self._read_text(relative_path),
            job_name,
            step_name,
            snippet,
            source=relative_path,
        )

    def workflow_step_lacks(
        self,
        relative_path: str,
        job_name: str,
        step_name: str,
        snippet: str,
    ) -> tuple[bool, str]:
        return workflow_step_lacks_text(
            self._read_text(relative_path),
            job_name,
            step_name,
            snippet,
            source=relative_path,
        )

    def pyproject_value(self, *keys: str) -> object:
        payload: object = self._load_pyproject()
        for key in keys:
            if not isinstance(payload, dict):
                raise KeyError(" -> ".join(keys))
            payload = payload[key]
        return payload

    def _read_text(self, relative_path: str) -> str:
        path = self.resolve(relative_path)
        cached = self._text_cache.get(path)
        if cached is None:
            cached = path.read_text(encoding="utf-8")
            self._text_cache[path] = cached
        return cached

    def _load_pyproject(self) -> dict:
        if self._pyproject_cache is None:
            with self.resolve("pyproject.toml").open("rb") as handle:
                self._pyproject_cache = tomllib.load(handle)
        return self._pyproject_cache


def _path_exists_task(
    ctx: ScanContext,
    task_id: str,
    category: str,
    description: str,
    relative_path: str,
    *,
    directory: bool = False,
) -> ScanTask:
    def check() -> tuple[bool, str]:
        if directory:
            return ctx.is_dir(relative_path)
        return ctx.exists(relative_path)

    return ScanTask(task_id, category, description, check)


def _text_contains_task(
    ctx: ScanContext,
    task_id: str,
    category: str,
    description: str,
    relative_path: str,
    snippet: str,
) -> ScanTask:
    def check() -> tuple[bool, str]:
        return ctx.contains(relative_path, snippet)

    return ScanTask(task_id, category, description, check)


def _workflow_job_contains_task(
    ctx: ScanContext,
    task_id: str,
    category: str,
    description: str,
    relative_path: str,
    job_name: str,
    snippet: str,
) -> ScanTask:
    def check() -> tuple[bool, str]:
        return ctx.workflow_job_contains(relative_path, job_name, snippet)

    return ScanTask(task_id, category, description, check)


def _workflow_step_contains_task(
    ctx: ScanContext,
    task_id: str,
    category: str,
    description: str,
    relative_path: str,
    job_name: str,
    step_name: str,
    snippet: str,
) -> ScanTask:
    def check() -> tuple[bool, str]:
        return ctx.workflow_step_contains(relative_path, job_name, step_name, snippet)

    return ScanTask(task_id, category, description, check)


def _workflow_step_lacks_task(
    ctx: ScanContext,
    task_id: str,
    category: str,
    description: str,
    relative_path: str,
    job_name: str,
    step_name: str,
    snippet: str,
) -> ScanTask:
    def check() -> tuple[bool, str]:
        return ctx.workflow_step_lacks(relative_path, job_name, step_name, snippet)

    return ScanTask(task_id, category, description, check)


def _pyproject_equals_task(
    ctx: ScanContext,
    task_id: str,
    category: str,
    description: str,
    keys: tuple[str, ...],
    expected: object,
) -> ScanTask:
    def check() -> tuple[bool, str]:
        value = ctx.pyproject_value(*keys)
        return value == expected, f"{'.'.join(keys)} == {expected!r}"

    return ScanTask(task_id, category, description, check)


def _pyproject_contains_task(
    ctx: ScanContext,
    task_id: str,
    category: str,
    description: str,
    keys: tuple[str, ...],
    expected: object,
) -> ScanTask:
    def check() -> tuple[bool, str]:
        values = ctx.pyproject_value(*keys)
        if not isinstance(values, (list, tuple)):
            return False, f"{'.'.join(keys)} is not a sequence"
        return expected in values, f"{expected!r} in {'.'.join(keys)}"

    return ScanTask(task_id, category, description, check)


def _pyproject_mapping_has_key_task(
    ctx: ScanContext,
    task_id: str,
    category: str,
    description: str,
    keys: tuple[str, ...],
    expected_key: str,
) -> ScanTask:
    def check() -> tuple[bool, str]:
        value = ctx.pyproject_value(*keys)
        if not isinstance(value, dict):
            return False, f"{'.'.join(keys)} is not a mapping"
        return expected_key in value, f"{expected_key!r} in {'.'.join(keys)}"

    return ScanTask(task_id, category, description, check)


def build_tasks(repo_root: Path) -> list[ScanTask]:
    ctx = ScanContext(repo_root)
    tasks: list[ScanTask] = []

    tasks.extend(
        [
            _path_exists_task(ctx, "001", "repo", "README.md exists", "README.md"),
            _path_exists_task(ctx, "002", "repo", "LICENSE exists", "LICENSE"),
            _path_exists_task(ctx, "003", "repo", "pyproject.toml exists", "pyproject.toml"),
            _path_exists_task(ctx, "004", "repo", "vgl/version.py exists", "vgl/version.py"),
            _path_exists_task(ctx, "005", "repo", "docs/core-concepts.md exists", "docs/core-concepts.md"),
            _path_exists_task(
                ctx,
                "006",
                "repo",
                "docs/public-surface-contract.md exists",
                "docs/public-surface-contract.md",
            ),
            _path_exists_task(ctx, "007", "repo", "docs/releasing.md exists", "docs/releasing.md"),
            _path_exists_task(ctx, "008", "repo", "CI workflow exists", ".github/workflows/ci.yml"),
            _path_exists_task(ctx, "009", "repo", "publish workflow exists", ".github/workflows/publish.yml"),
            _path_exists_task(ctx, "010", "repo", "release smoke script exists", "scripts/release_smoke.py"),
        ]
    )

    tasks.extend(
        [
            _pyproject_equals_task(ctx, "011", "package", "project name is sky-vgl", ("project", "name"), "sky-vgl"),
            _pyproject_equals_task(
                ctx,
                "012",
                "package",
                "requires-python is >=3.10",
                ("project", "requires-python"),
                ">=3.10",
            ),
            _pyproject_contains_task(
                ctx,
                "013",
                "package",
                "torch runtime dependency exists",
                ("project", "dependencies"),
                "torch>=2.4",
            ),
            _pyproject_contains_task(
                ctx,
                "014",
                "package",
                "typing_extensions runtime dependency exists",
                ("project", "dependencies"),
                "typing_extensions>=4.12",
            ),
            _pyproject_equals_task(
                ctx,
                "015",
                "package",
                "Homepage URL is set",
                ("project", "urls", "Homepage"),
                "https://github.com/skygazer42/sky-vgl",
            ),
            _pyproject_equals_task(
                ctx,
                "016",
                "package",
                "Repository URL is set",
                ("project", "urls", "Repository"),
                "https://github.com/skygazer42/sky-vgl",
            ),
            _pyproject_equals_task(
                ctx,
                "017",
                "package",
                "Documentation URL is set",
                ("project", "urls", "Documentation"),
                "https://skygazer42.github.io/sky-vgl",
            ),
            _pyproject_equals_task(
                ctx,
                "018",
                "package",
                "Issues URL is set",
                ("project", "urls", "Issues"),
                "https://github.com/skygazer42/sky-vgl/issues",
            ),
            _pyproject_equals_task(
                ctx,
                "019",
                "package",
                "Changelog URL is set",
                ("project", "urls", "Changelog"),
                "https://github.com/skygazer42/sky-vgl/releases",
            ),
            _pyproject_equals_task(
                ctx,
                "020",
                "package",
                "build backend is hatchling.build",
                ("build-system", "build-backend"),
                "hatchling.build",
            ),
            _pyproject_equals_task(
                ctx,
                "021",
                "package",
                "version path is vgl/version.py",
                ("tool", "hatch", "version", "path"),
                "vgl/version.py",
            ),
            _pyproject_equals_task(
                ctx,
                "022",
                "package",
                "wheel package target is vgl",
                ("tool", "hatch", "build", "targets", "wheel", "packages"),
                ["vgl"],
            ),
            _pyproject_contains_task(
                ctx,
                "023",
                "package",
                "sdist excludes docs/plans",
                ("tool", "hatch", "build", "targets", "sdist", "exclude"),
                "/docs/plans",
            ),
            _pyproject_contains_task(
                ctx,
                "024",
                "package",
                "sdist includes docs/releasing.md",
                ("tool", "hatch", "build", "targets", "sdist", "include"),
                "/docs/releasing.md",
            ),
            _pyproject_contains_task(
                ctx,
                "025",
                "package",
                "sdist includes scripts/release_smoke.py",
                ("tool", "hatch", "build", "targets", "sdist", "include"),
                "/scripts/release_smoke.py",
            ),
        ]
    )

    tasks.extend(
        [
            _pyproject_mapping_has_key_task(
                ctx,
                "026",
                "extras",
                "dev extra exists",
                ("project", "optional-dependencies"),
                "dev",
            ),
            _pyproject_mapping_has_key_task(
                ctx,
                "027",
                "extras",
                "scipy extra exists",
                ("project", "optional-dependencies"),
                "scipy",
            ),
            _pyproject_mapping_has_key_task(
                ctx,
                "028",
                "extras",
                "networkx extra exists",
                ("project", "optional-dependencies"),
                "networkx",
            ),
            _pyproject_mapping_has_key_task(
                ctx,
                "029",
                "extras",
                "tensorboard extra exists",
                ("project", "optional-dependencies"),
                "tensorboard",
            ),
            _pyproject_mapping_has_key_task(
                ctx,
                "030",
                "extras",
                "dgl extra exists",
                ("project", "optional-dependencies"),
                "dgl",
            ),
            _pyproject_mapping_has_key_task(
                ctx,
                "031",
                "extras",
                "pyg extra exists",
                ("project", "optional-dependencies"),
                "pyg",
            ),
            _pyproject_mapping_has_key_task(
                ctx,
                "032",
                "extras",
                "full extra exists",
                ("project", "optional-dependencies"),
                "full",
            ),
            _pyproject_contains_task(
                ctx,
                "033",
                "extras",
                "dev extra includes pytest",
                ("project", "optional-dependencies", "dev"),
                "pytest>=8.3",
            ),
            _pyproject_contains_task(
                ctx,
                "034",
                "extras",
                "dev extra includes ruff",
                ("project", "optional-dependencies", "dev"),
                "ruff>=0.6",
            ),
            _pyproject_contains_task(
                ctx,
                "035",
                "extras",
                "dev extra includes mypy",
                ("project", "optional-dependencies", "dev"),
                "mypy>=1.11",
            ),
        ]
    )

    asset_base = "https://raw.githubusercontent.com/skygazer42/sky-vgl/main/assets/"
    tasks.extend(
        [
            _text_contains_task(ctx, "036", "readme", "README documents pip install sky-vgl", "README.md", "pip install sky-vgl"),
            _text_contains_task(ctx, "037", "readme", "README documents sky-vgl[full]", "README.md", 'pip install "sky-vgl[full]"'),
            _text_contains_task(
                ctx,
                "038",
                "readme",
                "README documents sky-vgl[networkx]",
                "README.md",
                'pip install "sky-vgl[networkx]"',
            ),
            _text_contains_task(ctx, "039", "readme", "README documents sky-vgl[pyg]", "README.md", 'pip install "sky-vgl[pyg]"'),
            _text_contains_task(ctx, "040", "readme", "README documents sky-vgl[dgl]", "README.md", 'pip install "sky-vgl[dgl]"'),
            _text_contains_task(
                ctx,
                "041",
                "readme",
                "README documents the GitHub clone URL",
                "README.md",
                "git clone https://github.com/skygazer42/sky-vgl.git",
            ),
            _text_contains_task(ctx, "042", "readme", "README documents cd sky-vgl", "README.md", "cd sky-vgl"),
            _text_contains_task(ctx, "043", "readme", "README uses raw logo URL", "README.md", f'{asset_base}logo.svg'),
            _text_contains_task(
                ctx,
                "044",
                "readme",
                "README uses raw graph-types URL",
                "README.md",
                f'{asset_base}graph-types.svg',
            ),
            _text_contains_task(
                ctx,
                "045",
                "readme",
                "README uses raw architecture URL",
                "README.md",
                f'{asset_base}architecture.svg',
            ),
        ]
    )

    tasks.extend(
        [
            _text_contains_task(
                ctx,
                "046",
                "docs",
                "public surface contract documents pip install sky-vgl",
                "docs/public-surface-contract.md",
                "pip install sky-vgl",
            ),
            _text_contains_task(
                ctx,
                "047",
                "docs",
                "public surface contract documents sky-vgl[full]",
                "docs/public-surface-contract.md",
                'pip install "sky-vgl[full]"',
            ),
            _text_contains_task(
                ctx,
                "048",
                "docs",
                "public surface contract explains imports remain under vgl",
                "docs/public-surface-contract.md",
                "imports remain under `vgl`",
            ),
            _text_contains_task(
                ctx,
                "049",
                "docs",
                "releasing doc includes python -m build",
                "docs/releasing.md",
                "python -m build",
            ),
            _text_contains_task(
                ctx,
                "050",
                "docs",
                "releasing doc includes python -m twine check",
                "docs/releasing.md",
                "python -m twine check",
            ),
            _text_contains_task(
                ctx,
                "051",
                "docs",
                "releasing doc includes release_smoke command",
                "docs/releasing.md",
                "python scripts/release_smoke.py --artifact-dir dist --kind all",
            ),
            _text_contains_task(
                ctx,
                "052",
                "docs",
                "releasing doc mentions TestPyPI verification",
                "docs/releasing.md",
                "TestPyPI",
            ),
            _text_contains_task(
                ctx,
                "053",
                "docs",
                "releasing doc mentions pending publisher handling",
                "docs/releasing.md",
                "pending publisher",
            ),
            _text_contains_task(
                ctx,
                "054",
                "docs",
                "releasing doc mentions Manage Project -> Publishing",
                "docs/releasing.md",
                "Manage Project -> Publishing",
            ),
            _text_contains_task(
                ctx,
                "055",
                "docs",
                "releasing doc mentions PYPI_API_TOKEN",
                "docs/releasing.md",
                "PYPI_API_TOKEN",
            ),
            _text_contains_task(
                ctx,
                "055a",
                "docs",
                "releasing doc mentions package-check release gate",
                "docs/releasing.md",
                "package-check",
            ),
            _text_contains_task(
                ctx,
                "055b",
                "docs",
                "releasing doc mentions publish build release gate",
                "docs/releasing.md",
                "publish `build` job",
            ),
            _text_contains_task(
                ctx,
                "055c",
                "docs",
                "releasing doc mentions release gate interop extras install",
                "docs/releasing.md",
                "python scripts/install_release_extras.py --artifact-dir dist --extras pyg dgl",
            ),
            _text_contains_task(
                ctx,
                "055d",
                "docs",
                "releasing doc mentions all-backend release artifact smoke gate",
                "docs/releasing.md",
                "python scripts/release_smoke.py --artifact-dir dist --kind all --interop-backend all",
            ),
        ]
    )

    tasks.extend(
        [
            _text_contains_task(ctx, "056", "ci", "CI triggers on pushes to main", ".github/workflows/ci.yml", "branches:\n      - main"),
            _text_contains_task(ctx, "057", "ci", "CI triggers on pull requests", ".github/workflows/ci.yml", "pull_request:"),
            _text_contains_task(ctx, "058", "ci", "CI defines the test job", ".github/workflows/ci.yml", "  test:\n"),
            _text_contains_task(ctx, "059", "ci", "CI tests Python 3.10", ".github/workflows/ci.yml", '"3.10"'),
            _text_contains_task(ctx, "060", "ci", "CI tests Python 3.11", ".github/workflows/ci.yml", '"3.11"'),
            _text_contains_task(ctx, "061", "ci", "CI runs pytest", ".github/workflows/ci.yml", "python -m pytest -q"),
            _text_contains_task(ctx, "062", "ci", "CI defines package-check", ".github/workflows/ci.yml", "  package-check:\n"),
            _text_contains_task(ctx, "063", "ci", "CI runs python -m build", ".github/workflows/ci.yml", "python -m build"),
            _text_contains_task(ctx, "064", "ci", "CI runs python -m twine check", ".github/workflows/ci.yml", "python -m twine check"),
            _text_contains_task(ctx, "065", "ci", "CI runs full_scan.py", ".github/workflows/ci.yml", "python scripts/full_scan.py"),
            _workflow_step_contains_task(
                ctx,
                "065a",
                "ci",
                "CI package-check job installs release interop extras from built wheel",
                ".github/workflows/ci.yml",
                "package-check",
                "Install release interop extras",
                "python scripts/install_release_extras.py --artifact-dir dist --extras pyg dgl",
            ),
            _workflow_step_contains_task(
                ctx,
                "065b",
                "ci",
                "CI package-check job runs all-backend release artifact smoke",
                ".github/workflows/ci.yml",
                "package-check",
                "Smoke-test built distributions with all interop backends",
                "python scripts/release_smoke.py --artifact-dir dist --kind all --interop-backend all",
            ),
            _workflow_step_lacks_task(
                ctx,
                "065c",
                "ci",
                "CI package-check install step avoids editable checkout extras",
                ".github/workflows/ci.yml",
                "package-check",
                "Install release interop extras",
                'python -m pip install -e ".[pyg,dgl]"',
            ),
        ]
    )

    tasks.extend(
        [
            _text_contains_task(ctx, "066", "publish", "publish triggers on v* tags", ".github/workflows/publish.yml", '- "v*"'),
            _text_contains_task(
                ctx,
                "067",
                "publish",
                "publish triggers on published releases",
                ".github/workflows/publish.yml",
                "types: [published]",
            ),
            _text_contains_task(
                ctx,
                "068",
                "publish",
                "publish supports workflow_dispatch",
                ".github/workflows/publish.yml",
                "workflow_dispatch:",
            ),
            _text_contains_task(
                ctx,
                "069",
                "publish",
                "publish defines auth probe job",
                ".github/workflows/publish.yml",
                "probe-publish-auth:",
            ),
            _text_contains_task(ctx, "070", "publish", "publish defines build job", ".github/workflows/publish.yml", "\n  build:\n"),
            _text_contains_task(
                ctx,
                "071",
                "publish",
                "publish has TestPyPI token path",
                ".github/workflows/publish.yml",
                "Publish to TestPyPI with API token",
            ),
            _text_contains_task(
                ctx,
                "072",
                "publish",
                "publish has TestPyPI trusted path",
                ".github/workflows/publish.yml",
                "Publish to TestPyPI with Trusted Publishing",
            ),
            _text_contains_task(
                ctx,
                "073",
                "publish",
                "publish has PyPI token path",
                ".github/workflows/publish.yml",
                "Publish to PyPI with API token",
            ),
            _text_contains_task(
                ctx,
                "074",
                "publish",
                "publish has PyPI trusted path",
                ".github/workflows/publish.yml",
                "Publish to PyPI with Trusted Publishing",
            ),
            _text_contains_task(
                ctx,
                "075",
                "publish",
                "publish uploads built distributions as artifacts",
                ".github/workflows/publish.yml",
                "actions/upload-artifact@v4",
            ),
            _workflow_step_contains_task(
                ctx,
                "075a",
                "publish",
                "publish build job installs release interop extras from built wheel",
                ".github/workflows/publish.yml",
                "build",
                "Install release interop extras",
                "python scripts/install_release_extras.py --artifact-dir dist --extras pyg dgl",
            ),
            _workflow_step_contains_task(
                ctx,
                "075b",
                "publish",
                "publish build runs all-backend release artifact smoke",
                ".github/workflows/publish.yml",
                "build",
                "Smoke-test built distributions with all interop backends",
                "python scripts/release_smoke.py --artifact-dir dist --kind all --interop-backend all",
            ),
            _workflow_step_lacks_task(
                ctx,
                "075c",
                "publish",
                "publish build install step avoids editable checkout extras",
                ".github/workflows/publish.yml",
                "build",
                "Install release interop extras",
                'python -m pip install -e ".[pyg,dgl]"',
            ),
        ]
    )

    tasks.extend(
        [
            _path_exists_task(ctx, "076", "assets", "assets/logo.svg exists", "assets/logo.svg"),
            _path_exists_task(ctx, "077", "assets", "assets/graph-types.svg exists", "assets/graph-types.svg"),
            _path_exists_task(ctx, "078", "assets", "assets/architecture.svg exists", "assets/architecture.svg"),
            _path_exists_task(ctx, "079", "assets", "assets/pipeline.svg exists", "assets/pipeline.svg"),
            _path_exists_task(ctx, "080", "assets", "assets/conv-layers.svg exists", "assets/conv-layers.svg"),
            _path_exists_task(
                ctx,
                "081",
                "examples",
                "examples/homo/node_classification.py exists",
                "examples/homo/node_classification.py",
            ),
            _path_exists_task(
                ctx,
                "082",
                "examples",
                "examples/homo/link_prediction.py exists",
                "examples/homo/link_prediction.py",
            ),
            _path_exists_task(
                ctx,
                "083",
                "examples",
                "examples/hetero/node_classification.py exists",
                "examples/hetero/node_classification.py",
            ),
            _path_exists_task(
                ctx,
                "084",
                "examples",
                "examples/hetero/link_prediction.py exists",
                "examples/hetero/link_prediction.py",
            ),
            _path_exists_task(
                ctx,
                "085",
                "examples",
                "examples/temporal/event_prediction.py exists",
                "examples/temporal/event_prediction.py",
            ),
        ]
    )

    tasks.extend(
        [
            _path_exists_task(ctx, "086", "layout", "vgl/graph package exists", "vgl/graph", directory=True),
            _path_exists_task(ctx, "087", "layout", "vgl/dataloading package exists", "vgl/dataloading", directory=True),
            _path_exists_task(ctx, "088", "layout", "vgl/storage package exists", "vgl/storage", directory=True),
            _path_exists_task(ctx, "089", "layout", "vgl/sparse package exists", "vgl/sparse", directory=True),
            _path_exists_task(ctx, "090", "layout", "vgl/ops package exists", "vgl/ops", directory=True),
            _path_exists_task(
                ctx,
                "091",
                "layout",
                "vgl/distributed package exists",
                "vgl/distributed",
                directory=True,
            ),
            _path_exists_task(ctx, "092", "layout", "vgl/nn package exists", "vgl/nn", directory=True),
            _path_exists_task(ctx, "093", "layout", "vgl/tasks package exists", "vgl/tasks", directory=True),
            _path_exists_task(ctx, "094", "layout", "vgl/engine package exists", "vgl/engine", directory=True),
            _path_exists_task(ctx, "095", "layout", "vgl/compat package exists", "vgl/compat", directory=True),
        ]
    )

    tasks.extend(
        [
            _path_exists_task(
                ctx,
                "096",
                "tests",
                "tests/test_release_packaging.py exists",
                "tests/test_release_packaging.py",
            ),
            _path_exists_task(
                ctx,
                "097",
                "tests",
                "tests/test_runtime_compat.py exists",
                "tests/test_runtime_compat.py",
            ),
            _path_exists_task(
                ctx,
                "098",
                "tests",
                "tests/test_package_layout.py exists",
                "tests/test_package_layout.py",
            ),
            _path_exists_task(
                ctx,
                "099",
                "tests",
                "tests/core/test_graph_ops_api.py exists",
                "tests/core/test_graph_ops_api.py",
            ),
            _path_exists_task(
                ctx,
                "100",
                "tests",
                "tests/data/test_loader.py exists",
                "tests/data/test_loader.py",
            ),
            _path_exists_task(
                ctx,
                "101",
                "repo",
                "interop smoke script exists",
                "scripts/interop_smoke.py",
            ),
            _path_exists_task(
                ctx,
                "102",
                "repo",
                "interop smoke workflow exists",
                ".github/workflows/interop-smoke.yml",
            ),
            _text_contains_task(
                ctx,
                "103",
                "docs",
                "releasing doc includes interop smoke command",
                "docs/releasing.md",
                "python scripts/interop_smoke.py --backend all",
            ),
            _text_contains_task(
                ctx,
                "104",
                "docs",
                "releasing doc includes release artifact interop smoke command",
                "docs/releasing.md",
                "python scripts/release_smoke.py --artifact-dir dist --kind wheel --interop-backend dgl",
            ),
            _text_contains_task(
                ctx,
                "105",
                "tooling",
                "Makefile includes release artifact interop smoke target",
                "Makefile",
                "scripts/release_smoke.py --artifact-dir dist --kind wheel --interop-backend=$(RELEASE_INTEROP_BACKEND)",
            ),
            _text_contains_task(
                ctx,
                "106",
                "workflow",
                "interop workflow builds artifacts",
                ".github/workflows/interop-smoke.yml",
                "python -m build",
            ),
            _text_contains_task(
                ctx,
                "107",
                "workflow",
                "interop workflow runs all-backend checkout smoke",
                ".github/workflows/interop-smoke.yml",
                "python scripts/interop_smoke.py --backend all",
            ),
            _text_contains_task(
                ctx,
                "108",
                "workflow",
                "interop workflow runs all-backend release artifact smoke",
                ".github/workflows/interop-smoke.yml",
                "python scripts/release_smoke.py --artifact-dir dist --kind wheel --interop-backend all",
            ),
        ]
    )

    if len(tasks) < 100:
        raise RuntimeError(f"expected at least 100 scan tasks, found {len(tasks)}")
    return tasks


def _run_task(task: ScanTask) -> ScanResult:
    try:
        passed, details = task.check()
    except Exception as exc:  # pragma: no cover - defensive path
        return ScanResult(task, False, f"{type(exc).__name__}: {exc}")
    return ScanResult(task, bool(passed), details)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the repository-wide full scan.")
    parser.add_argument("--list", action="store_true", help="List scan tasks without executing them.")
    parser.add_argument(
        "--repo-root",
        default=Path(__file__).resolve().parents[1],
        type=Path,
        help="Repository root to scan. Defaults to the current checkout root.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    tasks = build_tasks(args.repo_root)
    if args.list:
        for task in tasks:
            print(f"SCAN {task.id} [{task.category}] {task.description}")
        print(f"SUMMARY listed {len(tasks)} tasks")
        return 0

    results = [_run_task(task) for task in tasks]
    passed = 0
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"{status} {result.task.id} [{result.task.category}] {result.task.description} :: {result.details}")
        if result.passed:
            passed += 1
    print(f"SUMMARY {passed}/{len(results)} passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
