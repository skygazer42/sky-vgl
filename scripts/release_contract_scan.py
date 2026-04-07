#!/usr/bin/env python3

from __future__ import annotations

import argparse
import email
import importlib
import re
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

try:
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib  # type: ignore[no-redef]


try:
    repo_script_imports = importlib.import_module("scripts.repo_script_imports")
except ModuleNotFoundError:
    repo_script_imports = importlib.import_module("repo_script_imports")

load_repo_module = repo_script_imports.load_repo_module
resolve_repo_relative_path = repo_script_imports.resolve_repo_relative_path


_contracts = load_repo_module("scripts.contracts")
OPTIONAL_EXTRAS = _contracts.OPTIONAL_EXTRAS
PROJECT_NAME = _contracts.PROJECT_NAME
PROJECT_URLS = _contracts.PROJECT_URLS
RELEASE_INTEROP_EXTRA_REQUIREMENTS = _contracts.RELEASE_INTEROP_EXTRA_REQUIREMENTS
REQUIRES_PYTHON = _contracts.REQUIRES_PYTHON
SDIST_EXCLUDED_SUBSTRINGS = _contracts.SDIST_EXCLUDED_SUBSTRINGS
SDIST_REQUIRED_SUFFIXES = _contracts.SDIST_REQUIRED_SUFFIXES
WHEEL_EXCLUDED_SUBSTRINGS = _contracts.WHEEL_EXCLUDED_SUBSTRINGS
WHEEL_REQUIRED_FILES = _contracts.WHEEL_REQUIRED_FILES
read_wheel_metadata = load_repo_module("scripts.release_artifact_metadata").read_wheel_metadata


CheckFn = Callable[[], tuple[bool, str]]


@dataclass(frozen=True)
class ScanTask:
    id: str
    category: str
    description: str
    check: CheckFn


class ArtifactContext:
    def __init__(self, repo_root: Path, artifact_dir: Path):
        self.repo_root = repo_root.resolve()
        self.artifact_dir = artifact_dir.resolve()
        self._pyproject: dict | None = None
        self._version: str | None = None
        self._wheel_path: tuple[Path | None, str] | None = None
        self._sdist_path: tuple[Path | None, str] | None = None
        self._wheel_metadata: tuple[email.message.Message | None, str] | None = None
        self._wheel_names: tuple[list[str] | None, str] | None = None
        self._sdist_names: tuple[list[str] | None, str] | None = None

    def pyproject_value(self, *keys: str) -> object:
        payload: object = self._load_pyproject()
        for key in keys:
            if not isinstance(payload, dict):
                raise KeyError(" -> ".join(keys))
            payload = payload[key]
        return payload

    def repo_version(self) -> str:
        if self._version is None:
            text = (self.repo_root / "vgl" / "version.py").read_text(encoding="utf-8")
            match = re.search(r"""__version__\s*=\s*["']([^"']+)["']""", text)
            if match is None:
                raise RuntimeError("unable to parse __version__ from vgl/version.py")
            self._version = match.group(1)
        return self._version

    def wheel_path(self) -> tuple[Path | None, str]:
        if self._wheel_path is None:
            self._wheel_path = self._single_artifact("*.whl")
        return self._wheel_path

    def sdist_path(self) -> tuple[Path | None, str]:
        if self._sdist_path is None:
            self._sdist_path = self._single_artifact("*.tar.gz")
        return self._sdist_path

    def wheel_metadata(self) -> tuple[email.message.Message | None, str]:
        if self._wheel_metadata is None:
            wheel_path, detail = self.wheel_path()
            if wheel_path is None:
                self._wheel_metadata = (None, detail)
            else:
                self._wheel_metadata = read_wheel_metadata(wheel_path)
        return self._wheel_metadata

    def wheel_names(self) -> tuple[list[str] | None, str]:
        if self._wheel_names is None:
            wheel_path, detail = self.wheel_path()
            if wheel_path is None:
                self._wheel_names = (None, detail)
            else:
                with zipfile.ZipFile(wheel_path) as archive:
                    names = archive.namelist()
                self._wheel_names = (names, f"{len(names)} wheel entries")
        return self._wheel_names

    def sdist_names(self) -> tuple[list[str] | None, str]:
        if self._sdist_names is None:
            sdist_path, detail = self.sdist_path()
            if sdist_path is None:
                self._sdist_names = (None, detail)
            else:
                with tarfile.open(sdist_path) as archive:
                    names = archive.getnames()
                self._sdist_names = (names, f"{len(names)} sdist entries")
        return self._sdist_names

    def _load_pyproject(self) -> dict:
        if self._pyproject is None:
            with (self.repo_root / "pyproject.toml").open("rb") as handle:
                self._pyproject = tomllib.load(handle)
        return self._pyproject

    def _single_artifact(self, pattern: str) -> tuple[Path | None, str]:
        matches = sorted(self.artifact_dir.glob(pattern))
        if len(matches) != 1:
            return None, f"{pattern} count == {len(matches)} in {self.artifact_dir}"
        return matches[0], matches[0].name


def _artifact_exists_task(
    ctx: ArtifactContext,
    task_id: str,
    description: str,
    resolver: Callable[[], tuple[Path | None, str]],
) -> ScanTask:
    def check() -> tuple[bool, str]:
        path, detail = resolver()
        return path is not None, detail

    return ScanTask(task_id, "artifact", description, check)


def _metadata_header_equals_task(
    ctx: ArtifactContext,
    task_id: str,
    description: str,
    header: str,
    expected: str,
) -> ScanTask:
    def check() -> tuple[bool, str]:
        metadata, detail = ctx.wheel_metadata()
        if metadata is None:
            return False, detail
        value = metadata.get(header)
        return value == expected, f"{header} == {expected!r}"

    return ScanTask(task_id, "metadata", description, check)


def _project_url_task(ctx: ArtifactContext, task_id: str, label: str, url: str) -> ScanTask:
    def check() -> tuple[bool, str]:
        metadata, detail = ctx.wheel_metadata()
        if metadata is None:
            return False, detail
        project_urls = metadata.get_all("Project-URL", [])
        expected = f"{label}, {url}"
        return expected in project_urls, expected

    return ScanTask(task_id, "metadata", f"wheel exposes {label} project URL", check)


def _provides_extra_task(ctx: ArtifactContext, task_id: str, extra: str) -> ScanTask:
    def check() -> tuple[bool, str]:
        metadata, detail = ctx.wheel_metadata()
        if metadata is None:
            return False, detail
        extras = metadata.get_all("Provides-Extra", [])
        return extra in extras, f"{extra!r} in Provides-Extra"

    return ScanTask(task_id, "metadata", f"wheel provides extra {extra}", check)


def _requires_dist_task(ctx: ArtifactContext, task_id: str, extra: str, requirement: str) -> ScanTask:
    def check() -> tuple[bool, str]:
        metadata, detail = ctx.wheel_metadata()
        if metadata is None:
            return False, detail
        requires_dist = metadata.get_all("Requires-Dist", [])
        return requirement in requires_dist, f"{requirement!r} in Requires-Dist"

    return ScanTask(
        task_id,
        "metadata",
        f"wheel metadata exposes {extra} extra requirement line {requirement}",
        check,
    )


def _wheel_contains_task(ctx: ArtifactContext, task_id: str, relative_path: str) -> ScanTask:
    def check() -> tuple[bool, str]:
        names, detail = ctx.wheel_names()
        if names is None:
            return False, detail
        return relative_path in names, relative_path

    return ScanTask(task_id, "wheel", f"wheel contains {relative_path}", check)


def _wheel_excludes_task(ctx: ArtifactContext, task_id: str, substring: str) -> ScanTask:
    def check() -> tuple[bool, str]:
        names, detail = ctx.wheel_names()
        if names is None:
            return False, detail
        return not any(substring in name for name in names), f"{substring!r} not present in wheel"

    return ScanTask(task_id, "wheel", f"wheel excludes {substring}", check)


def _sdist_contains_task(ctx: ArtifactContext, task_id: str, suffix: str) -> ScanTask:
    def check() -> tuple[bool, str]:
        names, detail = ctx.sdist_names()
        if names is None:
            return False, detail
        return any(name.endswith(suffix) for name in names), suffix

    return ScanTask(task_id, "sdist", f"sdist contains {suffix}", check)


def _sdist_excludes_task(ctx: ArtifactContext, task_id: str, substring: str) -> ScanTask:
    def check() -> tuple[bool, str]:
        names, detail = ctx.sdist_names()
        if names is None:
            return False, detail
        return not any(substring in name for name in names), f"{substring!r} not present in sdist"

    return ScanTask(task_id, "sdist", f"sdist excludes {substring}", check)


def build_tasks(repo_root: Path, artifact_dir: Path) -> list[ScanTask]:
    ctx = ArtifactContext(repo_root, artifact_dir)
    project_urls = ctx.pyproject_value("project", "urls")
    if project_urls != PROJECT_URLS:
        raise RuntimeError("pyproject project.urls must match scripts/contracts.py")
    if str(ctx.pyproject_value("project", "name")) != PROJECT_NAME:
        raise RuntimeError("pyproject project.name must match scripts/contracts.py")
    if str(ctx.pyproject_value("project", "requires-python")) != REQUIRES_PYTHON:
        raise RuntimeError("pyproject project.requires-python must match scripts/contracts.py")

    tasks = [
        _artifact_exists_task(ctx, "001", "built wheel exists", ctx.wheel_path),
        _artifact_exists_task(ctx, "002", "built sdist exists", ctx.sdist_path),
        _metadata_header_equals_task(ctx, "003", "wheel name matches project", "Name", PROJECT_NAME),
        _metadata_header_equals_task(ctx, "004", "wheel version matches repo version", "Version", ctx.repo_version()),
        _metadata_header_equals_task(
            ctx,
            "005",
            "wheel Requires-Python matches pyproject",
            "Requires-Python",
            REQUIRES_PYTHON,
        ),
        *[
            _project_url_task(ctx, f"{index:03d}", label, PROJECT_URLS[label])
            for index, label in enumerate(PROJECT_URLS, start=6)
        ],
        *[
            _provides_extra_task(ctx, f"{index:03d}", extra)
            for index, extra in enumerate(OPTIONAL_EXTRAS, start=11)
        ],
        *[
            _requires_dist_task(ctx, f"{index:03d}", extra, requirement)
            for index, (extra, requirement) in enumerate(RELEASE_INTEROP_EXTRA_REQUIREMENTS.items(), start=18)
        ],
        *[
            _wheel_contains_task(
                ctx,
                f"{index:03d}",
                relative_path,
            )
            for index, relative_path in enumerate(
                WHEEL_REQUIRED_FILES,
                start=18 + len(RELEASE_INTEROP_EXTRA_REQUIREMENTS),
            )
        ],
        *[
            _wheel_excludes_task(ctx, f"{index:03d}", substring)
            for index, substring in enumerate(
                WHEEL_EXCLUDED_SUBSTRINGS,
                start=18 + len(RELEASE_INTEROP_EXTRA_REQUIREMENTS) + len(WHEEL_REQUIRED_FILES),
            )
        ],
        *[
            _sdist_contains_task(ctx, f"{index:03d}", suffix)
            for index, suffix in enumerate(
                SDIST_REQUIRED_SUFFIXES,
                start=18
                + len(RELEASE_INTEROP_EXTRA_REQUIREMENTS)
                + len(WHEEL_REQUIRED_FILES)
                + len(WHEEL_EXCLUDED_SUBSTRINGS),
            )
        ],
        *[
            _sdist_excludes_task(ctx, f"{index:03d}", substring)
            for index, substring in enumerate(
                SDIST_EXCLUDED_SUBSTRINGS,
                start=18
                + len(RELEASE_INTEROP_EXTRA_REQUIREMENTS)
                + len(WHEEL_REQUIRED_FILES)
                + len(WHEEL_EXCLUDED_SUBSTRINGS)
                + len(SDIST_REQUIRED_SUFFIXES),
            )
        ],
    ]

    expected_task_count = 5 + len(PROJECT_URLS) + len(OPTIONAL_EXTRAS) + len(WHEEL_REQUIRED_FILES)
    expected_task_count += len(RELEASE_INTEROP_EXTRA_REQUIREMENTS)
    expected_task_count += len(WHEEL_EXCLUDED_SUBSTRINGS) + len(SDIST_REQUIRED_SUFFIXES) + len(SDIST_EXCLUDED_SUBSTRINGS)
    if len(tasks) != expected_task_count:
        raise RuntimeError(f"expected {expected_task_count} scan tasks, found {len(tasks)}")
    return tasks


def run_tasks(tasks: list[ScanTask]) -> int:
    passed = 0
    for task in tasks:
        ok, detail = task.check()
        status = "PASS" if ok else "FAIL"
        print(f"{status} {task.id} [{task.category}] {task.description} :: {detail}")
        if ok:
            passed += 1
    print(f"SUMMARY {passed}/{len(tasks)} passed")
    return 0 if passed == len(tasks) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan built release artifacts for packaging contracts.")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="Directory containing built wheel and sdist artifacts. Defaults to <repo-root>/dist.",
    )
    parser.add_argument("--list", action="store_true", help="List scan tasks without executing them.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root to scan. Defaults to this checkout root.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    artifact_dir = resolve_repo_relative_path(args.artifact_dir or Path("dist"), repo_root=repo_root)
    tasks = build_tasks(repo_root, artifact_dir)

    if args.list:
        for task in tasks:
            print(f"SCAN {task.id} [{task.category}] {task.description}")
        return 0
    return run_tasks(tasks)


if __name__ == "__main__":
    raise SystemExit(main())
