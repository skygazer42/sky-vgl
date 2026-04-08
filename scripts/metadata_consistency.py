#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import repo_script_imports


load_repo_module = repo_script_imports.load_repo_module
load_toml_file = repo_script_imports.load_toml_file
_contracts = load_repo_module("scripts.contracts")
DOCS_INDEX_VERSION_BADGE = _contracts.DOCS_INDEX_VERSION_BADGE
PROJECT_URLS = _contracts.PROJECT_URLS
README_VERSION_BADGE = _contracts.README_VERSION_BADGE
RELEASE_VERSION = _contracts.RELEASE_VERSION


def _check(condition: bool, message: str) -> tuple[bool, str]:
    return condition, message


def _pyproject_urls(repo_root: Path) -> tuple[bool, str]:
    pyproject = load_toml_file(repo_root / "pyproject.toml")
    return _check(pyproject["project"]["urls"] == PROJECT_URLS, "pyproject project.urls matches contracts")


def _readme_version_badge(repo_root: Path) -> tuple[bool, str]:
    readme = (repo_root / "README.md").read_text(encoding="utf-8")
    if README_VERSION_BADGE not in readme:
        return False, "README uses the shared dynamic PyPI version badge"
    return _check(
        f"version-{RELEASE_VERSION}" not in readme,
        "README avoids stale hard-coded version badge text",
    )


def _docs_index_version_badge(repo_root: Path) -> tuple[bool, str]:
    index = (repo_root / "docs" / "index.md").read_text(encoding="utf-8")
    if DOCS_INDEX_VERSION_BADGE not in index:
        return False, "docs index uses the shared dynamic PyPI version badge"
    return _check(
        f"version-{RELEASE_VERSION}" not in index,
        "docs index avoids stale hard-coded version badge text",
    )


def _installation_example(repo_root: Path) -> tuple[bool, str]:
    installation = (repo_root / "docs" / "getting-started" / "installation.md").read_text(encoding="utf-8")
    return _check(
        f"# {RELEASE_VERSION}" not in installation and "installed release version" in installation,
        "installation guide avoids hard-coded __version__ output",
    )


def _changelog(repo_root: Path) -> tuple[bool, str]:
    changelog = (repo_root / "docs" / "changelog.md").read_text(encoding="utf-8")
    has_current_section = f"## v{RELEASE_VERSION}" in changelog
    has_structured_headings = all(label in changelog for label in ("### Added", "### Changed", "### Fixed"))
    return _check(has_current_section and has_structured_headings, "changelog contains versioned structured sections")


def _git_ref_matches_version(git_ref: str | None) -> tuple[bool, str]:
    if not git_ref or not git_ref.startswith("refs/tags/v"):
        return True, "no release tag ref provided"
    tag_version = git_ref.removeprefix("refs/tags/v")
    return _check(tag_version == RELEASE_VERSION, f"tag version {tag_version!r} matches repo version")


def run(repo_root: Path, *, git_ref: str | None) -> int:
    checks = [
        _pyproject_urls(repo_root),
        _readme_version_badge(repo_root),
        _docs_index_version_badge(repo_root),
        _installation_example(repo_root),
        _changelog(repo_root),
        _git_ref_matches_version(git_ref),
    ]
    passed = 0
    for index, (ok, detail) in enumerate(checks, start=1):
        status = "PASS" if ok else "FAIL"
        print(f"{status} {index:03d} [metadata] {detail}")
        if ok:
            passed += 1
    print(f"SUMMARY {passed}/{len(checks)} passed")
    return 0 if passed == len(checks) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate release metadata and version display consistency.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root to scan. Defaults to this checkout root.",
    )
    parser.add_argument(
        "--git-ref",
        help="Optional Git ref such as refs/tags/v0.1.5 to verify against the repo version.",
    )
    args = parser.parse_args()
    return run(args.repo_root.resolve(), git_ref=args.git_ref)


if __name__ == "__main__":
    raise SystemExit(main())
