#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import repo_script_imports


load_repo_module = repo_script_imports.load_repo_module


_CONTRACTS = None


def _contracts_module():
    global _CONTRACTS
    if _CONTRACTS is None:
        _CONTRACTS = load_repo_module("scripts.contracts")
    return _CONTRACTS


def __getattr__(name: str):
    if name in {"DOCS_INDEX_VERSION_BADGE", "PROJECT_URLS", "README_VERSION_BADGE", "RELEASE_VERSION"}:
        return getattr(_contracts_module(), name)
    raise AttributeError(name)


def _check(condition: bool, message: str) -> tuple[bool, str]:
    return condition, message


def _pyproject_urls(repo_root: Path) -> tuple[bool, str]:
    try:
        import tomllib  # type: ignore[attr-defined]
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
        import tomli as tomllib  # type: ignore[no-redef]

    with (repo_root / "pyproject.toml").open("rb") as handle:
        pyproject = tomllib.load(handle)
    return _check(pyproject["project"]["urls"] == _contracts_module().PROJECT_URLS, "pyproject project.urls matches contracts")


def _readme_version_badge(repo_root: Path) -> tuple[bool, str]:
    contracts = _contracts_module()
    readme = (repo_root / "README.md").read_text(encoding="utf-8")
    if contracts.README_VERSION_BADGE not in readme:
        return False, "README uses the shared dynamic PyPI version badge"
    return _check(
        f"version-{contracts.RELEASE_VERSION}" not in readme,
        "README avoids stale hard-coded version badge text",
    )


def _docs_index_version_badge(repo_root: Path) -> tuple[bool, str]:
    contracts = _contracts_module()
    index = (repo_root / "docs" / "index.md").read_text(encoding="utf-8")
    if contracts.DOCS_INDEX_VERSION_BADGE not in index:
        return False, "docs index uses the shared dynamic PyPI version badge"
    return _check(
        f"version-{contracts.RELEASE_VERSION}" not in index,
        "docs index avoids stale hard-coded version badge text",
    )


def _installation_example(repo_root: Path) -> tuple[bool, str]:
    release_version = _contracts_module().RELEASE_VERSION
    installation = (repo_root / "docs" / "getting-started" / "installation.md").read_text(encoding="utf-8")
    return _check(
        f"# {release_version}" not in installation and "installed release version" in installation,
        "installation guide avoids hard-coded __version__ output",
    )


def _changelog(repo_root: Path) -> tuple[bool, str]:
    release_version = _contracts_module().RELEASE_VERSION
    changelog = (repo_root / "docs" / "changelog.md").read_text(encoding="utf-8")
    has_current_section = f"## v{release_version}" in changelog
    has_structured_headings = (
        "## Unreleased" in changelog
        and all(
            heading in changelog
            for heading in ("### API", "### Performance", "### Interop", "### Migration")
        )
    )
    return _check(has_current_section and has_structured_headings, "changelog contains versioned structured sections")


def _git_ref_matches_version(git_ref: str | None) -> tuple[bool, str]:
    if not git_ref or not git_ref.startswith("refs/tags/v"):
        return True, "no release tag ref provided"
    tag_version = git_ref.removeprefix("refs/tags/v")
    return _check(tag_version == _contracts_module().RELEASE_VERSION, f"tag version {tag_version!r} matches repo version")


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
