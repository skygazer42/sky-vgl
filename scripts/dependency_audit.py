#!/usr/bin/env python3

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import repo_script_imports


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit runtime requirements with pip-audit.")
    parser.add_argument(
        "--groups",
        nargs="+",
        metavar="GROUP",
        help="Dependency groups to audit. Defaults to runtime plus all optional groups.",
    )
    parser.add_argument(
        "--print-requirements",
        action="store_true",
        help="Print parsed runtime requirements and exit.",
    )
    parser.add_argument(
        "--repo-root",
        default=Path(__file__).resolve().parents[1],
        type=Path,
        help="Repository root containing pyproject.toml.",
    )
    return parser.parse_args()


def _dependency_groups(repo_root: Path) -> dict[str, list[str]]:
    payload = repo_script_imports.load_toml_file(repo_root / "pyproject.toml")
    project = payload["project"]
    groups = {
        "runtime": [str(requirement) for requirement in project["dependencies"]],
    }
    for name, requirements in project.get("optional-dependencies", {}).items():
        groups[str(name)] = [str(requirement) for requirement in requirements]
    return groups


def _requirements_for_groups(
    dependency_groups: dict[str, list[str]],
    selected_groups: list[str] | None,
) -> list[str]:
    group_names = list(dependency_groups)
    selected = selected_groups or group_names
    unknown = [group for group in selected if group not in dependency_groups]
    if unknown:
        raise SystemExit(f"unsupported dependency groups: {', '.join(unknown)}")

    requirements: list[str] = []
    seen = set()
    for group in selected:
        for requirement in dependency_groups[group]:
            if requirement in seen:
                continue
            seen.add(requirement)
            requirements.append(requirement)
    return requirements


def main() -> int:
    args = _parse_args()
    repo_root = args.repo_root.resolve()
    dependency_groups = _dependency_groups(repo_root)
    requirements = _requirements_for_groups(dependency_groups, args.groups)

    if args.print_requirements:
        for requirement in requirements:
            print(requirement)
        return 0

    with tempfile.TemporaryDirectory(prefix="vgl-dependency-audit-") as tmp:
        tmp_dir = Path(tmp)
        requirements_path = tmp_dir / "requirements.txt"
        cache_dir = tmp_dir / "cache"
        cache_dir.mkdir()
        requirements_path.write_text("".join(f"{item}\n" for item in requirements), encoding="utf-8")
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip_audit",
                "--progress-spinner",
                "off",
                "--cache-dir",
                str(cache_dir),
                "-r",
                str(requirements_path),
            ],
            cwd=repo_root,
        )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
