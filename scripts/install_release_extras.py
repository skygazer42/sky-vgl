#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
from pathlib import Path

from packaging.markers import default_environment
from packaging.requirements import Requirement


try:
    repo_script_imports = importlib.import_module("scripts.repo_script_imports")
except ModuleNotFoundError:
    repo_script_imports = importlib.import_module("repo_script_imports")

load_repo_module = repo_script_imports.load_repo_module
resolve_repo_relative_path = repo_script_imports.resolve_repo_relative_path


read_wheel_metadata = load_repo_module("scripts.release_artifact_metadata").read_wheel_metadata


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve optional extra requirements from built wheel metadata and "
            "optionally install them into the current environment."
        )
    )
    parser.add_argument(
        "--artifact-dir",
        required=True,
        help="Directory containing exactly one built wheel artifact.",
    )
    parser.add_argument(
        "--extras",
        nargs="+",
        required=True,
        help="Optional extras to resolve from wheel metadata.",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Print resolved requirements without invoking pip install.",
    )
    return parser.parse_args()


def _find_single_wheel(artifact_dir: Path) -> Path:
    matches = sorted(artifact_dir.glob("*.whl"))
    if len(matches) != 1:
        raise SystemExit(
            f"expected exactly one wheel artifact in {artifact_dir}, found {len(matches)}"
        )
    return matches[0]


def _wheel_metadata(wheel_path: Path):
    metadata, detail = read_wheel_metadata(wheel_path)
    if metadata is None:
        raise SystemExit(detail)
    return metadata


def _requirement_without_marker(requirement: Requirement) -> str:
    text = requirement.name
    if requirement.extras:
        text += f"[{','.join(sorted(requirement.extras))}]"
    if requirement.url is not None:
        text += f" @ {requirement.url}"
    else:
        text += str(requirement.specifier)
    return text


def _resolved_extra_requirements(wheel_path: Path, extras: list[str]) -> list[str]:
    metadata = _wheel_metadata(wheel_path)
    available_extras = {extra for extra in metadata.get_all("Provides-Extra", []) if extra}
    requirement_lines = metadata.get_all("Requires-Dist", [])
    parsed_requirements = [Requirement(line) for line in requirement_lines]
    unknown_extras = [extra for extra in extras if extra not in available_extras]

    if unknown_extras:
        raise SystemExit(
            "requested extras missing from wheel metadata: " + ", ".join(sorted(unknown_extras))
        )

    resolved: list[str] = []
    seen: set[str] = set()
    base_env = default_environment()

    for extra in extras:
        env = {**base_env, "extra": extra}
        matched_for_extra = False
        for requirement in parsed_requirements:
            if requirement.marker is None or not requirement.marker.evaluate(env):
                continue
            matched_for_extra = True
            rendered = _requirement_without_marker(requirement)
            if rendered in seen:
                continue
            resolved.append(rendered)
            seen.add(rendered)
        if not matched_for_extra:
            raise SystemExit(f"requested extra {extra!r} resolved to no wheel requirements")
    return resolved


def main() -> None:
    args = _parse_args()
    wheel_path = _find_single_wheel(resolve_repo_relative_path(Path(args.artifact_dir)))
    requirements = _resolved_extra_requirements(wheel_path, list(args.extras))

    if args.print_only:
        for requirement in requirements:
            print(requirement)
        return

    if not requirements:
        return

    subprocess.run(
        [sys.executable, "-m", "pip", "install", *requirements],
        check=True,
    )


if __name__ == "__main__":
    main()
