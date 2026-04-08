#!/usr/bin/env python3

from __future__ import annotations

import argparse
import repo_script_imports
import subprocess
import sys
import tempfile
from pathlib import Path


load_toml_file = repo_script_imports.load_toml_file


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit runtime requirements with pip-audit.")
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


def _runtime_requirements(repo_root: Path) -> list[str]:
    payload = load_toml_file(repo_root / "pyproject.toml")
    dependencies = payload["project"]["dependencies"]
    return [str(requirement) for requirement in dependencies]


def main() -> int:
    args = _parse_args()
    repo_root = args.repo_root.resolve()
    requirements = _runtime_requirements(repo_root)

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
