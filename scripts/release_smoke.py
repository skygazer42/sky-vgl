#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import site
import subprocess
import sys
import tempfile
from pathlib import Path

from contracts import WHEEL_IMPORT_SYMBOLS


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke-test built release artifacts by installing them into an isolated "
            "virtual environment and importing the public package surface."
        )
    )
    parser.add_argument(
        "--artifact-dir",
        default="dist",
        help="Directory containing built wheel and source distribution artifacts.",
    )
    parser.add_argument(
        "--kind",
        choices=("wheel", "sdist", "all"),
        default="all",
        help="Which built artifact kind to validate.",
    )
    return parser.parse_args()


def _find_single(artifact_dir: Path, pattern: str) -> Path:
    matches = sorted(artifact_dir.glob(pattern))
    if len(matches) != 1:
        raise SystemExit(
            f"expected exactly one artifact matching {pattern!r} in {artifact_dir}, "
            f"found {len(matches)}"
        )
    return matches[0]


def _run(command: list[str], *, cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def _venv_binaries(venv_dir: Path) -> tuple[Path, Path]:
    bin_dir = venv_dir / ("Scripts" if os.name == "nt" else "bin")
    return bin_dir / "python", bin_dir / "pip"


def _outer_site_packages() -> list[Path]:
    candidates = []
    for entry in [*site.getsitepackages(), site.getusersitepackages()]:
        path = Path(entry)
        if path.exists():
            candidates.append(path)
    return candidates


def _ensure_build_backend(python_bin: Path, pip_bin: Path) -> None:
    completed = subprocess.run(
        [str(python_bin), "-c", "import hatchling"],
        capture_output=True,
        text=True,
    )
    if completed.returncode == 0:
        return
    _run([str(pip_bin), "install", "hatchling>=1.26.0"])


def _import_check(
    python_bin: Path,
    *,
    cwd: Path,
    repo_root: Path,
    dependency_paths: list[Path],
) -> None:
    bootstrap = "".join(
        f"site.addsitedir({str(path)!r})\n"
        for path in dependency_paths
    )
    root_imports = ", ".join(WHEEL_IMPORT_SYMBOLS)
    symbol_prints = "".join(f"print({symbol})\n" for symbol in WHEEL_IMPORT_SYMBOLS)
    script = (
        "import site\n"
        "from pathlib import Path\n"
        f"{bootstrap}"
        "import vgl\n"
        f"from vgl import {root_imports}\n"
        f"repo_root = Path({str(repo_root)!r}).resolve()\n"
        "module_path = Path(vgl.__file__).resolve()\n"
        "assert repo_root not in module_path.parents, module_path\n"
        "print(vgl.__version__)\n"
        f"{symbol_prints}"
    )
    _run([str(python_bin), "-c", script], cwd=cwd)


def _smoke_install(kind: str, artifact: Path, *, repo_root: Path) -> None:
    with tempfile.TemporaryDirectory(prefix=f"vgl-release-{kind}-") as tmp:
        tmp_dir = Path(tmp)
        venv_dir = tmp_dir / "venv"
        _run([sys.executable, "-m", "venv", str(venv_dir)])
        python_bin, pip_bin = _venv_binaries(venv_dir)
        if kind == "sdist":
            _ensure_build_backend(python_bin, pip_bin)
            install_cmd = [
                str(pip_bin),
                "install",
                "--force-reinstall",
                "--no-deps",
                "--no-build-isolation",
                str(artifact),
            ]
        else:
            install_cmd = [
                str(pip_bin),
                "install",
                "--force-reinstall",
                "--no-deps",
                str(artifact),
            ]
        _run(install_cmd)
        _import_check(
            python_bin,
            cwd=tmp_dir,
            repo_root=repo_root,
            dependency_paths=_outer_site_packages(),
        )
        print(f"{kind} smoke check passed for {artifact.name}")


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    artifact_dir = Path(args.artifact_dir).resolve()

    if not artifact_dir.exists():
        raise SystemExit(f"artifact directory does not exist: {artifact_dir}")

    if args.kind in {"wheel", "all"}:
        _smoke_install("wheel", _find_single(artifact_dir, "*.whl"), repo_root=repo_root)
    if args.kind in {"sdist", "all"}:
        _smoke_install("sdist", _find_single(artifact_dir, "*.tar.gz"), repo_root=repo_root)


if __name__ == "__main__":
    main()
