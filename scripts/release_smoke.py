#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import repo_script_imports
import site
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Sequence


load_repo_module = repo_script_imports.load_repo_module
resolve_repo_relative_path = repo_script_imports.resolve_repo_relative_path


_CONTRACTS = None


def _contracts_module():
    global _CONTRACTS
    if _CONTRACTS is None:
        _CONTRACTS = load_repo_module("scripts.contracts")
    return _CONTRACTS


def _real_interop_backends() -> tuple[str, ...]:
    return tuple(_contracts_module().REAL_INTEROP_BACKENDS)


def _wheel_import_symbols() -> tuple[str, ...]:
    return tuple(_contracts_module().WHEEL_IMPORT_SYMBOLS)


def _wheel_import_modules() -> tuple[str, ...]:
    return tuple(_contracts_module().WHEEL_IMPORT_MODULES)


def _preferred_import_smokes() -> tuple[tuple[str, str], ...]:
    return tuple(_contracts_module().PREFERRED_IMPORT_SMOKES)


def _optional_extras() -> tuple[str, ...]:
    return tuple(_contracts_module().OPTIONAL_EXTRAS)


def _interop_extra_requirements() -> tuple[str, ...]:
    return tuple(_contracts_module().RELEASE_INTEROP_EXTRA_REQUIREMENTS.values())


def _wheel_required_files() -> tuple[str, ...]:
    return tuple(_contracts_module().WHEEL_REQUIRED_FILES)


def _wheel_excluded_substrings() -> tuple[str, ...]:
    return tuple(_contracts_module().WHEEL_EXCLUDED_SUBSTRINGS)


def _sdist_required_suffixes() -> tuple[str, ...]:
    return tuple(_contracts_module().SDIST_REQUIRED_SUFFIXES)


def _sdist_excluded_substrings() -> tuple[str, ...]:
    return tuple(_contracts_module().SDIST_EXCLUDED_SUBSTRINGS)


def _legacy_import_smokes() -> tuple[tuple[str, str, str], ...]:
    return (
        ("vgl.core", "Graph", "LegacyCoreGraph"),
        ("vgl.train", "Trainer", "LegacyTrainer"),
        ("vgl.data", "Loader", "LegacyLoader"),
    )


def _interop_backends() -> tuple[str, ...]:
    return ("none", *_real_interop_backends(), "all")


def __getattr__(name: str):
    if name == "REAL_INTEROP_BACKENDS":
        return _real_interop_backends()
    if name == "WHEEL_IMPORT_SYMBOLS":
        return _wheel_import_symbols()
    if name == "WHEEL_IMPORT_MODULES":
        return _wheel_import_modules()
    if name == "PREFERRED_IMPORT_SMOKES":
        return _preferred_import_smokes()
    if name == "INTEROP_BACKENDS":
        return _interop_backends()
    raise AttributeError(name)


INTEROP_BACKEND_IMPORT_MODULES = {
    "pyg": "torch_geometric",
    "dgl": "dgl",
}
RELEASE_INTEROP_EXTRA_SITE_DIRS_ENV = "RELEASE_INTEROP_EXTRA_SITE_DIRS"
MAX_WHEEL_BYTES = 5 * 1024 * 1024
MAX_SDIST_BYTES = 10 * 1024 * 1024


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
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
    parser.add_argument(
        "--interop-backend",
        default="none",
        help=(
            "Optionally run backend interop smoke inside the artifact-installed "
            "environment. Supported values are none, all, or a backend name from "
            "`python scripts/interop_smoke.py --list-backends`. Defaults to disabled."
        ),
    )
    parser.add_argument(
        "--max-import-seconds",
        type=float,
        default=None,
        help=(
            "Optional upper bound for installed-artifact import smoke. "
            "When set, the command fails if the root `vgl` import exceeds this limit."
        ),
    )
    parser.add_argument(
        "--max-wheel-bytes",
        type=int,
        default=None,
        help=(
            "Optional upper bound for the built wheel size in bytes. "
            "When unset, the default wheel budget is used."
        ),
    )
    parser.add_argument(
        "--max-sdist-bytes",
        type=int,
        default=None,
        help=(
            "Optional upper bound for the built source distribution size in bytes. "
            "When unset, the default sdist budget is used."
        ),
    )
    args = parser.parse_args(argv)
    if args.interop_backend not in _interop_backends():
        parser.error(f"argument --interop-backend: invalid choice: {args.interop_backend!r}")
    if args.max_import_seconds is not None and args.max_import_seconds <= 0:
        parser.error("argument --max-import-seconds: must be > 0")
    if args.max_wheel_bytes is not None and args.max_wheel_bytes <= 0:
        parser.error("argument --max-wheel-bytes: must be > 0")
    if args.max_sdist_bytes is not None and args.max_sdist_bytes <= 0:
        parser.error("argument --max-sdist-bytes: must be > 0")
    return args


def _find_single(artifact_dir: Path, pattern: str) -> Path:
    matches = sorted(artifact_dir.glob(pattern))
    if len(matches) != 1:
        raise SystemExit(
            f"expected exactly one artifact matching {pattern!r} in {artifact_dir}, "
            f"found {len(matches)}"
        )
    return matches[0]


def _artifact_size_budget(
    kind: str,
    *,
    max_wheel_bytes: int | None = None,
    max_sdist_bytes: int | None = None,
) -> int:
    if kind == "wheel":
        return MAX_WHEEL_BYTES if max_wheel_bytes is None else int(max_wheel_bytes)
    if kind == "sdist":
        return MAX_SDIST_BYTES if max_sdist_bytes is None else int(max_sdist_bytes)
    raise ValueError(f"unsupported artifact kind: {kind}")


def _validate_artifact_size_budget(kind: str, artifact: Path, *, max_bytes: int) -> None:
    artifact_bytes = artifact.stat().st_size
    if artifact_bytes > max_bytes:
        raise SystemExit(
            f"{artifact.name} exceeds {kind} size budget: "
            f"{artifact_bytes} > {max_bytes} bytes"
        )


def _validate_artifact_size(
    kind: str,
    artifact: Path,
    *,
    max_wheel_bytes: int | None = None,
    max_sdist_bytes: int | None = None,
) -> None:
    max_bytes = _artifact_size_budget(
        kind,
        max_wheel_bytes=max_wheel_bytes,
        max_sdist_bytes=max_sdist_bytes,
    )
    _validate_artifact_size_budget(kind, artifact, max_bytes=max_bytes)


def _validate_wheel_metadata(wheel_path: Path) -> None:
    metadata_helper = load_repo_module("scripts.release_artifact_metadata")
    metadata, detail = metadata_helper.read_wheel_metadata(wheel_path)
    if metadata is None:
        raise SystemExit(detail)
    project_name = _contracts_module().PROJECT_NAME
    if metadata.get("Name") != project_name:
        raise SystemExit(
            f"wheel metadata name mismatch for {wheel_path.name}: "
            f"{metadata.get('Name')!r} != {project_name!r}"
        )
    requires_python = _contracts_module().REQUIRES_PYTHON
    if metadata.get("Requires-Python") != requires_python:
        raise SystemExit(
            f"wheel metadata Requires-Python mismatch for {wheel_path.name}: "
            f"{metadata.get('Requires-Python')!r} != {requires_python!r}"
        )
    extras = set(metadata.get_all("Provides-Extra", []))
    missing_extras = [extra for extra in _optional_extras() if extra not in extras]
    if missing_extras:
        raise SystemExit(
            f"wheel metadata missing extras for {wheel_path.name}: {', '.join(missing_extras)}"
        )
    requires_dist = set(metadata.get_all("Requires-Dist", []))
    missing_requirements = [
        requirement for requirement in _interop_extra_requirements() if requirement not in requires_dist
    ]
    if missing_requirements:
        raise SystemExit(
            f"wheel metadata missing interop requirements for {wheel_path.name}: "
            + ", ".join(missing_requirements)
        )
    if not detail.endswith("METADATA"):
        raise SystemExit(f"unexpected wheel metadata path for {wheel_path.name}: {detail}")


def _validate_wheel_contents(wheel_path: Path) -> None:
    with zipfile.ZipFile(wheel_path) as archive:
        names = archive.namelist()

    missing = [name for name in _wheel_required_files() if name not in names]
    if missing:
        raise SystemExit(
            f"missing required wheel files for {wheel_path.name}: " + ", ".join(missing)
        )

    offenders = sorted(
        name
        for name in names
        if any(fragment in name for fragment in _wheel_excluded_substrings())
    )
    if offenders:
        raise SystemExit(
            f"wheel contains excluded content for {wheel_path.name}: " + ", ".join(offenders[:5])
        )


def _validate_sdist_contents(sdist_path: Path) -> None:
    with tarfile.open(sdist_path, "r:gz") as archive:
        names = archive.getnames()

    missing = [
        suffix
        for suffix in _sdist_required_suffixes()
        if not any(name.endswith(suffix) for name in names)
    ]
    if missing:
        raise SystemExit(
            f"missing required sdist files for {sdist_path.name}: " + ", ".join(missing)
        )

    offenders = sorted(
        name
        for name in names
        if any(fragment in name for fragment in _sdist_excluded_substrings())
    )
    if offenders:
        raise SystemExit(
            f"sdist contains excluded content for {sdist_path.name}: " + ", ".join(offenders[:5])
        )


def _preflight_artifact(
    kind: str,
    artifact: Path,
    *,
    max_wheel_bytes: int | None = None,
    max_sdist_bytes: int | None = None,
) -> None:
    _validate_artifact_size(
        kind,
        artifact,
        max_wheel_bytes=max_wheel_bytes,
        max_sdist_bytes=max_sdist_bytes,
    )
    if kind == "wheel":
        _validate_wheel_contents(artifact)
        _validate_wheel_metadata(artifact)
    elif kind == "sdist":
        _validate_sdist_contents(artifact)


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


def _extra_dependency_paths_from_env() -> list[Path]:
    raw = os.environ.get(RELEASE_INTEROP_EXTRA_SITE_DIRS_ENV, "")
    if not raw:
        return []
    extra_paths = []
    seen = set()
    for entry in raw.split(os.pathsep):
        if not entry:
            continue
        path = Path(entry)
        if not path.exists() or path in seen:
            continue
        extra_paths.append(path)
        seen.add(path)
    return extra_paths


def _resolved_dependency_paths() -> list[Path]:
    resolved = []
    seen = set()
    for path in [*_extra_dependency_paths_from_env(), *_outer_site_packages()]:
        if path in seen:
            continue
        resolved.append(path)
        seen.add(path)
    return resolved


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
    max_import_seconds: float | None = None,
) -> None:
    script = _build_import_check_script(
        repo_root=repo_root,
        dependency_paths=dependency_paths,
    )
    completed = subprocess.run(
        [str(python_bin), "-c", script],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )
    import_timings = {}
    for line in completed.stdout.splitlines():
        if not line.startswith("IMPORT_TIMING "):
            continue
        _, module_name, elapsed = line.split()
        import_timings[module_name] = float(elapsed)
    root_import_seconds = import_timings.get("vgl")
    if max_import_seconds is not None and root_import_seconds is not None and root_import_seconds > max_import_seconds:
        raise SystemExit(
            f"artifact import smoke exceeded {max_import_seconds:.3f}s for vgl: "
            f"{root_import_seconds:.3f}s"
        )
    sys.stdout.write(completed.stdout)
    if root_import_seconds is not None:
        print(f"IMPORT_BUDGET_OK vgl {root_import_seconds:.6f}")


def _build_import_check_script(
    *,
    repo_root: Path,
    dependency_paths: list[Path],
) -> str:
    wheel_import_modules = _wheel_import_modules()
    import_module_checks = "".join(
        "start = time.perf_counter()\n"
        f"module = importlib.import_module({module_name!r})\n"
        "elapsed = time.perf_counter() - start\n"
        "module_path = Path(module.__file__).resolve()\n"
        "assert repo_root not in module_path.parents, module_path\n"
        f"print('IMPORT_TIMING {module_name} ' + format(elapsed, '.6f'))\n"
        for module_name in wheel_import_modules
    )
    bootstrap = "".join(
        f"site.addsitedir({str(path)!r})\n"
        for path in dependency_paths
    )
    wheel_import_symbols = _wheel_import_symbols()
    root_imports = ", ".join(wheel_import_symbols)
    preferred_imports = "".join(
        f"from {module_name} import {symbol}\n"
        for module_name, symbol in _preferred_import_smokes()
    )
    legacy_imports = "".join(
        f"from {module_name} import {symbol} as {alias}\n"
        for module_name, symbol, alias in _legacy_import_smokes()
    )
    symbol_prints = "".join(f"print({symbol})\n" for symbol in wheel_import_symbols)
    return (
        "import site\n"
        "import importlib\n"
        "import time\n"
        "from pathlib import Path\n"
        f"{bootstrap}"
        "start = time.perf_counter()\n"
        "import vgl\n"
        "root_elapsed = time.perf_counter() - start\n"
        f"from vgl import {root_imports}\n"
        f"{preferred_imports}"
        f"{legacy_imports}"
        f"repo_root = Path({str(repo_root)!r}).resolve()\n"
        "module_path = Path(vgl.__file__).resolve()\n"
        "assert repo_root not in module_path.parents, module_path\n"
        "print('IMPORT_TIMING vgl ' + format(root_elapsed, '.6f'))\n"
        "print(vgl.__version__)\n"
        f"{import_module_checks}"
        f"{symbol_prints}"
    )


def _selected_interop_backends(backend: str | None) -> tuple[str, ...]:
    real_interop_backends = _real_interop_backends()
    if backend in {None, "none"}:
        return ()
    if backend == "all":
        return real_interop_backends
    if backend in real_interop_backends:
        return (backend,)
    raise ValueError(f"unsupported interop backend: {backend}")


def _backend_import_module_name(backend: str) -> str:
    try:
        return INTEROP_BACKEND_IMPORT_MODULES[backend]
    except KeyError as exc:
        raise ValueError(f"unsupported interop backend: {backend}") from exc


def _check_backend_availability(module_name: str, dependency_paths: list[Path]) -> bool:
    bootstrap = "".join(f"site.addsitedir({str(path)!r})\n" for path in dependency_paths)
    script = "import site\n" f"{bootstrap}" f"import {module_name}\n"
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.returncode == 0


def _preflight_interop_backends(selected_backends: Sequence[str], dependency_paths: list[Path]) -> None:
    missing = []
    for backend in selected_backends:
        module_name = _backend_import_module_name(backend)
        if not _check_backend_availability(module_name, dependency_paths):
            missing.append(f"{backend} ({module_name})")
    if missing:
        raise SystemExit(
            "artifact interop backend(s) unavailable from outer site-packages: "
            + ", ".join(missing)
        )


def _build_interop_check_script(
    backend: str,
    *,
    repo_root: Path,
    dependency_paths: list[Path],
) -> str:
    bootstrap = "".join(f"site.addsitedir({str(path)!r})\n" for path in dependency_paths)
    script = (
        "import site\n"
        "from pathlib import Path\n"
        f"{bootstrap}"
        "import torch\n"
        "import vgl\n"
        "from vgl import Graph\n"
        f"repo_root = Path({str(repo_root)!r}).resolve()\n"
        "module_path = Path(vgl.__file__).resolve()\n"
        "assert repo_root not in module_path.parents, module_path\n"
        "graph = Graph.homo(\n"
        "    edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),\n"
        "    x=torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),\n"
        "    y=torch.tensor([0, 1], dtype=torch.long),\n"
        "    edge_data={'edge_attr': torch.tensor([[0.25], [0.75]], dtype=torch.float32)},\n"
        ")\n"
    )
    if backend == "pyg":
        script += (
            "restored = Graph.from_pyg(graph.to_pyg())\n"
            "assert torch.equal(restored.edge_index, graph.edge_index)\n"
            "assert torch.equal(restored.x, graph.x)\n"
            "assert torch.equal(restored.y, graph.y)\n"
            "assert torch.equal(restored.edata['edge_attr'], graph.edata['edge_attr'])\n"
        )
    elif backend == "dgl":
        script += (
            "restored = Graph.from_dgl(graph.to_dgl())\n"
            "assert torch.equal(restored.edge_index, graph.edge_index)\n"
            "assert torch.equal(restored.x, graph.x)\n"
            "assert torch.equal(restored.y, graph.y)\n"
            "assert torch.equal(restored.edata['edge_attr'], graph.edata['edge_attr'])\n"
        )
    else:
        raise ValueError(f"unsupported interop backend: {backend}")
    return script


def _interop_check(
    python_bin: Path,
    *,
    cwd: Path,
    repo_root: Path,
    dependency_paths: list[Path],
    backend: str,
) -> None:
    selected_backends = _selected_interop_backends(backend)
    _preflight_interop_backends(selected_backends, dependency_paths)
    for selected_backend in selected_backends:
        script = _build_interop_check_script(
            selected_backend,
            repo_root=repo_root,
            dependency_paths=dependency_paths,
        )
        _run([str(python_bin), "-c", script], cwd=cwd)
        print(f"{selected_backend} interop smoke check passed")


def _smoke_install(
    kind: str,
    artifact: Path,
    *,
    repo_root: Path,
    interop_backend: str,
    max_import_seconds: float | None = None,
    max_wheel_bytes: int | None = None,
    max_sdist_bytes: int | None = None,
) -> None:
    _preflight_artifact(
        kind,
        artifact,
        max_wheel_bytes=max_wheel_bytes,
        max_sdist_bytes=max_sdist_bytes,
    )
    with tempfile.TemporaryDirectory(prefix=f"vgl-release-{kind}-") as tmp:
        tmp_dir = Path(tmp)
        venv_dir = tmp_dir / "venv"
        _run([sys.executable, "-m", "venv", str(venv_dir)])
        python_bin, pip_bin = _venv_binaries(venv_dir)
        dependency_paths = _resolved_dependency_paths()
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
            dependency_paths=dependency_paths,
            max_import_seconds=max_import_seconds,
        )
        _interop_check(
            python_bin,
            cwd=tmp_dir,
            repo_root=repo_root,
            dependency_paths=dependency_paths,
            backend=interop_backend,
        )
        print(f"{kind} smoke check passed for {artifact.name}")


def main() -> None:
    args = _parse_args()
    repo_root = repo_script_imports.REPO_ROOT
    artifact_dir = resolve_repo_relative_path(Path(args.artifact_dir))

    if not artifact_dir.exists():
        raise SystemExit(f"artifact directory does not exist: {artifact_dir}")

    if args.kind in {"wheel", "all"}:
        _smoke_install(
            "wheel",
            _find_single(artifact_dir, "*.whl"),
            repo_root=repo_root,
            interop_backend=args.interop_backend,
            max_import_seconds=args.max_import_seconds,
            max_wheel_bytes=args.max_wheel_bytes,
            max_sdist_bytes=args.max_sdist_bytes,
        )
    if args.kind in {"sdist", "all"}:
        _smoke_install(
            "sdist",
            _find_single(artifact_dir, "*.tar.gz"),
            repo_root=repo_root,
            interop_backend=args.interop_backend,
            max_import_seconds=args.max_import_seconds,
            max_wheel_bytes=args.max_wheel_bytes,
            max_sdist_bytes=args.max_sdist_bytes,
        )


if __name__ == "__main__":
    main()
