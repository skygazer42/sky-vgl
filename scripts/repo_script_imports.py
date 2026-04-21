from __future__ import annotations

import importlib
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parent.parent
TOP_LEVEL_MODULE_NAME = "repo_script_imports"
PACKAGE_MODULE_NAME = "scripts.repo_script_imports"


if __name__ == TOP_LEVEL_MODULE_NAME:
    sys.modules.setdefault(PACKAGE_MODULE_NAME, sys.modules[__name__])
elif __name__ == PACKAGE_MODULE_NAME:
    sys.modules.setdefault(TOP_LEVEL_MODULE_NAME, sys.modules[__name__])


def ensure_repo_root_on_path() -> Path:
    repo_root_str = str(REPO_ROOT)
    sys.path[:] = [entry for entry in sys.path if entry != repo_root_str]
    sys.path.insert(0, repo_root_str)
    return REPO_ROOT


def resolve_repo_relative_path(path: Path, *, repo_root: Path = REPO_ROOT) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


def load_repo_module(module_name: str):
    ensure_repo_root_on_path()
    return importlib.import_module(module_name)


def load_toml_file(path: Path):
    loaders = []
    try:
        import tomllib  # type: ignore[attr-defined]

        loaders.append(tomllib)
    except ModuleNotFoundError:
        pass
    try:
        import tomli  # type: ignore[import-not-found]

        loaders.append(tomli)
    except ModuleNotFoundError:
        pass
    try:
        from pip._vendor import tomli as pip_tomli  # type: ignore[import-not-found]

        loaders.append(pip_tomli)
    except ModuleNotFoundError:
        pass

    if not loaders:
        raise ModuleNotFoundError("No TOML parser available; install tomli on Python < 3.11")

    with Path(path).open("rb") as handle:
        for loader in loaders:
            handle.seek(0)
            return loader.load(handle)

    raise RuntimeError(f"unable to load TOML file: {path}")
