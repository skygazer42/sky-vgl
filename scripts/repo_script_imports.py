from __future__ import annotations

import importlib
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parent.parent


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


if sys.version_info >= (3, 11):
    def load_toml_file(path: Path):
        import tomllib

        with Path(path).open("rb") as handle:
            return tomllib.load(handle)
else:
    def load_toml_file(path: Path):
        import tomli

        with Path(path).open("rb") as handle:
            return tomli.load(handle)
