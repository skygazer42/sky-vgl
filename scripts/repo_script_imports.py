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
    if repo_root_str in sys.path:
        sys.path.remove(repo_root_str)
    sys.path.insert(0, repo_root_str)
    return REPO_ROOT


def resolve_repo_relative_path(path: Path, *, repo_root: Path = REPO_ROOT) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


def load_repo_module(module_name: str):
    ensure_repo_root_on_path()
    return importlib.import_module(module_name)
