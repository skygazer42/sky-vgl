from __future__ import annotations

import importlib
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parent.parent


def ensure_repo_root_on_path() -> Path:
    repo_root_str = str(REPO_ROOT)
    if repo_root_str in sys.path:
        sys.path.remove(repo_root_str)
    sys.path.insert(0, repo_root_str)
    return REPO_ROOT


def load_repo_module(module_name: str):
    ensure_repo_root_on_path()
    return importlib.import_module(module_name)
