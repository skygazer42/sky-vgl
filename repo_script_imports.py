from __future__ import annotations

import sys

from scripts import repo_script_imports as _shared_repo_script_imports


sys.modules[__name__] = _shared_repo_script_imports
