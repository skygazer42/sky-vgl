import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

from scripts import interop_smoke


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "interop_smoke.py"


def _load_interop_smoke_module(module_name: str = "interop_smoke"):
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _run_interop_smoke(tmp_path: Path, backend: str, fake_packages: dict[str, str]):
    package_root = tmp_path / "fake_packages"
    for relative_path, content in fake_packages.items():
        path = package_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(content), encoding="utf-8")

    env = os.environ.copy()
    pythonpath = os.pathsep.join((str(package_root), str(REPO_ROOT)))
    env["PYTHONPATH"] = pythonpath
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--backend", backend],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=env,
    )


def test_interop_smoke_lists_supported_backends():
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--list-backends"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == ["pyg", "dgl"]


def test_module_lists_stable_backend_catalog():
    assert interop_smoke.list_backends() == ("pyg", "dgl")


def test_interop_smoke_runs_pyg_round_trip_with_fake_backend(tmp_path):
    completed = _run_interop_smoke(
        tmp_path,
        "pyg",
        {
            "torch_geometric/__init__.py": "from . import data\n",
            "torch_geometric/data.py": """
class Data:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
""",
        },
    )

    assert completed.returncode == 0, completed.stderr
    assert "pyg interop smoke passed" in completed.stdout


def test_interop_smoke_runs_dgl_round_trip_with_fake_backend(tmp_path):
    completed = _run_interop_smoke(
        tmp_path,
        "dgl",
        {
            "dgl.py": """
import torch

NID = "dgl.NID"
EID = "dgl.EID"


class FakeDGLGraph:
    def __init__(self, edges, num_nodes=None):
        self._src, self._dst = edges
        if num_nodes is None:
            num_nodes = int(torch.max(torch.cat((self._src, self._dst))).item()) + 1
        self._num_nodes = num_nodes
        self.ndata = {}
        self.edata = {}

    def edges(self):
        return self._src, self._dst

    def num_nodes(self, node_type=None):
        return self._num_nodes


def graph(edges, num_nodes=None):
    return FakeDGLGraph(edges, num_nodes=num_nodes)
""",
        },
    )

    assert completed.returncode == 0, completed.stderr
    assert "dgl interop smoke passed" in completed.stdout


def test_run_backend_round_trip_wraps_import_error(monkeypatch):
    def _missing_backend(_graph=None):
        raise ImportError('Install it with `pip install "sky-vgl[pyg]"`.')

    monkeypatch.setattr(interop_smoke, "_smoke_pyg", _missing_backend)

    with pytest.raises(ImportError, match="pyg interoperability smoke failed"):
        interop_smoke.run_backend_round_trip("pyg")


def test_main_supports_backend_all(monkeypatch, capsys):
    calls: list[str] = []

    def _fake_run_backend_round_trip(backend: str, *, graph=None):
        del graph
        calls.append(backend)

    monkeypatch.setattr(interop_smoke, "run_backend_round_trip", _fake_run_backend_round_trip)

    exit_code = interop_smoke.main(["--backend", "all"])

    assert exit_code == 0
    assert calls == ["pyg", "dgl"]
    assert capsys.readouterr().out.splitlines() == ["pyg interop smoke passed", "dgl interop smoke passed"]
