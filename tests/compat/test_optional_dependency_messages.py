import importlib

import pytest
import torch

from vgl import Graph
from vgl.compat.pyg import to_pyg
from vgl.engine.logging import _tensorboard_summary_writer_class


def _block_import_module(package_name: str):
    original_import_module = importlib.import_module

    def _guarded_import_module(name, package=None):
        if name == package_name or name.startswith(f"{package_name}."):
            raise ModuleNotFoundError(f"No module named {package_name!r}")
        return original_import_module(name, package)

    return _guarded_import_module


def test_networkx_adapter_error_suggests_install_extra(monkeypatch):
    graph = Graph.homo(
        edge_index=torch.tensor([[0], [1]]),
        x=torch.tensor([[1.0], [2.0]]),
    )

    monkeypatch.setattr(importlib, "import_module", _block_import_module("networkx"))

    with pytest.raises(ImportError, match='pip install "vgl\\[networkx\\]"'):
        graph.to_networkx()


def test_pyg_adapter_error_suggests_install_extra(monkeypatch):
    graph = Graph.homo(
        edge_index=torch.tensor([[0], [1]]),
        x=torch.tensor([[1.0], [2.0]]),
    )

    monkeypatch.setattr(importlib, "import_module", _block_import_module("torch_geometric"))

    with pytest.raises(ImportError, match='pip install "vgl\\[pyg\\]"'):
        to_pyg(graph)


def test_dgl_adapter_error_suggests_install_extra(monkeypatch):
    graph = Graph.homo(
        edge_index=torch.tensor([[0], [1]]),
        x=torch.tensor([[1.0], [2.0]]),
    )

    monkeypatch.setattr(importlib, "import_module", _block_import_module("dgl"))

    with pytest.raises(ImportError, match='pip install "vgl\\[dgl\\]"'):
        graph.to_dgl()


def test_tensorboard_logger_error_suggests_install_extra(monkeypatch):
    monkeypatch.setattr(importlib, "import_module", _block_import_module("torch.utils.tensorboard"))

    with pytest.raises(ImportError, match='pip install "vgl\\[tensorboard\\]"'):
        _tensorboard_summary_writer_class()
