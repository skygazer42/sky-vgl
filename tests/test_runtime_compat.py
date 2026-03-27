import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

from vgl import Graph
from vgl.graph import NodeBatch
from vgl.graph.stores import EdgeStore
from vgl.ops import to_block


REPO_ROOT = Path(__file__).resolve().parents[1]


def _driverless_pin_memory_error(self):
    raise RuntimeError(
        "Found no NVIDIA driver on your system. Please check that you have an NVIDIA GPU "
        "and installed a driver from http://www.nvidia.com/Download/index.aspx"
    )


def test_import_vgl_does_not_require_numpy():
    script = textwrap.dedent(
        """
        import builtins

        original_import = builtins.__import__

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "numpy" or name.startswith("numpy."):
                raise ModuleNotFoundError("No module named 'numpy'")
            return original_import(name, globals, locals, fromlist, level)

        builtins.__import__ = guarded_import

        import vgl

        print(vgl.__version__)
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_edge_store_pin_memory_gracefully_skips_when_driver_is_unavailable(monkeypatch):
    store = EdgeStore(
        type_name=("node", "to", "node"),
        data={"edge_index": torch.tensor([[0, 1], [1, 0]]), "meta": {"source": "synthetic"}},
    )

    monkeypatch.setattr(torch.Tensor, "pin_memory", _driverless_pin_memory_error, raising=False)

    pinned = store.pin_memory()

    assert pinned is not store
    assert pinned.edge_index is store.edge_index
    assert not pinned.edge_index.is_pinned()
    assert pinned.data["meta"] is store.data["meta"]


def test_block_pin_memory_gracefully_skips_when_driver_is_unavailable(monkeypatch):
    block = to_block(
        Graph.homo(
            edge_index=torch.tensor([[0, 1], [1, 2]]),
            x=torch.tensor([[1.0], [2.0], [3.0]]),
        ),
        torch.tensor([1, 2]),
    )

    monkeypatch.setattr(torch.Tensor, "pin_memory", _driverless_pin_memory_error, raising=False)

    pinned = block.pin_memory()

    assert pinned is not block
    assert pinned.srcdata["x"] is block.srcdata["x"]
    assert pinned.src_n_id is block.src_n_id
    assert pinned.dst_n_id is block.dst_n_id
    assert not pinned.srcdata["x"].is_pinned()
    assert not pinned.src_n_id.is_pinned()


def test_node_batch_pin_memory_gracefully_skips_when_driver_is_unavailable(monkeypatch):
    batch = NodeBatch(
        graph=Graph.homo(
            edge_index=torch.tensor([[0], [1]], dtype=torch.long),
            x=torch.randn(2, 4),
        ),
        seed_index=torch.tensor([0, 1], dtype=torch.long),
        metadata=[{"seed": 0}, {"seed": 1}],
    )

    monkeypatch.setattr(torch.Tensor, "pin_memory", _driverless_pin_memory_error, raising=False)

    pinned = batch.pin_memory()

    assert pinned is not batch
    assert pinned.graph.x is batch.graph.x
    assert pinned.seed_index is batch.seed_index
    assert not pinned.graph.x.is_pinned()
    assert not pinned.seed_index.is_pinned()
