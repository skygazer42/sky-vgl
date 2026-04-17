import subprocess
import sys
import textwrap
from pathlib import Path

import torch

from vgl import Graph
from vgl.graph import NodeBatch
from vgl.graph.stores import EdgeStore
from vgl.ops import to_block


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_python_script(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


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

    completed = _run_python_script(script)

    assert completed.returncode == 0, completed.stderr


def test_legacy_namespace_imports_do_not_repeat_after_reload():
    expectations = {
        "vgl.core": "vgl.core is a legacy compatibility namespace; prefer `vgl.graph` for graph containers and related errors. See `docs/migration-guide.md` for import rewrite examples.",
        "vgl.data": "vgl.data is a legacy compatibility namespace; prefer `vgl.dataloading` for loaders, samplers, plans, and materialization helpers; dataset and catalog APIs remain under `vgl.data`. See `docs/migration-guide.md` for import rewrite examples.",
        "vgl.train": "vgl.train is a legacy compatibility namespace; prefer `vgl.engine`, `vgl.tasks`, and `vgl.metrics`. See `docs/migration-guide.md` for import rewrite examples.",
    }

    for module_name, expected_message in expectations.items():
        script = textwrap.dedent(
            f"""
            import importlib
            import warnings

            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always", FutureWarning)
                importlib.import_module({module_name!r})
                importlib.reload(importlib.import_module({module_name!r}))
                legacy_messages = [
                    str(warning.message)
                    for warning in captured
                    if str(warning.message).startswith("vgl.")
                ]
                print(len(legacy_messages))
                for message in legacy_messages:
                    print(message)
            """
        )

        completed = _run_python_script(script)

        assert completed.returncode == 0, completed.stderr
        lines = [line for line in completed.stdout.splitlines() if line]
        legacy_messages = lines[1:]
        assert lines[0] == "1"
        assert expected_message in legacy_messages
        assert len(legacy_messages) == len(set(legacy_messages))


def test_import_vgl_root_does_not_emit_legacy_namespace_warnings():
    script = textwrap.dedent(
        """
        import warnings

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always", FutureWarning)
            import vgl
            legacy_messages = [
                str(warning.message)
                for warning in captured
                if str(warning.message).startswith("vgl.")
            ]
            print(len(legacy_messages))
        """
    )

    completed = _run_python_script(script)

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "0"


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
