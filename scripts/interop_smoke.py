#!/usr/bin/env python3

from __future__ import annotations

import argparse
import repo_script_imports
from typing import Sequence

ensure_repo_root_on_path = repo_script_imports.ensure_repo_root_on_path
load_repo_module = repo_script_imports.load_repo_module


REAL_INTEROP_BACKENDS = load_repo_module("scripts.contracts").REAL_INTEROP_BACKENDS

def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test installed optional interop backends through the public VGL API."
    )
    parser.add_argument(
        "--backend",
        choices=(*REAL_INTEROP_BACKENDS, "all"),
        default="all",
        help="Interop backend to validate.",
    )
    parser.add_argument(
        "--list-backends",
        action="store_true",
        help="Print supported backend names and exit.",
    )
    return parser.parse_args(argv)


def list_backends() -> tuple[str, ...]:
    return tuple(REAL_INTEROP_BACKENDS)


def build_smoke_graph():
    ensure_repo_root_on_path()
    import torch
    from vgl import Graph

    return Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        x=torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
        y=torch.tensor([0, 1], dtype=torch.long),
        edge_data={"edge_attr": torch.tensor([[0.25], [0.75]], dtype=torch.float32)},
    )


def _assert_round_trip(original, restored) -> None:
    import torch

    assert torch.equal(restored.edge_index, original.edge_index)
    assert torch.equal(restored.x, original.x)
    assert torch.equal(restored.y, original.y)
    assert torch.equal(restored.edata["edge_attr"], original.edata["edge_attr"])


def _smoke_pyg(graph=None) -> None:
    ensure_repo_root_on_path()
    import torch
    from vgl import Graph
    from vgl.compat import from_pyg, to_pyg

    source_graph = build_smoke_graph() if graph is None else graph
    pyg_graph = to_pyg(source_graph)
    restored = from_pyg(pyg_graph)
    _assert_round_trip(source_graph, restored)
    public_restored = Graph.from_pyg(pyg_graph)
    public_round_trip = public_restored.to_pyg()
    assert torch.equal(public_round_trip.edge_index, pyg_graph.edge_index)
    assert torch.equal(public_round_trip.x, pyg_graph.x)
    assert torch.equal(public_round_trip.y, pyg_graph.y)
    assert torch.equal(public_round_trip.edge_attr, pyg_graph.edge_attr)


def _smoke_dgl(graph=None) -> None:
    ensure_repo_root_on_path()
    import torch
    from vgl import Graph
    from vgl.compat import from_dgl, to_dgl

    source_graph = build_smoke_graph() if graph is None else graph
    dgl_graph = to_dgl(source_graph)
    restored = from_dgl(dgl_graph)
    _assert_round_trip(source_graph, restored)
    public_restored = Graph.from_dgl(dgl_graph)
    public_round_trip = public_restored.to_dgl()
    src, dst = public_round_trip.edges()
    assert torch.equal(torch.stack((src, dst)), source_graph.edge_index)
    assert torch.equal(public_round_trip.ndata["x"], source_graph.x)
    assert torch.equal(public_round_trip.ndata["y"], source_graph.y)
    assert torch.equal(public_round_trip.edata["edge_attr"], source_graph.edata["edge_attr"])


def run_backend_round_trip(backend: str, *, graph=None) -> None:
    if backend not in REAL_INTEROP_BACKENDS:
        raise ValueError(f"unsupported backend {backend!r}; expected one of {', '.join(REAL_INTEROP_BACKENDS)}")

    smoke_fns = {
        "pyg": _smoke_pyg,
        "dgl": _smoke_dgl,
    }
    try:
        smoke_fns[backend](graph)
    except ImportError as exc:
        raise ImportError(f"{backend} interoperability smoke failed: {exc}") from exc


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.list_backends:
        for backend in list_backends():
            print(backend)
        return 0

    backends = list_backends() if args.backend == "all" else (args.backend,)
    for backend in backends:
        run_backend_round_trip(backend)
        print(f"{backend} interop smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
