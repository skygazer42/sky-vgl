#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ast
from pathlib import Path
import repo_script_imports
from typing import Sequence

ensure_repo_root_on_path = repo_script_imports.ensure_repo_root_on_path
load_repo_module = repo_script_imports.load_repo_module


_CONTRACTS = None
INTEROP_INSTALL_EXTRAS = {
    "pyg": "pyg",
    "dgl": "dgl",
}


def _contracts_module():
    global _CONTRACTS
    if _CONTRACTS is None:
        _CONTRACTS = load_repo_module("scripts.contracts")
    return _CONTRACTS


def _named_assignments(tree: ast.Module) -> dict[str, ast.AST]:
    assignments: dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assignments[target.id] = node.value
            continue
        if isinstance(node, ast.AnnAssign) and node.value is not None and isinstance(node.target, ast.Name):
            assignments[node.target.id] = node.value
    return assignments


def _real_interop_backends() -> tuple[str, ...]:
    return tuple(_contracts_module().REAL_INTEROP_BACKENDS)


def _string_sequence(
    node: ast.AST,
    assignments: dict[str, ast.AST],
    seen: set[str] | None = None,
) -> tuple[str, ...]:
    if seen is None:
        seen = set()
    if isinstance(node, ast.Name):
        if node.id not in assignments:
            raise RuntimeError(f"scripts/contracts.py missing {node.id!r}")
        if node.id in seen:
            raise RuntimeError(f"scripts/contracts.py contains a recursive backend catalog reference: {node.id}")
        return _string_sequence(assignments[node.id], assignments, seen | {node.id})
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _string_sequence(node.left, assignments, seen) + _string_sequence(node.right, assignments, seen)
    if not isinstance(node, (ast.List, ast.Tuple)):
        raise RuntimeError("REAL_INTEROP_BACKENDS must be a tuple/list or named composition of strings")
    values: list[str] = []
    for item in node.elts:
        values.append(_string_literal(item, assignments, seen))
    return tuple(values)


def _string_literal(
    node: ast.AST,
    assignments: dict[str, ast.AST],
    seen: set[str] | None = None,
) -> str:
    if seen is None:
        seen = set()
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in assignments:
            raise RuntimeError(f"scripts/contracts.py missing {node.id!r}")
        if node.id in seen:
            raise RuntimeError(f"scripts/contracts.py contains a recursive backend string reference: {node.id}")
        return _string_literal(assignments[node.id], assignments, seen | {node.id})
    raise RuntimeError("REAL_INTEROP_BACKENDS must contain only string literals or named string bindings")


def _listable_backends_from_repo(repo_root: Path) -> tuple[str, ...]:
    contracts_path = repo_root / "scripts" / "contracts.py"
    tree = ast.parse(contracts_path.read_text(encoding="utf-8"), filename=str(contracts_path))
    assignments = _named_assignments(tree)
    try:
        real_interop_backends = assignments["REAL_INTEROP_BACKENDS"]
    except KeyError as exc:
        raise RuntimeError("scripts/contracts.py missing REAL_INTEROP_BACKENDS") from exc
    return _string_sequence(real_interop_backends, assignments)


def __getattr__(name: str):
    if name == "REAL_INTEROP_BACKENDS":
        return _real_interop_backends()
    raise AttributeError(name)


def backend_install_extra(backend: str) -> str:
    try:
        return INTEROP_INSTALL_EXTRAS[backend]
    except KeyError as exc:
        real_interop_backends = ", ".join(_real_interop_backends())
        raise ValueError(f"unsupported backend {backend!r}; expected one of {real_interop_backends}") from exc


def backend_install_command(backend: str) -> str:
    return f'pip install "sky-vgl[{backend_install_extra(backend)}]"'


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test installed optional interop backends through the public VGL API."
    )
    parser.add_argument(
        "--backend",
        default="all",
        help="Interop backend to validate. Use --list-backends to print the supported backend names.",
    )
    parser.add_argument(
        "--list-backends",
        action="store_true",
        help="Print supported backend names and exit.",
    )
    args = parser.parse_args(argv)
    if not args.list_backends and args.backend not in {*_real_interop_backends(), "all"}:
        parser.error(f"argument --backend: invalid choice: {args.backend!r}")
    return args


def list_backends() -> tuple[str, ...]:
    return _listable_backends_from_repo(Path(__file__).resolve().parents[1])


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
    real_interop_backends = _real_interop_backends()
    if backend not in real_interop_backends:
        raise ValueError(f"unsupported backend {backend!r}; expected one of {', '.join(real_interop_backends)}")

    smoke_fns = {
        "pyg": _smoke_pyg,
        "dgl": _smoke_dgl,
    }
    try:
        smoke_fns[backend](graph)
    except ImportError as exc:
        message = f"{backend} interoperability smoke failed: {exc}"
        install_hint = backend_install_command(backend)
        if install_hint not in str(exc):
            message = f'{message}. Install it with `{install_hint}`.'
        raise ImportError(message) from exc


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
