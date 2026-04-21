import csv
from pathlib import Path

import torch

from vgl.compat.edgelist import from_edge_list, to_edge_list
from vgl.graph import Graph


def _read_rows(path, *, delimiter: str) -> tuple[list[str], list[dict[str, str]]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError("edge-list CSV requires a header row")
        return list(reader.fieldnames), list(reader)


def _resolve_edge_columns(fieldnames, *, src_column: str, dst_column: str, edge_columns):
    if src_column not in fieldnames:
        raise ValueError(f"missing source column {src_column!r}")
    if dst_column not in fieldnames:
        raise ValueError(f"missing destination column {dst_column!r}")
    if edge_columns is None:
        return [name for name in fieldnames if name not in {src_column, dst_column}]
    resolved = [str(name) for name in edge_columns]
    missing = [name for name in resolved if name not in fieldnames]
    if missing:
        raise ValueError(f"missing edge feature columns: {missing!r}")
    return resolved


def _parse_int(value: str, *, context: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{context} must be an integer column") from exc


def _parse_numeric_column(values: list[str], *, column: str) -> torch.Tensor:
    if not values:
        return torch.empty((0,), dtype=torch.float32)
    try:
        return torch.tensor([int(value) for value in values], dtype=torch.long)
    except ValueError:
        try:
            return torch.tensor([float(value) for value in values])
        except ValueError as exc:
            raise ValueError(f"edge feature column {column!r} must be numeric") from exc


def _infer_export_edge_columns(graph: Graph, edge_columns) -> list[str]:
    edge_store = graph.edges[("node", "to", "node")].data
    edge_count = int(graph.edge_index.size(1))
    if edge_columns is None:
        return [
            key
            for key, value in edge_store.items()
            if key != "edge_index"
            and isinstance(value, torch.Tensor)
            and value.ndim == 1
            and value.size(0) == edge_count
        ]
    resolved = [str(name) for name in edge_columns]
    for name in resolved:
        value = edge_store.get(name)
        if not isinstance(value, torch.Tensor) or value.ndim != 1 or value.size(0) != edge_count:
            raise ValueError(
                f"edge feature {name!r} must be a rank-1 tensor aligned to the number of edges"
            )
    return resolved


def _ensure_unique_public_edge_ids(values: torch.Tensor) -> None:
    flattened = values.detach().cpu().numpy().reshape(-1)
    if len({int(value) for value in flattened}) != len(flattened):
        raise ValueError("edge feature 'e_id' must contain unique public ids for edge-list CSV export")


def _scalar_edge_value(values: torch.Tensor, index: int, *, column: str):
    value = values[index]
    if not isinstance(value, torch.Tensor) or value.ndim != 0:
        raise ValueError(f"edge feature {column!r} must contain scalar values for CSV export")
    scalar = value.detach().cpu()
    scalar_value = scalar.numpy().reshape(()).item()
    if scalar.dtype == torch.bool:
        raise ValueError(f"edge feature {column!r} must contain numeric scalar values for CSV export")
    if torch.is_floating_point(scalar):
        python_value = float(scalar_value)
    elif scalar.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        python_value = int(scalar_value)
    else:
        raise ValueError(f"edge feature {column!r} must contain numeric scalar values for CSV export")
    return python_value


def from_edge_list_csv(
    path,
    *,
    src_column: str = "src",
    dst_column: str = "dst",
    edge_columns=None,
    delimiter: str = ",",
    num_nodes=None,
):
    fieldnames, rows = _read_rows(path, delimiter=delimiter)
    resolved_edge_columns = _resolve_edge_columns(
        fieldnames,
        src_column=src_column,
        dst_column=dst_column,
        edge_columns=edge_columns,
    )

    edge_pairs = []
    raw_edge_columns: dict[str, list[str]] = {name: [] for name in resolved_edge_columns}
    for row in rows:
        edge_pairs.append(
            (
                _parse_int(row[src_column], context=f"source column {src_column!r}"),
                _parse_int(row[dst_column], context=f"destination column {dst_column!r}"),
            )
        )
        for name in resolved_edge_columns:
            raw_edge_columns[name].append(row[name])

    edge_data = {
        name: _parse_numeric_column(values, column=name)
        for name, values in raw_edge_columns.items()
    }
    return from_edge_list(edge_pairs, num_nodes=num_nodes, edge_data=edge_data or None)


def to_edge_list_csv(
    graph: Graph,
    path,
    *,
    src_column: str = "src",
    dst_column: str = "dst",
    edge_columns=None,
    delimiter: str = ",",
) -> None:
    edge_list = to_edge_list(graph)
    resolved_edge_columns = _infer_export_edge_columns(graph, edge_columns)
    edge_store = graph.edges[("node", "to", "node")].data
    if "e_id" in resolved_edge_columns:
        _ensure_unique_public_edge_ids(edge_store["e_id"])
    fieldnames = [src_column, dst_column, *resolved_edge_columns]

    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=delimiter, lineterminator="\n")
        writer.writeheader()
        edge_rows = edge_list.detach().cpu().numpy()
        for edge_id, (src_index, dst_index) in enumerate(edge_rows):
            row = {
                src_column: int(src_index),
                dst_column: int(dst_index),
            }
            for name in resolved_edge_columns:
                row[name] = _scalar_edge_value(edge_store[name], edge_id, column=name)
            writer.writerow(row)
