import csv
from pathlib import Path

import torch

from vgl.compat.edgelist import from_edge_list
from vgl.graph import Graph


def _ensure_homo_graph(graph: Graph) -> None:
    if set(graph.nodes) != {"node"} or set(graph.edges) != {("node", "to", "node")}:
        raise ValueError("to_csv_tables currently supports homogeneous graphs only")


def _read_rows(path, *, delimiter: str) -> tuple[list[str], list[dict[str, str]]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError("CSV tables require a header row")
        return list(reader.fieldnames), list(reader)


def _parse_int(value: str, *, context: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{context} must be an integer column") from exc


def _parse_numeric_column(values: list[str], *, column: str, entity_kind: str) -> torch.Tensor:
    if not values:
        return torch.empty((0,), dtype=torch.float32)
    try:
        return torch.tensor([int(value) for value in values], dtype=torch.long)
    except ValueError:
        try:
            return torch.tensor([float(value) for value in values])
        except ValueError as exc:
            raise ValueError(f"{entity_kind} feature column {column!r} must be numeric") from exc


def _resolve_columns(fieldnames, *, required: tuple[str, ...], selected=None) -> list[str]:
    missing_required = [name for name in required if name not in fieldnames]
    if missing_required:
        if len(missing_required) == 1:
            raise ValueError(f"missing required column {missing_required[0]!r}")
        raise ValueError(f"missing required columns: {missing_required!r}")
    if selected is None:
        excluded = set(required)
        return [name for name in fieldnames if name not in excluded]
    resolved = [str(name) for name in selected]
    missing = [name for name in resolved if name not in fieldnames]
    if missing:
        raise ValueError(f"missing feature columns: {missing!r}")
    return resolved


def _collect_numeric_columns(rows, *, columns: list[str], entity_kind: str) -> dict[str, torch.Tensor]:
    raw_columns: dict[str, list[str]] = {name: [] for name in columns}
    for row in rows:
        for name in columns:
            raw_columns[name].append(row[name])
    return {
        name: _parse_numeric_column(values, column=name, entity_kind=entity_kind)
        for name, values in raw_columns.items()
    }


def _edge_pairs_to_edge_index(edge_pairs: list[tuple[int, int]]) -> torch.Tensor:
    if not edge_pairs:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()


def _node_public_ids(graph: Graph) -> torch.Tensor:
    num_nodes = graph.num_nodes()
    public_ids = graph.ndata.get("n_id")
    if public_ids is None:
        return torch.arange(num_nodes, dtype=torch.long)
    if not isinstance(public_ids, torch.Tensor) or public_ids.ndim != 1 or public_ids.size(0) != num_nodes:
        raise ValueError("node feature 'n_id' must be a rank-1 tensor aligned to the number of nodes")
    try:
        return torch.as_tensor(public_ids, dtype=torch.long)
    except Exception as exc:  # pragma: no cover - defensive path
        raise ValueError("node feature 'n_id' must contain integer public node ids") from exc


def _infer_export_columns(data: dict[str, object], *, count: int, excluded: set[str], entity_kind: str) -> list[str]:
    columns = []
    for key, value in data.items():
        if key in excluded:
            continue
        if isinstance(value, torch.Tensor) and value.ndim == 1 and value.size(0) == count:
            scalar = value[0] if value.numel() > 0 else None
            if scalar is not None and isinstance(scalar, torch.Tensor) and scalar.ndim != 0:
                continue
            columns.append(key)
    return columns


def _resolve_export_columns(data, *, count: int, excluded: set[str], selected, entity_kind: str) -> list[str]:
    if selected is None:
        return _infer_export_columns(data, count=count, excluded=excluded, entity_kind=entity_kind)
    resolved = [str(name) for name in selected]
    for name in resolved:
        value = data.get(name)
        if not isinstance(value, torch.Tensor) or value.ndim != 1 or value.size(0) != count:
            raise ValueError(
                f"{entity_kind} feature {name!r} must be a rank-1 tensor aligned to the number of {entity_kind}s"
            )
    return resolved


def _scalar_numeric_value(values: torch.Tensor, index: int, *, entity_kind: str, column: str):
    value = values[index]
    if not isinstance(value, torch.Tensor) or value.ndim != 0:
        raise ValueError(f"{entity_kind} feature {column!r} must contain scalar values for CSV export")
    python_value = value.detach().cpu().item()
    if isinstance(python_value, bool) or not isinstance(python_value, (int, float)):
        raise ValueError(f"{entity_kind} feature {column!r} must contain numeric scalar values for CSV export")
    return python_value


def from_csv_tables(
    nodes_path,
    edges_path,
    *,
    node_id_column: str = "node_id",
    src_column: str = "src",
    dst_column: str = "dst",
    node_columns=None,
    edge_columns=None,
    delimiter: str = ",",
):
    node_fieldnames, node_rows = _read_rows(nodes_path, delimiter=delimiter)
    resolved_node_columns = _resolve_columns(
        node_fieldnames,
        required=(node_id_column,),
        selected=node_columns,
    )

    public_node_ids = []
    for row in node_rows:
        public_node_ids.append(_parse_int(row[node_id_column], context=f"node id column {node_id_column!r}"))
    if len(set(public_node_ids)) != len(public_node_ids):
        raise ValueError("node id column must contain unique public node ids")
    node_id_to_index = {node_id: index for index, node_id in enumerate(public_node_ids)}

    node_data = _collect_numeric_columns(node_rows, columns=resolved_node_columns, entity_kind="node")
    node_data["n_id"] = torch.tensor(public_node_ids, dtype=torch.long)

    edge_fieldnames, edge_rows = _read_rows(edges_path, delimiter=delimiter)
    resolved_edge_columns = _resolve_columns(
        edge_fieldnames,
        required=(src_column, dst_column),
        selected=edge_columns,
    )

    edge_pairs = []
    for row in edge_rows:
        src = _parse_int(row[src_column], context=f"source column {src_column!r}")
        dst = _parse_int(row[dst_column], context=f"destination column {dst_column!r}")
        if src not in node_id_to_index or dst not in node_id_to_index:
            raise ValueError("edge table references an unknown node id")
        edge_pairs.append((node_id_to_index[src], node_id_to_index[dst]))

    edge_data = _collect_numeric_columns(edge_rows, columns=resolved_edge_columns, entity_kind="edge")
    return from_edge_list(_edge_pairs_to_edge_index(edge_pairs), node_data=node_data, edge_data=edge_data or None)


def to_csv_tables(
    graph: Graph,
    nodes_path,
    edges_path,
    *,
    node_id_column: str = "node_id",
    src_column: str = "src",
    dst_column: str = "dst",
    node_columns=None,
    edge_columns=None,
    delimiter: str = ",",
) -> None:
    _ensure_homo_graph(graph)
    node_count = graph.num_nodes()
    edge_count = graph.num_edges()
    public_ids = _node_public_ids(graph)
    node_columns = _resolve_export_columns(
        graph.ndata,
        count=node_count,
        excluded={"n_id"},
        selected=node_columns,
        entity_kind="node",
    )
    edge_columns = _resolve_export_columns(
        graph.edata,
        count=edge_count,
        excluded={"edge_index"},
        selected=edge_columns,
        entity_kind="edge",
    )

    with Path(nodes_path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[node_id_column, *node_columns],
            delimiter=delimiter,
            lineterminator="\n",
        )
        writer.writeheader()
        for node_id in range(node_count):
            row = {node_id_column: int(public_ids[node_id])}
            for column in node_columns:
                row[column] = _scalar_numeric_value(graph.ndata[column], node_id, entity_kind="node", column=column)
            writer.writerow(row)

    with Path(edges_path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[src_column, dst_column, *edge_columns],
            delimiter=delimiter,
            lineterminator="\n",
        )
        writer.writeheader()
        for edge_id in range(edge_count):
            src = int(public_ids[int(graph.edge_index[0, edge_id])])
            dst = int(public_ids[int(graph.edge_index[1, edge_id])])
            row = {src_column: src, dst_column: dst}
            for column in edge_columns:
                row[column] = _scalar_numeric_value(graph.edata[column], edge_id, entity_kind="edge", column=column)
            writer.writerow(row)
