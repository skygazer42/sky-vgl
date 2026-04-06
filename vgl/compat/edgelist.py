import torch

from vgl.graph import Graph


def _as_python_int(value) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.detach().cpu().numpy().reshape(()).item())
    return int(value)


def _ensure_homo_graph(graph: Graph) -> None:
    if set(graph.nodes) != {"node"} or set(graph.edges) != {("node", "to", "node")}:
        raise ValueError("to_edge_list currently supports homogeneous graphs only")


def _coerce_tensor(value, *, context: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    try:
        return torch.as_tensor(value)
    except Exception as exc:  # pragma: no cover - defensive path
        raise ValueError(f"{context} must be tensor-like for edge-list interoperability") from exc


def _normalize_feature_data(data, *, entity_kind: str) -> dict[str, torch.Tensor]:
    normalized = {}
    for key, value in dict(data or {}).items():
        normalized[str(key)] = _coerce_tensor(value, context=f"{entity_kind} feature {key!r}")
    return normalized


def _normalize_edge_list(edge_list) -> torch.Tensor:
    if isinstance(edge_list, torch.Tensor):
        tensor = torch.as_tensor(edge_list, dtype=torch.long)
    else:
        if len(edge_list) == 0:
            return torch.empty((2, 0), dtype=torch.long)
        tensor = torch.as_tensor(list(edge_list), dtype=torch.long)

    if tensor.ndim == 1:
        if tensor.numel() == 0:
            return torch.empty((2, 0), dtype=torch.long)
        raise ValueError("edge_list must be shaped (E, 2) or (2, E)")
    if tensor.ndim != 2:
        raise ValueError("edge_list must be shaped (E, 2) or (2, E)")
    if tensor.shape[0] == 2:
        return tensor.clone()
    if tensor.shape[1] == 2:
        return tensor.t().contiguous()
    raise ValueError("edge_list must be shaped (E, 2) or (2, E)")


def _feature_count(feature_data: dict[str, torch.Tensor]) -> int | None:
    count = None
    for key, value in feature_data.items():
        if value.ndim == 0:
            continue
        current = int(value.size(0))
        if count is None:
            count = current
            continue
        if current != count:
            raise ValueError(f"feature {key!r} does not match the inferred entity count")
    return count


def from_edge_list(edge_list, *, num_nodes=None, node_data=None, edge_data=None):
    edge_index = _normalize_edge_list(edge_list)
    normalized_node_data = _normalize_feature_data(node_data, entity_kind="node")
    normalized_edge_data = _normalize_feature_data(edge_data, entity_kind="edge")

    edge_count = int(edge_index.size(1))
    feature_edge_count = _feature_count(normalized_edge_data)
    if feature_edge_count is not None and feature_edge_count != edge_count:
        raise ValueError("edge features must align with the number of edges")

    inferred_num_nodes = int(edge_index.max().detach().cpu().numpy().reshape(()).item()) + 1 if edge_count > 0 else 0
    feature_node_count = _feature_count(normalized_node_data)
    if num_nodes is None:
        resolved_num_nodes = feature_node_count if feature_node_count is not None else inferred_num_nodes
    else:
        resolved_num_nodes = _as_python_int(num_nodes)
    if resolved_num_nodes < inferred_num_nodes:
        raise ValueError("num_nodes must cover all edge endpoints")
    if feature_node_count is not None and feature_node_count != resolved_num_nodes:
        raise ValueError("node features must align with num_nodes")
    if "n_id" not in normalized_node_data and _feature_count(normalized_node_data) is None:
        normalized_node_data["n_id"] = torch.arange(resolved_num_nodes, dtype=torch.long)

    return Graph.homo(
        edge_index=edge_index,
        edge_data=normalized_edge_data or None,
        **normalized_node_data,
    )


def to_edge_list(graph: Graph) -> torch.Tensor:
    _ensure_homo_graph(graph)
    return torch.as_tensor(graph.edge_index, dtype=torch.long).t().contiguous()
