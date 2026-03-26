import torch

from vgl.graph.graph import Graph


def _resolve_edge_type(graph: Graph, edge_type=None) -> tuple[str, str, str]:
    if edge_type is None:
        return graph._default_edge_type()
    return tuple(edge_type)


def _public_edge_ids(store) -> torch.Tensor:
    public_ids = store.data.get("e_id")
    if public_ids is None:
        return torch.arange(store.edge_index.size(1), dtype=torch.long, device=store.edge_index.device)
    return torch.as_tensor(public_ids, dtype=torch.long, device=store.edge_index.device).view(-1)


def _edge_positions_by_public_id(store) -> dict[int, int]:
    return {int(edge_id): index for index, edge_id in enumerate(_public_edge_ids(store).tolist())}


def _pair_positions(store) -> dict[tuple[int, int], list[int]]:
    positions: dict[tuple[int, int], list[int]] = {}
    for index, (src, dst) in enumerate(store.edge_index.t().tolist()):
        positions.setdefault((int(src), int(dst)), []).append(index)
    return positions


def _normalize_edge_ids(edge_ids) -> torch.Tensor:
    return torch.as_tensor(edge_ids, dtype=torch.long).view(-1)


def _normalize_node_pairs(u, v) -> tuple[torch.Tensor, torch.Tensor, bool]:
    u_tensor = torch.as_tensor(u, dtype=torch.long)
    v_tensor = torch.as_tensor(v, dtype=torch.long)
    scalar_input = u_tensor.ndim == 0 and v_tensor.ndim == 0
    u_ids = u_tensor.view(-1)
    v_ids = v_tensor.view(-1)
    if u_ids.numel() != v_ids.numel():
        raise ValueError("u and v must describe the same number of node pairs")
    return u_ids, v_ids, scalar_input


def _normalize_node_ids(nodes) -> torch.Tensor:
    return torch.as_tensor(nodes, dtype=torch.long).view(-1)


def _normalize_optional_node_ids(nodes) -> tuple[torch.Tensor | None, bool]:
    if nodes is None:
        return None, False
    node_tensor = torch.as_tensor(nodes, dtype=torch.long)
    return node_tensor.view(-1), node_tensor.ndim == 0


def _normalize_single_node(node, *, name: str) -> int:
    node_tensor = torch.as_tensor(node, dtype=torch.long)
    if node_tensor.numel() != 1:
        raise ValueError(f"{name} requires a single node id")
    return int(node_tensor.view(-1)[0])


def _validate_node_ids(graph: Graph, node_type: str, node_ids: torch.Tensor, *, role: str) -> None:
    count = graph._node_count(node_type)
    if torch.any((node_ids < 0) | (node_ids >= count)):
        raise ValueError(f"{role} node ids are out of range")


def _validate_node_pairs(graph: Graph, edge_type, u_ids: torch.Tensor, v_ids: torch.Tensor) -> None:
    src_type, _, dst_type = edge_type
    _validate_node_ids(graph, src_type, u_ids, role="source")
    _validate_node_ids(graph, dst_type, v_ids, role="destination")


def _edge_positions_for_endpoint(graph: Graph, edge_type, nodes, *, endpoint: int) -> tuple[object, torch.Tensor]:
    store = graph.edges[edge_type]
    node_ids = _normalize_node_ids(nodes)
    node_type = edge_type[0] if endpoint == 0 else edge_type[2]
    role = "source" if endpoint == 0 else "destination"
    _validate_node_ids(graph, node_type, node_ids, role=role)
    if node_ids.numel() == 0:
        return store, torch.empty(0, dtype=torch.long, device=store.edge_index.device)
    device_nodes = node_ids.to(device=store.edge_index.device)
    mask = torch.isin(store.edge_index[endpoint], device_nodes)
    return store, torch.nonzero(mask, as_tuple=False).view(-1)


def _format_edge_selection(store, positions: torch.Tensor, *, form: str):
    if form not in {"uv", "eid", "all"}:
        raise ValueError("form must be one of 'uv', 'eid', or 'all'")
    edge_index = store.edge_index[:, positions] if positions.numel() > 0 else store.edge_index[:, :0]
    edge_ids = _public_edge_ids(store)[positions]
    if form == "uv":
        return edge_index[0], edge_index[1]
    if form == "eid":
        return edge_ids
    return edge_index[0], edge_index[1], edge_ids


def _degrees_for_endpoint(graph: Graph, edge_type, nodes, *, endpoint: int):
    store = graph.edges[edge_type]
    node_ids, scalar_input = _normalize_optional_node_ids(nodes)
    node_type = edge_type[0] if endpoint == 0 else edge_type[2]
    role = "source" if endpoint == 0 else "destination"
    node_count = graph._node_count(node_type)
    degrees = torch.bincount(store.edge_index[endpoint], minlength=node_count)
    if node_ids is None:
        return degrees
    _validate_node_ids(graph, node_type, node_ids, role=role)
    if node_ids.numel() == 0:
        return torch.empty(0, dtype=degrees.dtype, device=degrees.device)
    selected = degrees.index_select(0, node_ids.to(device=degrees.device))
    if scalar_input:
        return int(selected[0].item())
    return selected


def find_edges(graph: Graph, eids, *, edge_type=None) -> tuple[torch.Tensor, torch.Tensor]:
    edge_type = _resolve_edge_type(graph, edge_type)
    store = graph.edges[edge_type]
    requested = _normalize_edge_ids(eids)
    positions_by_id = _edge_positions_by_public_id(store)
    positions: list[int] = []
    for edge_id in requested.tolist():
        try:
            positions.append(positions_by_id[int(edge_id)])
        except KeyError as exc:
            raise ValueError(f"unknown edge id {edge_id}") from exc
    if not positions:
        empty = torch.empty(0, dtype=store.edge_index.dtype, device=store.edge_index.device)
        return empty, empty
    position_tensor = torch.tensor(positions, dtype=torch.long, device=store.edge_index.device)
    edge_index = store.edge_index[:, position_tensor]
    return edge_index[0], edge_index[1]


def edge_ids(graph: Graph, u, v, *, return_uv: bool = False, edge_type=None):
    edge_type = _resolve_edge_type(graph, edge_type)
    store = graph.edges[edge_type]
    u_ids, v_ids, _ = _normalize_node_pairs(u, v)
    _validate_node_pairs(graph, edge_type, u_ids, v_ids)
    positions = _pair_positions(store)
    public_ids = _public_edge_ids(store).tolist()
    device = store.edge_index.device

    if return_uv:
        matched_src: list[int] = []
        matched_dst: list[int] = []
        matched_eids: list[int] = []
        for src, dst in zip(u_ids.tolist(), v_ids.tolist()):
            matches = positions.get((int(src), int(dst)))
            if not matches:
                raise ValueError(f"no edge exists between {src} and {dst}")
            for index in matches:
                matched_src.append(int(src))
                matched_dst.append(int(dst))
                matched_eids.append(int(public_ids[index]))
        return (
            torch.tensor(matched_src, dtype=torch.long, device=device),
            torch.tensor(matched_dst, dtype=torch.long, device=device),
            torch.tensor(matched_eids, dtype=torch.long, device=device),
        )

    matched: list[int] = []
    for src, dst in zip(u_ids.tolist(), v_ids.tolist()):
        matches = positions.get((int(src), int(dst)))
        if not matches:
            raise ValueError(f"no edge exists between {src} and {dst}")
        matched.append(int(public_ids[matches[0]]))
    return torch.tensor(matched, dtype=torch.long, device=device)


def has_edges_between(graph: Graph, u, v, *, edge_type=None):
    edge_type = _resolve_edge_type(graph, edge_type)
    store = graph.edges[edge_type]
    u_ids, v_ids, scalar_input = _normalize_node_pairs(u, v)
    _validate_node_pairs(graph, edge_type, u_ids, v_ids)
    positions = _pair_positions(store)
    exists = [((int(src), int(dst)) in positions) for src, dst in zip(u_ids.tolist(), v_ids.tolist())]
    if scalar_input:
        return bool(exists[0])
    return torch.tensor(exists, dtype=torch.bool, device=store.edge_index.device)


def in_degrees(graph: Graph, v=None, *, edge_type=None):
    edge_type = _resolve_edge_type(graph, edge_type)
    return _degrees_for_endpoint(graph, edge_type, v, endpoint=1)


def out_degrees(graph: Graph, u=None, *, edge_type=None):
    edge_type = _resolve_edge_type(graph, edge_type)
    return _degrees_for_endpoint(graph, edge_type, u, endpoint=0)


def in_edges(graph: Graph, v, *, form: str = "uv", edge_type=None):
    edge_type = _resolve_edge_type(graph, edge_type)
    store, positions = _edge_positions_for_endpoint(graph, edge_type, v, endpoint=1)
    return _format_edge_selection(store, positions, form=form)


def out_edges(graph: Graph, u, *, form: str = "uv", edge_type=None):
    edge_type = _resolve_edge_type(graph, edge_type)
    store, positions = _edge_positions_for_endpoint(graph, edge_type, u, endpoint=0)
    return _format_edge_selection(store, positions, form=form)


def predecessors(graph: Graph, v, *, edge_type=None) -> torch.Tensor:
    edge_type = _resolve_edge_type(graph, edge_type)
    node_id = _normalize_single_node(v, name="predecessors")
    store, positions = _edge_positions_for_endpoint(graph, edge_type, torch.tensor([node_id]), endpoint=1)
    if positions.numel() == 0:
        return torch.empty(0, dtype=store.edge_index.dtype, device=store.edge_index.device)
    return store.edge_index[0, positions]


def successors(graph: Graph, v, *, edge_type=None) -> torch.Tensor:
    edge_type = _resolve_edge_type(graph, edge_type)
    node_id = _normalize_single_node(v, name="successors")
    store, positions = _edge_positions_for_endpoint(graph, edge_type, torch.tensor([node_id]), endpoint=0)
    if positions.numel() == 0:
        return torch.empty(0, dtype=store.edge_index.dtype, device=store.edge_index.device)
    return store.edge_index[1, positions]
