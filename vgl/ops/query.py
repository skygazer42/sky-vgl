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


def _all_edges_edge_type(graph: Graph, edge_type=None) -> tuple[str, str, str]:
    if edge_type is not None:
        return tuple(edge_type)
    if len(graph.edges) == 1:
        return next(iter(graph.edges))
    raise ValueError("all_edges requires edge_type when graph has multiple edge types")


def _ordered_edge_positions(store, *, order) -> torch.Tensor:
    if order not in {None, "eid", "srcdst"}:
        raise ValueError("order must be one of None, 'eid', or 'srcdst'")
    count = int(store.edge_index.size(1))
    device = store.edge_index.device
    positions = torch.arange(count, dtype=torch.long, device=device)
    if order is None or count == 0:
        return positions
    public_ids = _public_edge_ids(store)
    positions = positions[torch.argsort(public_ids[positions], stable=True)]
    if order == "eid":
        return positions
    positions = positions[torch.argsort(store.edge_index[1, positions], stable=True)]
    positions = positions[torch.argsort(store.edge_index[0, positions], stable=True)]
    return positions


def _normalize_sparse_layout(layout):
    from vgl.sparse import SparseLayout

    if isinstance(layout, str):
        layout = SparseLayout(layout.lower())
    return layout


def _ordered_edge_tensors(store) -> tuple[torch.Tensor, torch.Tensor]:
    positions = _ordered_edge_positions(store, order="eid")
    if positions.numel() == 0:
        empty = store.edge_index[:, :0]
        return empty, _public_edge_ids(store)[:0]
    return store.edge_index[:, positions], _public_edge_ids(store)[positions]


def _compress_edge_payloads(
    major: torch.Tensor,
    *payloads,
    major_size: int,
) -> tuple:
    order = torch.argsort(major, stable=True)
    major = major[order]
    counts = torch.bincount(major, minlength=major_size)
    pointers = torch.zeros(major_size + 1, dtype=torch.long, device=major.device)
    pointers[1:] = torch.cumsum(counts, dim=0)
    ordered_payloads = tuple(None if payload is None else payload[order] for payload in payloads)
    return (pointers, *ordered_payloads)


def _edge_values(store, positions: torch.Tensor, *, eweight_name: str | None):
    if eweight_name is None:
        return torch.ones(positions.numel(), dtype=torch.float32, device=store.edge_index.device)
    values = store.data.get(eweight_name)
    if values is None:
        raise ValueError(f"unknown edge feature {eweight_name!r}")
    values = torch.as_tensor(values, device=store.edge_index.device)
    return values[positions]


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


def num_nodes(graph: Graph, node_type=None) -> int:
    if node_type is None:
        return sum(graph._node_count(current_type) for current_type in graph.schema.node_types)
    return graph._node_count(str(node_type))


def number_of_nodes(graph: Graph, node_type=None) -> int:
    return num_nodes(graph, node_type)


def num_edges(graph: Graph, edge_type=None) -> int:
    if edge_type is None:
        return sum(int(store.edge_index.size(1)) for store in graph.edges.values())
    return int(graph.edges[tuple(edge_type)].edge_index.size(1))


def number_of_edges(graph: Graph, edge_type=None) -> int:
    return num_edges(graph, edge_type)


def all_edges(graph: Graph, *, form: str = "uv", order: str | None = "eid", edge_type=None):
    edge_type = _all_edges_edge_type(graph, edge_type)
    store = graph.edges[edge_type]
    positions = _ordered_edge_positions(store, order=order)
    return _format_edge_selection(store, positions, form=form)


def adj(graph: Graph, *, edge_type=None, eweight_name: str | None = None, layout="coo"):
    from vgl.sparse import SparseLayout, SparseTensor

    edge_type = _resolve_edge_type(graph, edge_type)
    layout = _normalize_sparse_layout(layout)
    store = graph.edges[edge_type]
    positions = _ordered_edge_positions(store, order="eid")
    ordered, _ = _ordered_edge_tensors(store)
    values = _edge_values(store, positions, eweight_name=eweight_name)
    src_type, _, dst_type = edge_type
    shape = (graph._node_count(src_type), graph._node_count(dst_type))

    if layout is SparseLayout.COO:
        return SparseTensor(
            layout=SparseLayout.COO,
            shape=shape,
            row=ordered[0],
            col=ordered[1],
            values=values,
        )

    if layout is SparseLayout.CSR:
        crow_indices, col_indices, compressed_values = _compress_edge_payloads(
            ordered[0],
            ordered[1],
            values,
            major_size=shape[0],
        )
        return SparseTensor(
            layout=SparseLayout.CSR,
            shape=shape,
            crow_indices=crow_indices,
            col_indices=col_indices,
            values=compressed_values,
        )

    if layout is SparseLayout.CSC:
        ccol_indices, row_indices, compressed_values = _compress_edge_payloads(
            ordered[1],
            ordered[0],
            values,
            major_size=shape[1],
        )
        return SparseTensor(
            layout=SparseLayout.CSC,
            shape=shape,
            ccol_indices=ccol_indices,
            row_indices=row_indices,
            values=compressed_values,
        )

    raise ValueError(f"Unsupported sparse layout: {layout}")


def adj_tensors(graph: Graph, layout="coo", *, edge_type=None):
    from vgl.sparse import SparseLayout

    edge_type = _resolve_edge_type(graph, edge_type)
    layout = _normalize_sparse_layout(layout)
    store = graph.edges[edge_type]
    ordered, public_ids = _ordered_edge_tensors(store)
    src_type, _, dst_type = edge_type

    if layout is SparseLayout.COO:
        return ordered[0], ordered[1]

    if layout is SparseLayout.CSR:
        return _compress_edge_payloads(
            ordered[0],
            ordered[1],
            public_ids,
            major_size=graph._node_count(src_type),
        )

    if layout is SparseLayout.CSC:
        return _compress_edge_payloads(
            ordered[1],
            ordered[0],
            public_ids,
            major_size=graph._node_count(dst_type),
        )

    raise ValueError(f"Unsupported sparse layout: {layout}")


def inc(graph: Graph, typestr: str = "both", *, layout="coo", edge_type=None):
    from vgl.sparse import from_edge_index

    edge_type = _resolve_edge_type(graph, edge_type)
    layout = _normalize_sparse_layout(layout)
    if typestr not in {"in", "out", "both"}:
        raise ValueError("typestr must be one of 'in', 'out', or 'both'")

    store = graph.edges[edge_type]
    src_type, _, dst_type = edge_type
    ordered, _ = _ordered_edge_tensors(store)
    edge_columns = torch.arange(ordered.size(1), dtype=torch.long, device=store.edge_index.device)

    if typestr == "in":
        values = torch.ones(edge_columns.numel(), dtype=torch.float32, device=store.edge_index.device)
        return from_edge_index(
            torch.stack((ordered[1], edge_columns)),
            shape=(graph._node_count(dst_type), edge_columns.numel()),
            layout=layout,
            values=values,
        )

    if typestr == "out":
        values = torch.ones(edge_columns.numel(), dtype=torch.float32, device=store.edge_index.device)
        return from_edge_index(
            torch.stack((ordered[0], edge_columns)),
            shape=(graph._node_count(src_type), edge_columns.numel()),
            layout=layout,
            values=values,
        )

    if src_type != dst_type:
        raise ValueError("typestr='both' requires a relation with the same source and destination node type")

    keep = ordered[0] != ordered[1]
    rows = torch.cat((ordered[0, keep], ordered[1, keep]))
    cols = torch.cat((edge_columns[keep], edge_columns[keep]))
    values = torch.cat(
        (
            -torch.ones(int(keep.sum().item()), dtype=torch.float32, device=store.edge_index.device),
            torch.ones(int(keep.sum().item()), dtype=torch.float32, device=store.edge_index.device),
        )
    )
    return from_edge_index(
        torch.stack((rows, cols)),
        shape=(graph._node_count(src_type), edge_columns.numel()),
        layout=layout,
        values=values,
    )


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
