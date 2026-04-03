import torch

from vgl.graph.block import Block, HeteroBlock
from vgl.graph.graph import Graph
from vgl.ops.subgraph import (
    _ordered_unique,
    _positions_for_endpoint_values,
    _resolve_edge_type,
    _slice_edge_store,
    _slice_node_data,
)


def _validate_destination_nodes(dst_nodes, count: int) -> torch.Tensor:
    dst_nodes = torch.as_tensor(dst_nodes, dtype=torch.long)
    if dst_nodes.ndim > 1:
        raise ValueError("destination nodes must be rank-1")
    dst_nodes = _ordered_unique(dst_nodes.view(-1))
    if ((dst_nodes < 0) | (dst_nodes >= count)).any():
        raise ValueError("destination nodes must fall within the destination node range")
    return dst_nodes


def _ordered_prefix_union(prefix: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    prefix = torch.as_tensor(prefix, dtype=torch.long).view(-1)
    values = torch.as_tensor(values, dtype=torch.long).view(-1)
    combined = torch.cat((prefix, values))
    if combined.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=prefix.device if prefix.numel() > 0 else values.device)
    order = torch.argsort(combined, stable=True)
    sorted_values = combined[order]
    keep = torch.ones(sorted_values.numel(), dtype=torch.bool, device=combined.device)
    if sorted_values.numel() > 1:
        keep[1:] = sorted_values[1:] != sorted_values[:-1]
    first_occurrences = torch.sort(order[keep], stable=True).values
    return combined[first_occurrences]


def _lookup_positions(index_ids: torch.Tensor, values: torch.Tensor, *, entity_name: str) -> torch.Tensor:
    index_ids = torch.as_tensor(index_ids, dtype=torch.long).view(-1)
    values = torch.as_tensor(values, dtype=torch.long, device=index_ids.device).view(-1)
    if values.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=index_ids.device)

    sorted_index_ids, sort_perm = torch.sort(index_ids, stable=True)
    positions = torch.searchsorted(sorted_index_ids, values)
    if bool((positions >= sorted_index_ids.numel()).any()):
        missing_value = int(values[positions >= sorted_index_ids.numel()][0].item())
        raise KeyError(f"missing {entity_name} id {missing_value}")
    matched_values = sorted_index_ids[positions]
    if bool((matched_values != values).any()):
        missing_value = int(values[matched_values != values][0].item())
        raise KeyError(f"missing {entity_name} id {missing_value}")
    return sort_perm[positions]


def _relabel_block_edge_index(edge_index: torch.Tensor, src_node_ids: torch.Tensor, dst_node_ids: torch.Tensor) -> torch.Tensor:
    if edge_index.numel() == 0:
        return edge_index
    return torch.stack(
        (
            _lookup_positions(src_node_ids, edge_index[0], entity_name="block source node"),
            _lookup_positions(dst_node_ids, edge_index[1], entity_name="block destination node"),
        ),
        dim=0,
    )


def _block_store_types(src_type: str, dst_type: str) -> tuple[str, str]:
    if src_type == dst_type:
        return f"{src_type}__src", f"{dst_type}__dst"
    return src_type, dst_type


def _ordered_merge(existing: torch.Tensor | None, values: torch.Tensor) -> torch.Tensor:
    values = torch.as_tensor(values, dtype=torch.long).view(-1)
    if existing is None:
        return values
    return _ordered_prefix_union(existing, values)


def _normalize_dst_nodes_by_type(graph: Graph, dst_nodes_by_type) -> dict[str, torch.Tensor]:
    if isinstance(dst_nodes_by_type, dict):
        normalized = {}
        for node_type, node_ids in dst_nodes_by_type.items():
            node_type = str(node_type)
            if node_type not in graph.nodes:
                raise ValueError(f"unknown destination node type {node_type!r}")
            normalized[node_type] = _validate_destination_nodes(node_ids, graph._node_count(node_type))
        return normalized
    if len(graph.nodes) != 1:
        raise ValueError("heterogeneous to_hetero_block requires destination nodes keyed by node type")
    node_type = next(iter(graph.nodes))
    return {node_type: _validate_destination_nodes(dst_nodes_by_type, graph._node_count(node_type))}


def _resolve_hetero_block_edge_types(graph: Graph, edge_types=None) -> tuple[tuple[str, str, str], ...]:
    if edge_types is None:
        return tuple(graph.edges)
    resolved = []
    for edge_type in edge_types:
        edge_type = tuple(edge_type)
        if edge_type not in graph.edges:
            raise ValueError(f"unknown edge_type for to_hetero_block: {edge_type!r}")
        resolved.append(edge_type)
    return tuple(dict.fromkeys(resolved))


def _hetero_block_store_type_maps(
    src_node_types: tuple[str, ...],
    dst_node_types: tuple[str, ...],
) -> tuple[dict[str, str], dict[str, str]]:
    src_type_set = set(src_node_types)
    dst_type_set = set(dst_node_types)
    src_store_types = {
        node_type: f"{node_type}__src" if node_type in dst_type_set else node_type
        for node_type in src_node_types
    }
    dst_store_types = {
        node_type: f"{node_type}__dst" if node_type in src_type_set else node_type
        for node_type in dst_node_types
    }
    return src_store_types, dst_store_types


def _public_node_ids(graph: Graph, *, node_type: str, local_node_ids: torch.Tensor) -> torch.Tensor:
    local_node_ids = torch.as_tensor(local_node_ids, dtype=torch.long).view(-1)
    public_ids = graph.nodes[node_type].data.get("n_id")
    if public_ids is None:
        return local_node_ids
    public_ids = torch.as_tensor(public_ids, dtype=torch.long, device=local_node_ids.device)
    return public_ids[local_node_ids]


def _public_edge_ids(store, edge_ids: torch.Tensor) -> torch.Tensor:
    edge_ids = torch.as_tensor(edge_ids, dtype=torch.long).view(-1)
    public_ids = store.data.get("e_id")
    if public_ids is None:
        return edge_ids
    public_ids = torch.as_tensor(public_ids, dtype=torch.long, device=edge_ids.device)
    return public_ids[edge_ids]


def _edge_ids_for_block_destinations(store, dst_local_n_id: torch.Tensor) -> torch.Tensor:
    dst_local_n_id = torch.as_tensor(
        dst_local_n_id,
        dtype=torch.long,
        device=store.edge_index.device,
    ).view(-1)
    if dst_local_n_id.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=store.edge_index.device)
    return _positions_for_endpoint_values(store, dst_local_n_id, endpoint=1)


def to_block(graph: Graph, dst_nodes, *, edge_type=None, include_dst_in_src: bool = True) -> Block:
    try:
        edge_type = _resolve_edge_type(graph, edge_type)
    except AttributeError as exc:
        raise ValueError("to_block requires edge_type when the source graph relation is ambiguous") from exc
    src_type, _, dst_type = edge_type
    store = graph.edges[edge_type]
    dst_local_n_id = _validate_destination_nodes(dst_nodes, graph._node_count(dst_type))

    edge_index = store.edge_index
    edge_ids = _edge_ids_for_block_destinations(store, dst_local_n_id)
    selected_edge_index = edge_index[:, edge_ids] if edge_ids.numel() > 0 else edge_index[:, :0]
    predecessor_ids = (
        _ordered_unique(selected_edge_index[0])
        if selected_edge_index.numel() > 0
        else torch.empty((0,), dtype=torch.long, device=edge_index.device)
    )

    if src_type == dst_type and include_dst_in_src:
        src_local_n_id = _ordered_prefix_union(dst_local_n_id.to(device=edge_index.device), predecessor_ids)
    else:
        src_local_n_id = predecessor_ids

    src_store_type, dst_store_type = _block_store_types(src_type, dst_type)
    src_n_id = _public_node_ids(graph, node_type=src_type, local_node_ids=src_local_n_id)
    dst_n_id = _public_node_ids(graph, node_type=dst_type, local_node_ids=dst_local_n_id)

    src_node_data = _slice_node_data(graph, src_local_n_id, node_type=src_type)
    dst_node_data = _slice_node_data(graph, dst_local_n_id, node_type=dst_type)
    src_node_data["n_id"] = src_n_id
    dst_node_data["n_id"] = dst_n_id

    edge_data = _slice_edge_store(store, edge_ids)
    edge_data["e_id"] = _public_edge_ids(store, edge_ids)
    edge_data["edge_index"] = _relabel_block_edge_index(edge_data["edge_index"], src_local_n_id, dst_local_n_id)

    nodes = {src_store_type: src_node_data, dst_store_type: dst_node_data}
    edges = {(src_store_type, edge_type[1], dst_store_type): edge_data}
    if graph.schema.time_attr is None:
        block_graph = Graph.hetero(nodes=nodes, edges=edges)
    else:
        block_graph = Graph.temporal(nodes=nodes, edges=edges, time_attr=graph.schema.time_attr)
    block_graph.feature_store = graph.feature_store
    return Block(
        graph=block_graph,
        edge_type=edge_type,
        src_type=src_type,
        dst_type=dst_type,
        src_n_id=src_n_id,
        dst_n_id=dst_n_id,
        src_store_type=src_store_type,
        dst_store_type=dst_store_type,
    )


def to_hetero_block(
    graph: Graph,
    dst_nodes_by_type,
    *,
    edge_types=None,
    include_dst_in_src: bool = True,
) -> HeteroBlock:
    selected_edge_types = _resolve_hetero_block_edge_types(graph, edge_types=edge_types)
    normalized_dst_nodes = _normalize_dst_nodes_by_type(graph, dst_nodes_by_type)

    src_nodes_by_type = {}
    for edge_type in selected_edge_types:
        src_type, _, dst_type = edge_type
        dst_local_n_id = normalized_dst_nodes.get(
            dst_type,
            torch.empty((0,), dtype=torch.long, device=graph.edges[edge_type].edge_index.device),
        )
        dst_local_n_id = torch.as_tensor(dst_local_n_id, dtype=torch.long).view(-1)
        if include_dst_in_src and src_type == dst_type:
            src_nodes_by_type[src_type] = _ordered_merge(src_nodes_by_type.get(src_type), dst_local_n_id)

        store = graph.edges[edge_type]
        edge_index = store.edge_index
        edge_ids = _edge_ids_for_block_destinations(store, dst_local_n_id)
        selected_edge_index = edge_index[:, edge_ids] if edge_ids.numel() > 0 else edge_index[:, :0]
        predecessor_ids = (
            _ordered_unique(selected_edge_index[0])
            if selected_edge_index.numel() > 0
            else torch.empty((0,), dtype=torch.long, device=edge_index.device)
        )
        src_nodes_by_type[src_type] = _ordered_merge(src_nodes_by_type.get(src_type), predecessor_ids)

    src_node_types = tuple(src_nodes_by_type)
    dst_node_types = tuple(normalized_dst_nodes)
    src_store_types, dst_store_types = _hetero_block_store_type_maps(src_node_types, dst_node_types)

    src_public_ids = {}
    dst_public_ids = {}
    nodes = {}
    for node_type, local_node_ids in src_nodes_by_type.items():
        src_public_ids[node_type] = _public_node_ids(graph, node_type=node_type, local_node_ids=local_node_ids)
        node_data = _slice_node_data(graph, local_node_ids, node_type=node_type)
        node_data["n_id"] = src_public_ids[node_type]
        nodes[src_store_types[node_type]] = node_data
    for node_type, local_node_ids in normalized_dst_nodes.items():
        dst_public_ids[node_type] = _public_node_ids(graph, node_type=node_type, local_node_ids=local_node_ids)
        node_data = _slice_node_data(graph, local_node_ids, node_type=node_type)
        node_data["n_id"] = dst_public_ids[node_type]
        nodes[dst_store_types[node_type]] = node_data

    edges = {}
    for edge_type in selected_edge_types:
        src_type, rel_type, dst_type = edge_type
        store = graph.edges[edge_type]
        dst_local_n_id = normalized_dst_nodes.get(
            dst_type,
            torch.empty((0,), dtype=torch.long, device=store.edge_index.device),
        )
        edge_ids = _edge_ids_for_block_destinations(store, dst_local_n_id)
        edge_data = _slice_edge_store(store, edge_ids)
        edge_data["e_id"] = _public_edge_ids(store, edge_ids)
        edge_data["edge_index"] = _relabel_block_edge_index(
            edge_data["edge_index"],
            torch.as_tensor(src_nodes_by_type[src_type], dtype=torch.long),
            torch.as_tensor(normalized_dst_nodes[dst_type], dtype=torch.long),
        )
        edges[(src_store_types[src_type], rel_type, dst_store_types[dst_type])] = edge_data

    if graph.schema.time_attr is None:
        block_graph = Graph.hetero(nodes=nodes, edges=edges)
    else:
        block_graph = Graph.temporal(nodes=nodes, edges=edges, time_attr=graph.schema.time_attr)
    block_graph.feature_store = graph.feature_store
    return HeteroBlock(
        graph=block_graph,
        edge_types=selected_edge_types,
        src_n_id=src_public_ids,
        dst_n_id=dst_public_ids,
        src_store_types=src_store_types,
        dst_store_types=dst_store_types,
    )
