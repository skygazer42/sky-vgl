import torch

from vgl.graph.graph import Graph
from vgl.graph.schema import GraphSchema
from vgl.graph.stores import EdgeStore, NodeStore


def _as_python_int(value) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.detach().cpu().numpy().reshape(()).item())
    return int(value)


def _ordered_unique(ids: torch.Tensor | list[int] | tuple[int, ...]) -> torch.Tensor:
    ids = torch.as_tensor(ids, dtype=torch.long).view(-1)
    unique = []
    seen = set()
    for value_tensor in ids:
        value = _as_python_int(value_tensor)
        if value not in seen:
            seen.add(value)
            unique.append(value)
    return torch.tensor(unique, dtype=torch.long, device=ids.device)


def _unique_sorted_tensor(values) -> torch.Tensor:
    values = torch.as_tensor(values, dtype=torch.long).view(-1)
    if values.numel() <= 1:
        return values
    if bool((values[1:] > values[:-1]).all()):
        return values
    sorted_values = torch.sort(values, stable=True).values
    keep = torch.ones(sorted_values.numel(), dtype=torch.bool, device=sorted_values.device)
    if sorted_values.numel() > 1:
        keep[1:] = sorted_values[1:] != sorted_values[:-1]
    return sorted_values[keep]


def _slice_node_data(graph: Graph, node_ids: torch.Tensor, *, node_type: str) -> dict[str, torch.Tensor]:
    store = graph.nodes[node_type]
    node_count = graph._node_count(node_type)
    data = {}
    for key, value in store.data.items():
        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.size(0) == node_count:
            data[key] = value[node_ids]
        else:
            data[key] = value
    return data


def _resolve_edge_type(graph: Graph, edge_type=None) -> tuple[str, str, str]:
    if edge_type is None:
        return graph._default_edge_type()
    return tuple(edge_type)


def _slice_edge_store(store, edge_ids: torch.Tensor) -> dict[str, torch.Tensor]:
    edge_data = {"edge_index": store.edge_index[:, edge_ids]}
    edge_count = int(store.edge_index.size(1))
    for key, value in store.data.items():
        if key == "edge_index":
            continue
        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.size(0) == edge_count:
            edge_data[key] = value[edge_ids]
        else:
            edge_data[key] = value
    return edge_data


def _tensor_signature(tensor: torch.Tensor) -> tuple[object, ...]:
    return (
        int(tensor.data_ptr()) if tensor.numel() > 0 else 0,
        tuple(int(dim) for dim in tensor.shape),
        str(tensor.dtype),
        str(tensor.device),
    )


def _left_searchsorted_matches(sorted_values: torch.Tensor, requested: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.searchsorted(sorted_values, requested, right=False)
    if sorted_values.numel() == 0:
        return positions, torch.zeros_like(requested, dtype=torch.bool)
    capped_positions = positions.clamp_max(sorted_values.numel() - 1)
    in_range = positions < sorted_values.numel()
    return positions, in_range & (sorted_values[capped_positions] == requested)


def _expand_interval_values(values: torch.Tensor, counts: torch.Tensor, *, step: int) -> torch.Tensor:
    if values.numel() == 0:
        return values.new_empty(0)

    positive = counts > 0
    if not bool(positive.all()):
        values = values[positive]
        counts = counts[positive]
        if values.numel() == 0:
            return values.new_empty(0)

    offsets = torch.cumsum(counts, dim=0) - counts
    base = values - step * offsets
    deltas = torch.empty_like(base)
    deltas[0] = base[0]
    if base.numel() > 1:
        deltas[1:] = base[1:] - base[:-1]
    markers = torch.zeros(counts.sum(), dtype=values.dtype, device=values.device)
    markers[offsets] = deltas
    expanded = torch.cumsum(markers, dim=0)
    if step != 0:
        expanded = expanded + step * torch.arange(counts.sum(), dtype=values.dtype, device=values.device)
    return expanded


def _endpoint_lookup(store, *, endpoint: int) -> tuple[torch.Tensor, torch.Tensor]:
    if hasattr(store, "query_cache"):
        cache = store.query_cache
    else:
        cache = {}

    cache_key = ("endpoint_lookup", int(endpoint))
    signature = (_tensor_signature(store.edge_index), int(endpoint))
    entry = cache.get(cache_key)
    if entry is not None and entry["signature"] == signature:
        return entry["sorted_endpoint_ids"], entry["sorted_positions"]

    endpoint_ids = store.edge_index[endpoint].to(dtype=torch.long)
    positions = torch.arange(endpoint_ids.numel(), dtype=torch.long, device=endpoint_ids.device)
    order = torch.argsort(endpoint_ids, stable=True)
    entry = {
        "signature": signature,
        "sorted_endpoint_ids": endpoint_ids[order],
        "sorted_positions": positions[order],
    }
    cache[cache_key] = entry
    if hasattr(store, "query_cache"):
        store.query_cache = cache
    return entry["sorted_endpoint_ids"], entry["sorted_positions"]


def _membership_mask(values: torch.Tensor, allowed_ids: torch.Tensor) -> torch.Tensor:
    values = torch.as_tensor(values, dtype=torch.long).view(-1)
    allowed_ids = _unique_sorted_tensor(
        torch.as_tensor(allowed_ids, dtype=torch.long, device=values.device).view(-1)
    )
    if values.numel() == 0 or allowed_ids.numel() == 0:
        return torch.zeros(values.numel(), dtype=torch.bool, device=values.device)
    lower = allowed_ids[0]
    upper = allowed_ids[-1]
    if (upper - lower + 1) == allowed_ids.numel():
        return (values >= lower) & (values <= upper)
    positions = torch.searchsorted(allowed_ids, values, right=False)
    capped_positions = positions.clamp_max(allowed_ids.numel() - 1)
    return (positions < allowed_ids.numel()) & (allowed_ids[capped_positions] == values)


def _positions_for_endpoint_values(store, values: torch.Tensor, *, endpoint: int) -> torch.Tensor:
    values = _unique_sorted_tensor(
        torch.as_tensor(values, dtype=torch.long, device=store.edge_index.device).view(-1)
    )
    if values.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=store.edge_index.device)
    sorted_endpoint_ids, sorted_positions = _endpoint_lookup(store, endpoint=endpoint)
    starts, found = _left_searchsorted_matches(sorted_endpoint_ids, values)
    if not bool(found.any()):
        return torch.empty(0, dtype=torch.long, device=store.edge_index.device)

    starts = starts[found]
    values = values[found]
    ends = torch.searchsorted(sorted_endpoint_ids, values, right=True)
    counts = ends - starts
    positions = sorted_positions[_expand_interval_values(starts, counts, step=1)]
    return torch.sort(positions, stable=True).values


def _lookup_positions(index_ids: torch.Tensor, values: torch.Tensor, *, entity_name: str) -> torch.Tensor:
    index_ids = torch.as_tensor(index_ids, dtype=torch.long).view(-1)
    values = torch.as_tensor(values, dtype=torch.long, device=index_ids.device).view(-1)
    if values.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=index_ids.device)

    sorted_index_ids, sort_perm = torch.sort(index_ids, stable=True)
    positions = torch.searchsorted(sorted_index_ids, values)
    if bool((positions >= sorted_index_ids.numel()).any()):
        missing_value = _as_python_int(values[positions >= sorted_index_ids.numel()][0])
        raise KeyError(f"missing {entity_name} id {missing_value}")
    matched_values = sorted_index_ids[positions]
    if bool((matched_values != values).any()):
        missing_value = _as_python_int(values[matched_values != values][0])
        raise KeyError(f"missing {entity_name} id {missing_value}")
    return sort_perm[positions]


def _relabel_bipartite_edge_index(edge_index, src_node_ids: torch.Tensor, dst_node_ids: torch.Tensor) -> torch.Tensor:
    if edge_index.numel() == 0:
        return edge_index
    src_node_ids = torch.as_tensor(src_node_ids, dtype=torch.long, device=edge_index.device).view(-1)
    dst_node_ids = torch.as_tensor(dst_node_ids, dtype=torch.long, device=edge_index.device).view(-1)
    return torch.stack(
        (
            _lookup_positions(src_node_ids, edge_index[0], entity_name="source node"),
            _lookup_positions(dst_node_ids, edge_index[1], entity_name="destination node"),
        ),
        dim=0,
    )


def _normalize_frontier_nodes(graph: Graph, nodes) -> dict[str, torch.Tensor]:
    node_types = graph.schema.node_types
    if isinstance(nodes, dict):
        frontiers = {}
        valid = set(node_types)
        for node_type, node_ids in nodes.items():
            if node_type not in valid:
                raise ValueError(f"unknown node type {node_type!r}")
            frontiers[node_type] = _ordered_unique(node_ids)
        return frontiers
    if len(node_types) != 1:
        raise ValueError("heterogeneous frontier subgraph requires node ids keyed by node type")
    return {node_types[0]: _ordered_unique(nodes)}


def _public_edge_ids(store, edge_ids: torch.Tensor) -> torch.Tensor:
    edge_ids = torch.as_tensor(edge_ids, dtype=torch.long).view(-1)
    public_ids = store.data.get("e_id")
    if public_ids is None:
        return edge_ids
    public_ids = torch.as_tensor(public_ids, dtype=torch.long, device=edge_ids.device)
    return public_ids[edge_ids]


def _preserved_node_stores(graph: Graph) -> dict[str, NodeStore]:
    return {
        node_type: NodeStore(node_type, graph.nodes[node_type].data)
        for node_type in graph.schema.node_types
    }


def _frontier_edge_ids(store, frontier: torch.Tensor | None, *, endpoint: int) -> torch.Tensor:
    device_frontier = None if frontier is None else frontier.to(device=store.edge_index.device)
    if device_frontier is None or int(device_frontier.numel()) == 0:
        return torch.empty(0, dtype=torch.long, device=store.edge_index.device)
    return _positions_for_endpoint_values(store, device_frontier, endpoint=endpoint)


def _frontier_subgraph(graph: Graph, nodes, *, endpoint: int) -> Graph:
    frontiers = _normalize_frontier_nodes(graph, nodes)
    node_stores = _preserved_node_stores(graph)
    edge_stores = {}
    edge_features = {}
    for edge_type in graph.schema.edge_types:
        store = graph.edges[edge_type]
        node_type = edge_type[2] if endpoint == 1 else edge_type[0]
        edge_ids = _frontier_edge_ids(store, frontiers.get(node_type), endpoint=endpoint)
        edge_data = _slice_edge_store(store, edge_ids)
        edge_data["e_id"] = _public_edge_ids(store, edge_ids)
        edge_stores[edge_type] = EdgeStore(edge_type, edge_data)
        edge_features[edge_type] = tuple(edge_data.keys())
    schema = GraphSchema(
        node_types=graph.schema.node_types,
        edge_types=graph.schema.edge_types,
        node_features={
            node_type: tuple(graph.nodes[node_type].data.keys())
            for node_type in graph.schema.node_types
        },
        edge_features=edge_features,
        time_attr=graph.schema.time_attr,
    )
    return Graph(
        schema=schema,
        nodes=node_stores,
        edges=edge_stores,
        feature_store=graph.feature_store,
        graph_store=graph.graph_store,
    )


def node_subgraph(graph: Graph, node_ids, *, edge_type=None) -> Graph:
    edge_type = _resolve_edge_type(graph, edge_type)
    src_type, _, dst_type = edge_type
    if src_type == dst_type:
        node_ids = _ordered_unique(node_ids)
        store = graph.edges[edge_type]
        node_ids_device = node_ids.to(device=store.edge_index.device)
        candidate_edge_ids = _positions_for_endpoint_values(store, node_ids_device, endpoint=0)
        edge_ids = candidate_edge_ids[
            _membership_mask(store.edge_index[1, candidate_edge_ids], node_ids_device)
        ]
        edge_index = store.edge_index[:, edge_ids] if edge_ids.numel() > 0 else store.edge_index[:, :0]
        if edge_index.numel() > 0:
            relabelled = _relabel_bipartite_edge_index(edge_index, node_ids, node_ids)
        else:
            relabelled = edge_index
        edge_data = {"edge_index": relabelled}
        edge_count = int(store.edge_index.size(1))
        for key, value in store.data.items():
            if key == "edge_index":
                continue
            if isinstance(value, torch.Tensor) and value.ndim > 0 and value.size(0) == edge_count:
                edge_data[key] = value[edge_ids]
            else:
                edge_data[key] = value
        node_data = _slice_node_data(graph, node_ids, node_type=src_type)
        if len(graph.nodes) == 1 and len(graph.edges) == 1 and src_type == "node":
            return Graph.homo(edge_index=relabelled, edge_data={k: v for k, v in edge_data.items() if k != "edge_index"}, **node_data)
        nodes = {src_type: node_data}
        return Graph.hetero(nodes=nodes, edges={edge_type: edge_data}, time_attr=graph.schema.time_attr)

    if not isinstance(node_ids, dict):
        raise ValueError("heterogeneous node_subgraph requires node_ids keyed by node type")
    if src_type not in node_ids or dst_type not in node_ids:
        raise ValueError("heterogeneous node_subgraph requires selections for both source and destination node types")

    src_node_ids = _ordered_unique(node_ids[src_type])
    dst_node_ids = _ordered_unique(node_ids[dst_type])

    store = graph.edges[edge_type]
    src_node_ids_device = src_node_ids.to(device=store.edge_index.device)
    dst_node_ids_device = dst_node_ids.to(device=store.edge_index.device)
    candidate_edge_ids = _positions_for_endpoint_values(store, src_node_ids_device, endpoint=0)
    edge_ids = candidate_edge_ids[
        _membership_mask(store.edge_index[1, candidate_edge_ids], dst_node_ids_device)
    ]
    edge_data = _slice_edge_store(store, edge_ids)
    edge_data["edge_index"] = _relabel_bipartite_edge_index(edge_data["edge_index"], src_node_ids, dst_node_ids)

    nodes = {
        src_type: _slice_node_data(graph, src_node_ids, node_type=src_type),
        dst_type: _slice_node_data(graph, dst_node_ids, node_type=dst_type),
    }
    return Graph.hetero(nodes=nodes, edges={edge_type: edge_data}, time_attr=graph.schema.time_attr)


def edge_subgraph(graph: Graph, edge_ids, *, edge_type=None) -> Graph:
    edge_type = _resolve_edge_type(graph, edge_type)
    edge_ids = torch.as_tensor(edge_ids, dtype=torch.long).view(-1)
    store = graph.edges[edge_type]
    edge_data = _slice_edge_store(store, edge_ids)
    if len(graph.nodes) == 1 and len(graph.edges) == 1 and edge_type == ("node", "to", "node"):
        return Graph.homo(
            edge_index=edge_data["edge_index"],
            edge_data={k: v for k, v in edge_data.items() if k != "edge_index"},
            **dict(graph.nodes["node"].data),
        )

    src_type, _, dst_type = edge_type
    nodes = {src_type: dict(graph.nodes[src_type].data)}
    if dst_type != src_type:
        nodes[dst_type] = dict(graph.nodes[dst_type].data)
    return Graph.hetero(nodes=nodes, edges={edge_type: edge_data}, time_attr=graph.schema.time_attr)


def in_subgraph(graph: Graph, nodes) -> Graph:
    return _frontier_subgraph(graph, nodes, endpoint=1)


def out_subgraph(graph: Graph, nodes) -> Graph:
    return _frontier_subgraph(graph, nodes, endpoint=0)
