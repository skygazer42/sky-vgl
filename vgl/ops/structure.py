import torch

from vgl.graph.graph import Graph
from vgl.graph.schema import GraphSchema
from vgl.graph.stores import EdgeStore, NodeStore


def _resolve_edge_type(graph: Graph, edge_type=None) -> tuple[str, str, str]:
    if edge_type is None:
        return graph._default_edge_type()
    return tuple(edge_type)


def _edge_store_with_index(store: EdgeStore, edge_index: torch.Tensor) -> EdgeStore:
    edge_count = int(store.edge_index.size(1))
    data = {"edge_index": edge_index}
    extra_rows = edge_index.size(1) - edge_count
    for key, value in store.data.items():
        if key == "edge_index":
            continue
        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.size(0) == edge_count:
            if extra_rows > 0:
                pad_shape = (extra_rows, *value.shape[1:])
                pad = value.new_zeros(pad_shape)
                data[key] = torch.cat((value, pad), dim=0)
            else:
                data[key] = value
        else:
            data[key] = value
    return EdgeStore(store.type_name, data)


def _public_edge_ids(store: EdgeStore) -> torch.Tensor:
    public_ids = store.data.get("e_id")
    if public_ids is None:
        return torch.arange(store.edge_index.size(1), dtype=torch.long, device=store.edge_index.device)
    return torch.as_tensor(public_ids, dtype=torch.long, device=store.edge_index.device).view(-1)


def _edge_pair_keys(edge_index: torch.Tensor, *, dst_count: int) -> torch.Tensor:
    if edge_index.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=edge_index.device)
    return edge_index[0].to(dtype=torch.long) * int(dst_count) + edge_index[1].to(dtype=torch.long)


def _stable_unique_positions(keys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if keys.numel() == 0:
        empty = torch.empty(0, dtype=torch.long, device=keys.device)
        return empty, empty

    sorted_order = torch.argsort(keys, stable=True)
    sorted_keys = keys.index_select(0, sorted_order)
    group_starts = torch.ones(sorted_keys.numel(), dtype=torch.bool, device=keys.device)
    group_starts[1:] = sorted_keys[1:] != sorted_keys[:-1]

    first_positions = sorted_order[group_starts]
    group_ids = torch.cumsum(group_starts.to(dtype=torch.long), dim=0) - 1
    counts = torch.bincount(group_ids)

    order = torch.argsort(first_positions, stable=True)
    return first_positions.index_select(0, order), counts.index_select(0, order)


def _membership_mask(values: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    values = torch.as_tensor(values, dtype=torch.long).view(-1)
    allowed = torch.as_tensor(allowed, dtype=torch.long, device=values.device).view(-1)
    if values.numel() == 0 or allowed.numel() == 0:
        return torch.zeros(values.numel(), dtype=torch.bool, device=values.device)
    sorted_allowed = torch.sort(allowed, stable=True).values
    positions = torch.searchsorted(sorted_allowed, values, right=False)
    capped_positions = positions.clamp_max(sorted_allowed.numel() - 1)
    return (positions < sorted_allowed.numel()) & (sorted_allowed[capped_positions] == values)


def _preserved_time_attr(graph: Graph, node_features: dict[str, tuple[str, ...]], edge_features: dict[tuple[str, str, str], tuple[str, ...]]) -> str | None:
    time_attr = graph.schema.time_attr
    if time_attr is None:
        return None
    if any(time_attr in features for features in node_features.values()):
        return time_attr
    if any(time_attr in features for features in edge_features.values()):
        return time_attr
    return None


def _graph_with_updated_edges(graph: Graph, edge_updates: dict[tuple[str, str, str], EdgeStore]) -> Graph:
    edges = dict(graph.edges)
    edges.update(edge_updates)
    node_features = {
        node_type: tuple(graph.nodes[node_type].data.keys())
        for node_type in graph.schema.node_types
    }
    edge_features = {
        edge_type: tuple(edges[edge_type].data.keys())
        for edge_type in graph.schema.edge_types
    }
    schema = GraphSchema(
        node_types=graph.schema.node_types,
        edge_types=graph.schema.edge_types,
        node_features=node_features,
        edge_features=edge_features,
        time_attr=_preserved_time_attr(graph, node_features, edge_features),
    )
    return Graph(
        schema=schema,
        nodes=dict(graph.nodes),
        edges=edges,
        feature_store=graph.feature_store,
        graph_store=graph.graph_store,
    )


def add_self_loops(graph: Graph, *, edge_type=None) -> Graph:
    edge_type = _resolve_edge_type(graph, edge_type)
    src_type, _, dst_type = edge_type
    if src_type != dst_type:
        raise ValueError("add_self_loops requires matching source and destination node types")
    store = graph.edges[edge_type]
    edge_index = store.edge_index
    num_nodes = graph._node_count(src_type)
    has_self_loop = torch.zeros(num_nodes, dtype=torch.bool, device=edge_index.device)
    self_mask = edge_index[0] == edge_index[1]
    if bool(self_mask.any()):
        has_self_loop[edge_index[0, self_mask].to(dtype=torch.long)] = True
    missing = torch.nonzero(~has_self_loop, as_tuple=False).view(-1).to(dtype=edge_index.dtype)
    if missing.numel() == 0:
        return Graph(schema=graph.schema, nodes=graph.nodes, edges=dict(graph.edges))
    loops = torch.stack((missing, missing), dim=0)
    updated_index = torch.cat((edge_index, loops), dim=1)
    edges = dict(graph.edges)
    edges[edge_type] = _edge_store_with_index(store, updated_index)
    return Graph(schema=graph.schema, nodes=graph.nodes, edges=edges)


def remove_self_loops(graph: Graph, *, edge_type=None) -> Graph:
    edge_type = _resolve_edge_type(graph, edge_type)
    store = graph.edges[edge_type]
    mask = store.edge_index[0] != store.edge_index[1]
    edge_data = Graph._slice_edge_data(store, mask)
    edges = dict(graph.edges)
    edges[edge_type] = EdgeStore(edge_type, edge_data)
    return Graph(schema=graph.schema, nodes=graph.nodes, edges=edges)


def to_bidirected(graph: Graph, *, edge_type=None) -> Graph:
    edge_type = _resolve_edge_type(graph, edge_type)
    src_type, _, dst_type = edge_type
    if src_type != dst_type:
        raise ValueError("to_bidirected requires matching source and destination node types")
    store = graph.edges[edge_type]
    edge_index = store.edge_index
    pair_keys = _edge_pair_keys(edge_index, dst_count=graph._node_count(dst_type))
    reverse_index = edge_index[[1, 0]].contiguous()
    reverse_keys = _edge_pair_keys(reverse_index, dst_count=graph._node_count(src_type))
    missing_mask = ~_membership_mask(reverse_keys, pair_keys)
    if not bool(missing_mask.any()):
        return Graph(schema=graph.schema, nodes=graph.nodes, edges=dict(graph.edges))
    updated_index = torch.cat((edge_index, reverse_index[:, missing_mask]), dim=1)
    edges = dict(graph.edges)
    edges[edge_type] = _edge_store_with_index(store, updated_index)
    return Graph(schema=graph.schema, nodes=graph.nodes, edges=edges)


def to_simple(graph: Graph, *, edge_type=None, count_attr=None) -> Graph:
    edge_type = _resolve_edge_type(graph, edge_type)
    store = graph.edges[edge_type]
    edge_index = store.edge_index
    device = edge_index.device
    _, _, dst_type = edge_type
    representative_tensor, counts = _stable_unique_positions(
        _edge_pair_keys(edge_index, dst_count=graph._node_count(dst_type))
    )
    simple_edge_index = edge_index[:, representative_tensor] if representative_tensor.numel() > 0 else edge_index[:, :0]

    edge_count = int(edge_index.size(1))
    edge_data = {"edge_index": simple_edge_index}
    for key, value in store.data.items():
        if key in {"edge_index", "e_id"}:
            continue
        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.size(0) == edge_count:
            edge_data[key] = value[representative_tensor]
        else:
            edge_data[key] = value
    if count_attr is not None:
        edge_data[str(count_attr)] = counts.to(device=device, dtype=torch.long)

    return _graph_with_updated_edges(graph, {edge_type: EdgeStore(edge_type, edge_data)})


def reverse(graph: Graph, *, copy_ndata: bool = True, copy_edata: bool = False) -> Graph:
    node_stores = {
        node_type: NodeStore(node_type, graph.nodes[node_type].data if copy_ndata else {})
        for node_type in graph.schema.node_types
    }
    node_features = {node_type: tuple(node_stores[node_type].data.keys()) for node_type in graph.schema.node_types}

    edge_stores: dict[tuple[str, str, str], EdgeStore] = {}
    edge_features: dict[tuple[str, str, str], tuple[str, ...]] = {}
    for edge_type in graph.schema.edge_types:
        store = graph.edges[edge_type]
        reversed_edge_type = (edge_type[2], edge_type[1], edge_type[0])
        edge_data = {
            "edge_index": store.edge_index[[1, 0]].contiguous(),
            "e_id": _public_edge_ids(store),
        }
        if copy_edata:
            edge_count = int(store.edge_index.size(1))
            for key, value in store.data.items():
                if key in {"edge_index", "e_id"}:
                    continue
                if isinstance(value, torch.Tensor) and value.ndim > 0 and value.size(0) == edge_count:
                    edge_data[key] = value
                else:
                    edge_data[key] = value
        edge_stores[reversed_edge_type] = EdgeStore(reversed_edge_type, edge_data)
        edge_features[reversed_edge_type] = tuple(edge_data.keys())

    schema = GraphSchema(
        node_types=graph.schema.node_types,
        edge_types=tuple(edge_stores),
        node_features=node_features,
        edge_features=edge_features,
        time_attr=_preserved_time_attr(graph, node_features, edge_features),
    )
    return Graph(
        schema=schema,
        nodes=node_stores,
        edges=edge_stores,
        feature_store=graph.feature_store,
        graph_store=graph.graph_store,
    )
