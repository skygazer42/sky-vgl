import torch

from vgl.graph.graph import Graph


def _ordered_unique(ids: torch.Tensor | list[int] | tuple[int, ...]) -> torch.Tensor:
    ids = torch.as_tensor(ids, dtype=torch.long).view(-1)
    unique = []
    seen = set()
    for value in ids.tolist():
        value = int(value)
        if value not in seen:
            seen.add(value)
            unique.append(value)
    return torch.tensor(unique, dtype=torch.long, device=ids.device)


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


def node_subgraph(graph: Graph, node_ids, *, edge_type=None) -> Graph:
    edge_type = _resolve_edge_type(graph, edge_type)
    src_type, _, dst_type = edge_type
    if src_type != dst_type:
        raise ValueError("node_subgraph currently requires matching source and destination node types")
    node_ids = _ordered_unique(node_ids)
    node_set = set(node_ids.tolist())
    mapping = {node_id: index for index, node_id in enumerate(node_ids.tolist())}
    store = graph.edges[edge_type]
    selected_edges = [
        index
        for index, (src, dst) in enumerate(store.edge_index.t().tolist())
        if int(src) in node_set and int(dst) in node_set
    ]
    edge_ids = torch.tensor(selected_edges, dtype=torch.long, device=store.edge_index.device)
    edge_index = store.edge_index[:, edge_ids] if edge_ids.numel() > 0 else store.edge_index[:, :0]
    if edge_index.numel() > 0:
        relabelled = torch.tensor(
            [[mapping[int(src)], mapping[int(dst)]] for src, dst in edge_index.t().tolist()],
            dtype=edge_index.dtype,
            device=edge_index.device,
        ).t().contiguous()
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
    raise ValueError("node_subgraph heterogeneous support is not implemented yet")


def edge_subgraph(graph: Graph, edge_ids, *, edge_type=None) -> Graph:
    edge_type = _resolve_edge_type(graph, edge_type)
    edge_ids = torch.as_tensor(edge_ids, dtype=torch.long).view(-1)
    store = graph.edges[edge_type]
    edge_data = {"edge_index": store.edge_index[:, edge_ids]}
    edge_count = int(store.edge_index.size(1))
    for key, value in store.data.items():
        if key == "edge_index":
            continue
        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.size(0) == edge_count:
            edge_data[key] = value[edge_ids]
        else:
            edge_data[key] = value
    if len(graph.nodes) == 1 and len(graph.edges) == 1 and edge_type == ("node", "to", "node"):
        return Graph.homo(
            edge_index=edge_data["edge_index"],
            edge_data={k: v for k, v in edge_data.items() if k != "edge_index"},
            **dict(graph.nodes["node"].data),
        )
    raise ValueError("edge_subgraph heterogeneous support is not implemented yet")
