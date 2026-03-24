import torch

from vgl.graph.graph import Graph
from vgl.graph.stores import EdgeStore


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


def add_self_loops(graph: Graph, *, edge_type=None) -> Graph:
    edge_type = _resolve_edge_type(graph, edge_type)
    src_type, _, dst_type = edge_type
    if src_type != dst_type:
        raise ValueError("add_self_loops requires matching source and destination node types")
    store = graph.edges[edge_type]
    edge_index = store.edge_index
    num_nodes = graph._node_count(src_type)
    existing_self = {(int(src), int(dst)) for src, dst in edge_index.t().tolist() if int(src) == int(dst)}
    missing = [node for node in range(num_nodes) if (node, node) not in existing_self]
    if not missing:
        return Graph(schema=graph.schema, nodes=graph.nodes, edges=dict(graph.edges))
    loops = torch.tensor([missing, missing], dtype=edge_index.dtype, device=edge_index.device)
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
    existing = {tuple(edge) for edge in edge_index.t().tolist()}
    reverse_edges = []
    for src, dst in edge_index.t().tolist():
        reverse = (int(dst), int(src))
        if reverse not in existing:
            reverse_edges.append(reverse)
    if not reverse_edges:
        return Graph(schema=graph.schema, nodes=graph.nodes, edges=dict(graph.edges))
    reverse_index = torch.tensor(reverse_edges, dtype=edge_index.dtype, device=edge_index.device).t().contiguous()
    updated_index = torch.cat((edge_index, reverse_index), dim=1)
    edges = dict(graph.edges)
    edges[edge_type] = _edge_store_with_index(store, updated_index)
    return Graph(schema=graph.schema, nodes=graph.nodes, edges=edges)
