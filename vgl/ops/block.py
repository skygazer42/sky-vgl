import torch

from vgl.graph.block import Block
from vgl.graph.graph import Graph
from vgl.ops.subgraph import _ordered_unique, _relabel_bipartite_edge_index, _resolve_edge_type, _slice_edge_store, _slice_node_data


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
    ordered = []
    seen = set()
    for tensor in (prefix, values):
        for value in tensor.tolist():
            value = int(value)
            if value in seen:
                continue
            seen.add(value)
            ordered.append(value)
    return torch.tensor(ordered, dtype=torch.long, device=prefix.device if prefix.numel() > 0 else values.device)


def _block_store_types(src_type: str, dst_type: str) -> tuple[str, str]:
    if src_type == dst_type:
        return f"{src_type}__src", f"{dst_type}__dst"
    return src_type, dst_type


def to_block(graph: Graph, dst_nodes, *, edge_type=None, include_dst_in_src: bool = True) -> Block:
    try:
        edge_type = _resolve_edge_type(graph, edge_type)
    except AttributeError as exc:
        raise ValueError("to_block requires edge_type when the source graph relation is ambiguous") from exc
    src_type, _, dst_type = edge_type
    store = graph.edges[edge_type]
    dst_n_id = _validate_destination_nodes(dst_nodes, graph._node_count(dst_type))

    edge_index = store.edge_index
    dst_mask = torch.isin(edge_index[1], dst_n_id.to(device=edge_index.device))
    edge_ids = torch.nonzero(dst_mask, as_tuple=False).view(-1)
    selected_edge_index = edge_index[:, edge_ids] if edge_ids.numel() > 0 else edge_index[:, :0]
    predecessor_ids = _ordered_unique(selected_edge_index[0]) if selected_edge_index.numel() > 0 else torch.empty((0,), dtype=torch.long, device=edge_index.device)

    if src_type == dst_type and include_dst_in_src:
        src_n_id = _ordered_prefix_union(dst_n_id.to(device=edge_index.device), predecessor_ids)
    else:
        src_n_id = predecessor_ids

    src_store_type, dst_store_type = _block_store_types(src_type, dst_type)
    src_mapping = {int(node_id): index for index, node_id in enumerate(src_n_id.tolist())}
    dst_mapping = {int(node_id): index for index, node_id in enumerate(dst_n_id.tolist())}

    src_node_data = _slice_node_data(graph, src_n_id, node_type=src_type)
    dst_node_data = _slice_node_data(graph, dst_n_id, node_type=dst_type)
    src_node_data["n_id"] = src_n_id
    dst_node_data["n_id"] = dst_n_id

    edge_data = _slice_edge_store(store, edge_ids)
    edge_data["e_id"] = edge_ids
    edge_data["edge_index"] = _relabel_bipartite_edge_index(edge_data["edge_index"], src_mapping, dst_mapping)

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
