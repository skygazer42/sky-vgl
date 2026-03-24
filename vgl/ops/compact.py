import torch

from vgl.graph.graph import Graph
from vgl.ops.subgraph import _ordered_unique


def compact_nodes(graph: Graph, node_ids, *, edge_type=None) -> tuple[Graph, dict[int, int]]:
    if edge_type is None:
        edge_type = graph._default_edge_type()
    edge_type = tuple(edge_type)
    src_type, _, dst_type = edge_type
    if src_type != dst_type or src_type != "node" or len(graph.nodes) != 1 or len(graph.edges) != 1:
        raise ValueError("compact_nodes currently supports homogeneous graphs only")
    node_ids = _ordered_unique(node_ids)
    mapping = {int(node_id): index for index, node_id in enumerate(node_ids.tolist())}
    edge_index = graph.edges[edge_type].edge_index
    relabelled = torch.tensor(
        [[mapping[int(src)], mapping[int(dst)]] for src, dst in edge_index.t().tolist()],
        dtype=edge_index.dtype,
        device=edge_index.device,
    ).t().contiguous()
    node_data = {}
    node_count = graph._node_count("node")
    for key, value in graph.nodes["node"].data.items():
        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.size(0) == node_count:
            node_data[key] = value[node_ids]
        else:
            node_data[key] = value
    edge_data = {}
    edge_count = int(edge_index.size(1))
    for key, value in graph.edges[edge_type].data.items():
        if key == "edge_index":
            continue
        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.size(0) == edge_count:
            edge_data[key] = value
        else:
            edge_data[key] = value
    return Graph.homo(edge_index=relabelled, edge_data=edge_data, **node_data), mapping
