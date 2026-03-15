import torch


def coerce_homo_inputs(graph_or_x, edge_index, layer_name):
    if edge_index is None:
        if len(graph_or_x.nodes) != 1:
            raise ValueError(f"{layer_name} currently supports homogeneous graphs only")
        return graph_or_x.x, graph_or_x.edge_index
    return graph_or_x, edge_index


def mean_propagate(x, edge_index):
    row, col = edge_index
    out = torch.zeros_like(x)
    out.index_add_(0, col, x[row])
    degree = torch.zeros(x.size(0), dtype=x.dtype, device=x.device)
    degree.index_add_(
        0,
        col,
        torch.ones(col.size(0), dtype=x.dtype, device=x.device),
    )
    return out / degree.clamp_min(1).unsqueeze(-1)
