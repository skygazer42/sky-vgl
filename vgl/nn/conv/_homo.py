import torch

from vgl.sparse import edge_softmax as sparse_edge_softmax
from vgl.sparse import from_edge_index


def _as_python_int(value) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.detach().cpu().numpy().reshape(()).item())
    return int(value)


def coerce_homo_inputs(graph_or_x, edge_index, layer_name):
    if edge_index is None:
        if len(graph_or_x.nodes) != 1:
            raise ValueError(f"{layer_name} currently supports homogeneous graphs only")
        return graph_or_x.x, graph_or_x.edge_index
    return graph_or_x, edge_index


def resolve_node_feature(graph_or_x, use_graph_data, value, key, layer_name):
    if value is not None:
        return value
    if use_graph_data:
        if key in graph_or_x.ndata:
            return graph_or_x.ndata[key]
        raise ValueError(f"{layer_name} requires {key} in graph.ndata or as an explicit argument")
    raise ValueError(f"{layer_name} requires {key} when called with x and edge_index")


def mean_propagate(x, edge_index):
    row, col = edge_index
    out = torch.zeros_like(x)
    out.index_add_(0, col, x[row])
    degree = node_degree(edge_index, x.size(0), x.dtype, x.device)
    return out / degree.clamp_min(1).unsqueeze(-1)


def symmetric_propagate(x, edge_index):
    row, col = edge_index
    degree = torch.zeros(x.size(0), dtype=x.dtype, device=x.device)
    ones = torch.ones(col.size(0), dtype=x.dtype, device=x.device)
    degree.index_add_(0, row, ones)
    degree.index_add_(0, col, ones)
    norm = degree[row].clamp_min(1).pow(-0.5) * degree[col].clamp_min(1).pow(-0.5)
    out = torch.zeros_like(x)
    out.index_add_(0, col, x[row] * norm.unsqueeze(-1))
    return out


def edge_softmax(scores, edge_index, num_nodes):
    num_rows = 0
    if edge_index.numel() > 0:
        num_rows = _as_python_int(edge_index[0].max()) + 1
    sparse = from_edge_index(edge_index, shape=(num_rows, num_nodes))
    return sparse_edge_softmax(sparse, scores)


def propagate_steps(x, edge_index, steps):
    outputs = [x]
    current = x
    for _ in range(steps):
        current = symmetric_propagate(current, edge_index)
        outputs.append(current)
    return outputs


def sum_propagate(x, edge_index):
    row, col = edge_index
    out = torch.zeros_like(x)
    out.index_add_(0, col, x[row])
    return out


def max_propagate(x, edge_index):
    row, col = edge_index
    out = torch.full_like(x, float("-inf"))
    index = col.unsqueeze(-1).expand(-1, x.size(-1))
    out.scatter_reduce_(0, index, x[row], reduce="amax", include_self=True)
    return torch.where(torch.isneginf(out), torch.zeros_like(out), out)


def node_degree(edge_index, num_nodes, dtype, device):
    _, col = edge_index
    degree = torch.zeros(num_nodes, dtype=dtype, device=device)
    degree.index_add_(
        0,
        col,
        torch.ones(col.size(0), dtype=dtype, device=device),
    )
    return degree


def degree_reference_from_histogram(deg, dtype, device):
    histogram = deg.to(dtype=dtype, device=device)
    if histogram.numel() == 0:
        return torch.tensor(1.0, dtype=dtype, device=device)
    bins = torch.arange(histogram.numel(), dtype=dtype, device=device)
    total = histogram.sum().clamp_min(1.0)
    return ((histogram * torch.log1p(bins + 1.0)).sum() / total).clamp_min(1.0)
