import torch


def _num_graphs(graph_index: torch.Tensor) -> int:
    return int(graph_index.max().item()) + 1 if graph_index.numel() > 0 else 0


def global_sum_pool(x: torch.Tensor, graph_index: torch.Tensor) -> torch.Tensor:
    out = torch.zeros((_num_graphs(graph_index), x.size(-1)), dtype=x.dtype, device=x.device)
    out.index_add_(0, graph_index, x)
    return out


def global_mean_pool(x: torch.Tensor, graph_index: torch.Tensor) -> torch.Tensor:
    out = global_sum_pool(x, graph_index)
    counts = torch.bincount(graph_index, minlength=out.size(0)).clamp_min(1).unsqueeze(-1)
    return out / counts


def global_max_pool(x: torch.Tensor, graph_index: torch.Tensor) -> torch.Tensor:
    num_graphs = _num_graphs(graph_index)
    out = torch.full((num_graphs, x.size(-1)), float("-inf"), dtype=x.dtype, device=x.device)
    for idx in range(num_graphs):
        out[idx] = x[graph_index == idx].max(dim=0).values
    return out
