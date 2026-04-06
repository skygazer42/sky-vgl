import torch


def _as_python_int(value) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.detach().cpu().numpy().reshape(()).item())
    return int(value)


def _num_graphs(graph_index: torch.Tensor) -> int:
    return _as_python_int(graph_index.max()) + 1 if graph_index.numel() > 0 else 0


def _dtype_min_value(dtype):
    if torch.empty((), dtype=dtype).is_floating_point():
        return torch.finfo(dtype).min
    return torch.iinfo(dtype).min


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
    out = torch.full(
        (num_graphs, x.size(-1)),
        _dtype_min_value(x.dtype),
        dtype=x.dtype,
        device=x.device,
    )
    if x.numel() == 0 or num_graphs == 0:
        return out
    index = graph_index.view(-1, 1).expand(-1, x.size(-1))
    out.scatter_reduce_(0, index, x, reduce="amax", include_self=True)
    return out
