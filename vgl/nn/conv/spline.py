import math

import torch
from torch import nn

from vgl.nn.conv._homo import coerce_homo_inputs
from vgl.nn.conv.nnconv import _resolve_edge_features


class SplineConv(nn.Module):
    def __init__(self, in_channels, out_channels, dim, kernel_size, root_weight=True, aggr="mean"):
        super().__init__()
        if aggr not in {"sum", "mean"}:
            raise ValueError("SplineConv only supports sum and mean aggregation")

        self.out_channels = out_channels
        self.dim = dim
        self.aggr = aggr
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size,) * dim
        else:
            if len(kernel_size) != dim:
                raise ValueError("SplineConv kernel_size must have one entry per pseudo-coordinate dimension")
            self.kernel_size = tuple(kernel_size)
        self.num_kernels = math.prod(self.kernel_size)
        self.weight = nn.Parameter(torch.randn(self.num_kernels, in_channels, out_channels) / math.sqrt(in_channels))
        self.root_linear = nn.Linear(in_channels, out_channels) if root_weight else None

    def _basis(self, pseudo):
        pseudo = pseudo.clamp(0.0, 1.0)
        bases = []
        for dimension, size in enumerate(self.kernel_size):
            if size == 1:
                bases.append(torch.ones(pseudo.size(0), 1, dtype=pseudo.dtype, device=pseudo.device))
                continue
            grid = torch.linspace(0.0, 1.0, steps=size, dtype=pseudo.dtype, device=pseudo.device)
            distance = (pseudo[:, dimension].unsqueeze(-1) - grid.unsqueeze(0)).abs()
            bases.append((1.0 - distance * (size - 1)).clamp_min(0.0))

        basis = bases[0]
        for current in bases[1:]:
            basis = (basis.unsqueeze(-1) * current.unsqueeze(1)).reshape(pseudo.size(0), -1)
        return basis

    def forward(self, graph_or_x, edge_index=None, pseudo=None):
        use_graph_data = edge_index is None
        x, edge_index = coerce_homo_inputs(graph_or_x, edge_index, "SplineConv")
        pseudo = _resolve_edge_features(graph_or_x, use_graph_data, pseudo, "pseudo", "SplineConv")
        row, col = edge_index

        basis = self._basis(pseudo)
        messages = torch.einsum("ek,ei,kio->eo", basis, x[row], self.weight)
        out = torch.zeros(x.size(0), self.out_channels, dtype=x.dtype, device=x.device)
        out.index_add_(0, col, messages)
        if self.aggr == "mean":
            degree = torch.zeros(x.size(0), dtype=x.dtype, device=x.device)
            degree.index_add_(0, col, torch.ones(col.size(0), dtype=x.dtype, device=x.device))
            out = out / degree.clamp_min(1).unsqueeze(-1)
        if self.root_linear is not None:
            out = out + self.root_linear(x)
        return out
