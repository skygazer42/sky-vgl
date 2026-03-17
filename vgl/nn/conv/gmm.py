import torch
from torch import nn

from vgl.nn.conv._homo import coerce_homo_inputs
from vgl.nn.conv.nnconv import _resolve_edge_features


class GMMConv(nn.Module):
    def __init__(self, in_channels, out_channels, dim, kernel_size, root_weight=True):
        super().__init__()
        self.out_channels = out_channels
        self.dim = dim
        self.kernel_size = kernel_size
        self.root_weight = root_weight

        self.message_linear = nn.Linear(in_channels, out_channels * kernel_size, bias=False)
        self.mu = nn.Parameter(torch.randn(kernel_size, dim))
        self.sigma = nn.Parameter(torch.ones(kernel_size, dim))
        self.root_linear = nn.Linear(in_channels, out_channels) if root_weight else None

    def forward(self, graph_or_x, edge_index=None, pseudo=None):
        use_graph_data = edge_index is None
        x, edge_index = coerce_homo_inputs(graph_or_x, edge_index, "GMMConv")
        pseudo = _resolve_edge_features(graph_or_x, use_graph_data, pseudo, "pseudo", "GMMConv")
        row, col = edge_index

        projected = self.message_linear(x).view(x.size(0), self.kernel_size, self.out_channels)
        centered = pseudo.unsqueeze(1) - self.mu.unsqueeze(0)
        gaussian = torch.exp(-0.5 * ((centered / self.sigma.abs().clamp_min(1e-6).unsqueeze(0)) ** 2).sum(dim=-1))
        messages = projected[row] * gaussian.unsqueeze(-1)
        out = torch.zeros(x.size(0), self.kernel_size, self.out_channels, dtype=x.dtype, device=x.device)
        out.index_add_(0, col, messages)
        out = out.mean(dim=1)
        if self.root_linear is not None:
            out = out + self.root_linear(x)
        return out
