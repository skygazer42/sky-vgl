import torch
from torch import nn

from vgl.nn.conv._homo import coerce_homo_inputs, mean_propagate


class ChebConv(nn.Module):
    def __init__(self, in_channels, out_channels, k=2):
        super().__init__()
        self.out_channels = out_channels
        self.k = k
        self.linear = nn.Linear(in_channels * (k + 1), out_channels)

    def forward(self, graph_or_x, edge_index=None):
        x, edge_index = coerce_homo_inputs(graph_or_x, edge_index, "ChebConv")
        terms = [x]
        if self.k >= 1:
            terms.append(mean_propagate(x, edge_index))
        while len(terms) <= self.k:
            propagated = mean_propagate(terms[-1], edge_index)
            terms.append(2 * propagated - terms[-2])
        return self.linear(torch.cat(terms, dim=-1))
