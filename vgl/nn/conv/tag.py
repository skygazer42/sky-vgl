import torch
from torch import nn

from vgl.nn.conv._homo import coerce_homo_inputs, mean_propagate


class TAGConv(nn.Module):
    def __init__(self, in_channels, out_channels, k=2):
        super().__init__()
        self.out_channels = out_channels
        self.k = k
        self.linear = nn.Linear(in_channels * (k + 1), out_channels)

    def forward(self, graph_or_x, edge_index=None):
        x, edge_index = coerce_homo_inputs(graph_or_x, edge_index, "TAGConv")
        features = [x]
        propagated = x
        for _ in range(self.k):
            propagated = mean_propagate(propagated, edge_index)
            features.append(propagated)
        return self.linear(torch.cat(features, dim=-1))
