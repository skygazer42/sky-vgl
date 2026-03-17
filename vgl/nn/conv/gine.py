import torch
from torch import nn

from vgl.nn.conv._homo import coerce_homo_inputs
from vgl.nn.conv.nnconv import _resolve_edge_features


class GINEConv(nn.Module):
    def __init__(self, in_channels, out_channels, edge_channels, eps=0.0, train_eps=False):
        super().__init__()
        self.out_channels = out_channels
        if train_eps:
            self.eps = nn.Parameter(torch.tensor(float(eps)))
        else:
            self.register_buffer("eps", torch.tensor(float(eps)))
        self.edge_linear = nn.Linear(edge_channels, in_channels)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )

    def forward(self, graph_or_x, edge_index=None, edge_attr=None):
        use_graph_data = edge_index is None
        x, edge_index = coerce_homo_inputs(graph_or_x, edge_index, "GINEConv")
        edge_attr = _resolve_edge_features(graph_or_x, use_graph_data, edge_attr, "edge_attr", "GINEConv")
        row, col = edge_index
        edge_embedding = self.edge_linear(edge_attr)
        messages = torch.relu(x[row] + edge_embedding)
        aggregated = torch.zeros_like(x)
        aggregated.index_add_(0, col, messages)
        return self.mlp((1 + self.eps) * x + aggregated)
