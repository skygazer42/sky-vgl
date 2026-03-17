import torch
import torch.nn.functional as F
from torch import nn

from vgl.nn.conv._homo import coerce_homo_inputs
from vgl.nn.conv.nnconv import _resolve_edge_features


class GatedGCNConv(nn.Module):
    def __init__(self, in_channels, out_channels, edge_channels, dropout=0.0, residual=True):
        super().__init__()
        if dropout < 0.0 or dropout > 1.0:
            raise ValueError("GatedGCNConv requires dropout to be in [0, 1]")

        self.out_channels = out_channels
        self.dropout = dropout
        self.residual = residual and in_channels == out_channels

        self.root_linear = nn.Linear(in_channels, out_channels)
        self.message_linear = nn.Linear(in_channels, out_channels, bias=False)
        self.src_gate_linear = nn.Linear(in_channels, out_channels, bias=False)
        self.dst_gate_linear = nn.Linear(in_channels, out_channels, bias=False)
        self.edge_gate_linear = nn.Linear(edge_channels, out_channels, bias=False)

    def forward(self, graph_or_x, edge_index=None, edge_attr=None):
        use_graph_data = edge_index is None
        x, edge_index = coerce_homo_inputs(graph_or_x, edge_index, "GatedGCNConv")
        edge_attr = _resolve_edge_features(graph_or_x, use_graph_data, edge_attr, "edge_attr", "GatedGCNConv")
        row, col = edge_index

        gates = torch.sigmoid(
            self.src_gate_linear(x[row]) + self.dst_gate_linear(x[col]) + self.edge_gate_linear(edge_attr)
        )
        gates = F.dropout(gates, p=self.dropout, training=self.training)
        messages = self.message_linear(x[row]) * gates

        out = torch.zeros(x.size(0), self.out_channels, dtype=x.dtype, device=x.device)
        out.index_add_(0, col, messages)

        normalizer = torch.zeros(x.size(0), self.out_channels, dtype=x.dtype, device=x.device)
        normalizer.index_add_(0, col, gates)
        out = out / normalizer.clamp_min(1e-6)
        out = out + self.root_linear(x)
        if self.residual:
            out = out + x
        return out
