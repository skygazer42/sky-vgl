import torch
import torch.nn.functional as F
from torch import nn

from vgl.nn.conv._homo import coerce_homo_inputs
from vgl.nn.conv.nnconv import _resolve_edge_features


def _aggregate_messages(messages, col, num_nodes, aggr):
    out = torch.zeros(num_nodes, messages.size(-1), dtype=messages.dtype, device=messages.device)
    if aggr == "max":
        out.fill_(float("-inf"))
        index = col.unsqueeze(-1).expand(-1, messages.size(-1))
        out.scatter_reduce_(0, index, messages, reduce="amax", include_self=True)
        return torch.where(torch.isneginf(out), torch.zeros_like(out), out)

    out.index_add_(0, col, messages)
    if aggr == "mean":
        degree = torch.zeros(num_nodes, dtype=messages.dtype, device=messages.device)
        degree.index_add_(0, col, torch.ones(col.size(0), dtype=messages.dtype, device=messages.device))
        out = out / degree.clamp_min(1).unsqueeze(-1)
    return out


class CGConv(nn.Module):
    def __init__(self, channels, edge_channels=0, aggr="sum", batch_norm=False):
        super().__init__()
        if aggr not in {"sum", "mean", "max"}:
            raise ValueError("CGConv only supports sum, mean, and max aggregation")

        self.channels = channels
        self.out_channels = channels
        self.edge_channels = edge_channels
        self.aggr = aggr
        message_channels = channels * 2 + edge_channels
        self.gate_linear = nn.Linear(message_channels, channels)
        self.message_linear = nn.Linear(message_channels, channels)
        self.norm = nn.BatchNorm1d(channels) if batch_norm else None

    def forward(self, graph_or_x, edge_index=None, edge_attr=None):
        use_graph_data = edge_index is None
        x, edge_index = coerce_homo_inputs(graph_or_x, edge_index, "CGConv")
        row, col = edge_index

        pieces = [x[row], x[col]]
        if self.edge_channels > 0:
            edge_attr = _resolve_edge_features(graph_or_x, use_graph_data, edge_attr, "edge_attr", "CGConv")
            pieces.append(edge_attr)
        z = torch.cat(pieces, dim=-1)

        gate = torch.sigmoid(self.gate_linear(z))
        candidate = F.softplus(self.message_linear(z))
        aggregated = _aggregate_messages(gate * candidate, col, x.size(0), self.aggr)
        out = x + aggregated
        if self.norm is not None:
            out = self.norm(out)
        return out
